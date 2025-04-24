# --- START OF FILE ppo_agentsv5.py ---

# --- Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import random
import yaml
import os
import sys
import time
import logging
import datetime
from typing import List, Tuple, Dict, Any, Callable, Optional, Union, Generator
from dataclasses import dataclass, field, fields
import argparse
from enum import Enum

# Rich and Tqdm for visualization
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID,
)

# --- Import Critic Components ---
try:
    from daedalus.critics import level_critic as actual_critic
    from daedalus.critics.critic_approximator import (
        CriticApproximatorMLP,
        CriticConfig,
        configure_critic_from_yaml,
    )
except ImportError as e:
    print(f"Error importing Daedalus components: {e}")
    print("Please ensure 'daedalus' package structure is correct or PYTHONPATH is set.")
    print(
        "Attempting relative import based on assumed structure (agents/ and critics/ siblings)..."
    )
    try:
        critics_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "critics")
        )
        base_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, critics_path)
        sys.path.insert(0, base_dir_path)
        from daedalus.critics import level_critic as actual_critic
        from critic_approximator import (
            CriticApproximatorMLP,
            CriticConfig,
            configure_critic_from_yaml,
        )

        print("Relative import successful.")
    except ImportError as inner_e:
        print(f"Relative import failed: {inner_e}")
        print("Cannot proceed without critic components.")
        sys.exit(1)


# --- Mock Entry Enum ---
class Entry(Enum):
    TOP = 0
    LEFT = 1
    BOTTOM = 2
    RIGHT = 3


# --- Helper Class for Reward Normalization ---
class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 0 else 1  # Handle scalar input
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


# --- Model Definition ---
class PolicyNetworkEncoder(nn.Module):
    """Encodes the map and optionally hero state into a latent representation."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (12, 12),
        output_size: int = 1024,
        hero_tensor_size: int = 5,
        hidden_dims: List[int] = [1024, 2048],
        use_hero_tensor: bool = True,  # Flag to control hero tensor usage
    ):
        super().__init__()
        self.input_size = self.input_size_x, self.input_size_y = input_size
        self.map_flat_dim = self.input_size_x * self.input_size_y
        self.hero_tensor_size = hero_tensor_size
        self.use_hero_tensor = use_hero_tensor

        # Adjust input_dim based on flag
        self.input_dim = self.map_flat_dim
        if self.use_hero_tensor:
            self.input_dim += self.hero_tensor_size

        self.output_size = output_size
        self.hidden_dims = hidden_dims

        layers = []
        current_dim = self.input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, self.output_size))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, hero_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for the NN encoder."""
        x_flat = torch.flatten(x, start_dim=1)
        if self.use_hero_tensor:
            if hero_tensor is None:
                raise ValueError("hero_tensor is required when use_hero_tensor=True")
            hero_tensor = hero_tensor.float()
            combined = torch.cat([x_flat, hero_tensor], dim=1)
        else:
            combined = x_flat  # Only use map if hero is disabled

        latent = self.net(combined)
        return latent


class PolicyNetworkDecoder(nn.Module):
    """Decodes latent representation into action probabilities or state values."""

    def __init__(
        self,
        input_size: int = 1024,
        hidden_sizes: List[int] = [512, 256],
        output_size: int = 7,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        layers = []
        current_dim = input_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOPolicyNetwork(nn.Module):
    """Combined PPO Actor Network using Encoder-Decoder structure."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (12, 12),
        encoder_output_size: int = 1024,
        decoder_hidden_sizes: List[int] = [512, 256],
        action_dim: int = 7,
        hero_tensor_size: int = 5,
        encoder_hidden_dims: List[int] = [512, 512],
        use_hero_tensor_in_encoder: bool = True,  # Flag ARG
    ):
        super().__init__()
        self.use_hero_tensor_in_encoder = use_hero_tensor_in_encoder  # STORE FLAG
        self.encoder = PolicyNetworkEncoder(
            input_size=input_size,
            output_size=encoder_output_size,
            hero_tensor_size=hero_tensor_size,
            hidden_dims=encoder_hidden_dims,
            use_hero_tensor=use_hero_tensor_in_encoder,  # PASS FLAG
        )
        self.actor_decoder = PolicyNetworkDecoder(
            input_size=encoder_output_size,
            hidden_sizes=decoder_hidden_sizes,
            output_size=action_dim,
        )

    def forward(
        self,
        map_tensor: torch.Tensor,
        hero_tensor: Optional[torch.Tensor] = None,  # Make optional
    ) -> torch.Tensor:
        """Forward pass returning action logits."""
        if map_tensor.dim() == 3:
            map_tensor = map_tensor.unsqueeze(1)

        # Pass hero_tensor only if needed by encoder
        latent = self.encoder(
            map_tensor, hero_tensor if self.use_hero_tensor_in_encoder else None
        )
        action_logits = self.actor_decoder(latent)
        return action_logits


class ValueNetwork(nn.Module):
    """PPO Critic Network using Encoder-Decoder structure."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (12, 12),
        encoder_output_size: int = 1024,
        decoder_hidden_sizes: List[int] = [512, 256],
        hero_tensor_size: int = 5,
        encoder_hidden_dims: List[int] = [512, 512],
        use_hero_tensor_in_encoder: bool = True,  # Flag ARG
    ):
        super().__init__()
        self.use_hero_tensor_in_encoder = use_hero_tensor_in_encoder  # STORE FLAG
        self.encoder = PolicyNetworkEncoder(
            input_size=input_size,
            output_size=encoder_output_size,
            hero_tensor_size=hero_tensor_size,
            hidden_dims=encoder_hidden_dims,
            use_hero_tensor=use_hero_tensor_in_encoder,  # PASS FLAG
        )
        self.value_decoder = PolicyNetworkDecoder(
            input_size=encoder_output_size,
            hidden_sizes=decoder_hidden_sizes,
            output_size=1,  # Output a single value
        )

    def forward(
        self,
        map_tensor: torch.Tensor,
        hero_tensor: Optional[torch.Tensor] = None,  # Make optional
    ) -> torch.Tensor:
        """Forward pass returning state value prediction."""
        if map_tensor.dim() == 3:
            map_tensor = map_tensor.unsqueeze(1)

        # Pass hero_tensor only if needed by encoder
        latent = self.encoder(
            map_tensor, hero_tensor if self.use_hero_tensor_in_encoder else None
        )
        value = self.value_decoder(latent)
        return value


# --- Configuration ---
@dataclass
class PPOConfig:
    mode: str = "narrow"
    map_size: Tuple[int, int] = (12, 12)
    hero_tensor_size: int = 5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_episodes: int = 1000
    episode_length: int = 256
    num_envs: int = 128
    n_steps_per_rollout: int = 64
    num_epochs_per_update: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    temperature: float = 0.5
    save_checkpoint_freq: int = 100
    print_maps_freq: int = 50
    checkpoint_dir: str = "ppo_checkpoints_episodic_metrics"
    run_name: Optional[str] = None
    log_file_name: str = "ppo_training_log.log"
    initial_map_walk_steps: int = 36
    initial_map_empty_prob: float = 0.75
    reward_baseline_alpha: float = 0.05
    encoder_output_size: int = 1024
    decoder_hidden_sizes: List[int] = field(default_factory=lambda: [512, 256])
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 512])
    use_neural_critic: bool = True
    neural_critic_checkpoint_path: str = "critics/neural_critic_checkpoints/critic_MLP_20250422_113333/latest_checkpoint.pth"
    normalize_rewards: bool = True  # <<< ADDED config option
    reward_clip_value: float = 10.0  # <<< ADDED config option

    batch_size: int = field(init=False)
    action_dim: int = field(init=False)

    def __post_init__(self):
        self.batch_size = self.num_envs * self.n_steps_per_rollout
        if self.batch_size % self.minibatch_size != 0:
            print(
                f"Warning: Minibatch size ({self.minibatch_size}) is not a divisor of the total batch size ({self.batch_size})."
            )
        n_tiles = self.map_size[0] * self.map_size[1]
        mode_actions = {"narrow": 7, "turtle": 11, "wide": 7 * n_tiles}
        if self.mode not in mode_actions:
            raise ValueError(f"Unknown mode: {self.mode}")
        self.action_dim = mode_actions[self.mode]
        if self.run_name is None:
            critic_type = "NeuralCritic" if self.use_neural_critic else "ActualCritic"
            self.run_name = (
                f"ppo_{self.mode}_{critic_type}_ep_{time.strftime('%Y%m%d_%H%M%S')}"
            )
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.run_name)
        self.log_file_name = f"ppo_training_{self.run_name}.log"


def configure_from_yaml(yaml_path: str) -> PPOConfig:
    try:
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        if not isinstance(yaml_config, dict):
            raise TypeError(f"YAML file {yaml_path} did not parse into a dictionary.")
        valid_keys = {f.name for f in fields(PPOConfig) if f.init}
        filtered_config = {k: v for k, v in yaml_config.items() if k in valid_keys}
        if "use_neural_critic" in filtered_config:
            val = filtered_config["use_neural_critic"]
            filtered_config["use_neural_critic"] = str(val).lower() in [
                "true",
                "1",
                "yes",
                "y",
            ]
        if "normalize_rewards" in filtered_config:  # Handle new boolean
            val = filtered_config["normalize_rewards"]
            filtered_config["normalize_rewards"] = str(val).lower() in [
                "true",
                "1",
                "yes",
                "y",
            ]
        if "map_size" in filtered_config and isinstance(
            filtered_config["map_size"], list
        ):
            filtered_config["map_size"] = tuple(filtered_config["map_size"])
        return PPOConfig(**filtered_config)
    except FileNotFoundError:
        print(
            f"Warning: YAML config file not found at {yaml_path}. Using default PPOConfig."
        )
        return PPOConfig()
    except Exception as e:
        print(
            f"Error loading YAML config from {yaml_path}: {e}. Using default PPOConfig."
        )
        return PPOConfig()


# --- Environment Utilities ---
def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def initialize_map_hero(
    config: PPOConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    map_tensor = torch.zeros(config.map_size, dtype=torch.int64)
    size_x, size_y = config.map_size
    if size_x <= 0 or size_y <= 0:
        return (
            map_tensor,
            torch.zeros(config.hero_tensor_size, dtype=torch.int64),
            (0, 0),
        )
    start_pos = (random.randint(0, size_x - 1), random.randint(0, size_y - 1))
    current_pos = start_pos
    for _ in range(config.initial_map_walk_steps):
        tile_value = (
            (1 if random.random() < 0.9 else 6)
            if random.random() < config.initial_map_empty_prob
            else random.randint(2, 5)
        )
        map_tensor[current_pos] = tile_value
        dx, dy = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        next_x = max(0, min(size_x - 1, current_pos[0] + dx))
        next_y = max(0, min(size_y - 1, current_pos[1] + dy))
        current_pos = (next_x, next_y)
    if map_tensor[start_pos] == 0:
        map_tensor[start_pos] = 1
    hero_tensor = torch.tensor(
        [
            random.randint(1, 10),
            random.choice([0, 1]),
            random.choice([0, 1]),
            random.choice([e.value for e in Entry]),
            random.randint(0, 10),
        ],
        dtype=torch.int64,
    )
    return map_tensor, hero_tensor, start_pos


# --- Environment Simulation ---
class BatchedEnvSimulator:
    def __init__(self, config: PPOConfig, critic_function_to_use: Callable):
        self.config = config
        self.num_envs = config.num_envs
        self.map_size_x, self.map_size_y = config.map_size
        self.critic_func = critic_function_to_use
        self.device = get_device(config.device)
        self.maps = torch.zeros(
            (self.num_envs, self.map_size_x, self.map_size_y),
            dtype=torch.int64,
            device=self.device,
        )
        self.heroes = torch.zeros(
            (self.num_envs, config.hero_tensor_size),
            dtype=torch.int64,
            device=self.device,
        )
        self.agent_positions = [(0, 0)] * self.num_envs

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial_maps_list_cpu = []
        for i in range(self.num_envs):
            map_i_cpu, hero_i_cpu, pos_i = initialize_map_hero(self.config)
            self.maps[i] = map_i_cpu.to(self.device)
            self.heroes[i] = hero_i_cpu.to(self.device)
            self.agent_positions[i] = pos_i
            initial_maps_list_cpu.append(map_i_cpu)
        initial_maps_cpu_batch = torch.stack(initial_maps_list_cpu)
        map_obs, hero_obs = self._get_observation()
        return map_obs, hero_obs, initial_maps_cpu_batch

    def _get_observation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        map_obs = self.maps.unsqueeze(1).float()
        hero_obs = self.heroes.float()
        return map_obs, hero_obs

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, List[Dict]
    ]:
        actions_device = actions.to(self.device)
        for i in range(self.num_envs):
            action = actions_device[i].item()
            pos_x, pos_y = self.agent_positions[i]
            current_map = self.maps[i]
            mode = self.config.mode
            if mode == "narrow":
                if 0 <= action <= 6:
                    current_map[pos_x, pos_y] = action
                dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                next_x = max(0, min(self.map_size_x - 1, pos_x + dx))
                next_y = max(0, min(self.map_size_y - 1, pos_y + dy))
                self.agent_positions[i] = (next_x, next_y)
            elif mode == "turtle":
                if 0 <= action <= 6:
                    current_map[pos_x, pos_y] = action
                elif action == 7:
                    pos_x = max(0, pos_x - 1)
                elif action == 8:
                    pos_y = max(0, pos_y - 1)
                elif action == 9:
                    pos_x = min(self.map_size_x - 1, pos_x + 1)
                elif action == 10:
                    pos_y = min(self.map_size_y - 1, pos_y + 1)
                self.agent_positions[i] = (pos_x, pos_y)
            elif mode == "wide":
                num_tile_types = 7
                n_tiles_total = self.map_size_x * self.map_size_y
                tile_type = action // n_tiles_total
                flat_index = action % n_tiles_total
                target_x = flat_index // self.map_size_y
                target_y = flat_index % self.map_size_y
                if 0 <= tile_type < num_tile_types:
                    if (
                        0 <= target_x < self.map_size_x
                        and 0 <= target_y < self.map_size_y
                    ):
                        current_map[target_x, target_y] = tile_type
            else:
                raise ValueError(f"Unknown mode in step: {mode}")

        rewards = self.critic_func(self.maps, self.heroes)  # Pass device tensors
        rewards = rewards.to(self.device).float().squeeze()
        if rewards.shape != (self.num_envs,):
            # Handle scalar reward if num_envs=1
            if self.num_envs == 1 and rewards.ndim == 0:
                rewards = rewards.unsqueeze(0)
            else:
                raise ValueError(
                    f"Critic function returned unexpected rewards shape: {rewards.shape}. Expected ({self.num_envs},)"
                )

        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        next_map_obs, next_hero_obs = self._get_observation()
        infos = [{} for _ in range(self.num_envs)]
        return (next_map_obs, next_hero_obs), rewards, dones, infos


# --- PPO Memory ---
class PPOMemory:
    def __init__(
        self,
        num_envs: int,
        n_steps: int,
        map_shape: Tuple[int, int],
        hero_shape: int,
        device: torch.device,
    ):
        self.n_steps, self.num_envs, self.device = n_steps, num_envs, device
        map_c, map_h, map_w = 1, map_shape[0], map_shape[1]
        self.maps = torch.zeros(
            (n_steps, num_envs, map_c, map_h, map_w), dtype=torch.float32, device=device
        )
        self.heroes = torch.zeros(
            (n_steps, num_envs, hero_shape), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(
            (n_steps, num_envs), dtype=torch.int64, device=device
        )
        self.log_probs = torch.zeros(
            (n_steps, num_envs), dtype=torch.float32, device=device
        )
        self.rewards = torch.zeros(
            (n_steps, num_envs), dtype=torch.float32, device=device
        )
        self.dones = torch.zeros((n_steps, num_envs), dtype=torch.bool, device=device)
        self.values = torch.zeros(
            (n_steps, num_envs), dtype=torch.float32, device=device
        )
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None
        self.ptr = 0

    def add(
        self,
        map_obs: torch.Tensor,
        hero_obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ):
        if self.ptr >= self.n_steps:
            raise IndexError("Memory buffer overflow.")
        self.maps[self.ptr] = map_obs.to(self.device)
        self.heroes[self.ptr] = hero_obs.to(self.device)
        self.actions[self.ptr] = action.to(self.device)
        self.log_probs[self.ptr] = log_prob.to(self.device)
        self.rewards[self.ptr] = reward.to(
            self.device
        )  # Store possibly normalized reward
        self.dones[self.ptr] = done.to(self.device)
        self.values[self.ptr] = value.to(self.device)
        self.ptr += 1

    def compute_gae_returns(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ):
        if self.ptr != self.n_steps:
            print(
                f"Warning: Computing GAE on incomplete buffer (ptr={self.ptr}, n_steps={self.n_steps})."
            )
        if self.ptr == 0:
            self.advantages = torch.zeros_like(self.rewards[: self.ptr])
            self.returns = torch.zeros_like(self.rewards[: self.ptr])
            return
        last_value = last_value.to(self.device).squeeze()
        if last_value.shape != (self.num_envs,):
            if self.num_envs == 1 and last_value.ndim == 0:
                last_value = last_value.unsqueeze(0)
            else:
                raise ValueError(
                    f"Expected last_value shape ({self.num_envs},), got {last_value.shape}"
                )

        last_gae_lam = 0
        num_steps_filled = self.ptr
        self.advantages = torch.zeros(
            (num_steps_filled, self.num_envs), dtype=torch.float32, device=self.device
        )
        for t in reversed(range(num_steps_filled)):
            if t == num_steps_filled - 1:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1].float()
                next_values = self.values[t + 1]
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        self.returns = self.advantages + self.values[:num_steps_filled]

    def get_minibatches(
        self, batch_size: int, minibatch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        if self.advantages is None or self.returns is None:
            raise ValueError("Advantages/returns must be computed first.")
        num_steps_filled = self.advantages.shape[0]
        num_transitions = num_steps_filled * self.num_envs
        if num_transitions == 0:
            return
        actual_batch_size = min(batch_size, num_transitions)
        actual_minibatch_size = min(minibatch_size, actual_batch_size)
        if num_transitions < batch_size:
            print(
                f"Warning: Requested batch_size {batch_size} > available transitions {num_transitions}. Using {actual_batch_size}."
            )
        if minibatch_size > batch_size:
            print(
                f"Warning: Minibatch size ({minibatch_size}) > effective batch size ({actual_batch_size}). Using {actual_minibatch_size}."
            )

        flat_maps = self.maps[:num_steps_filled].reshape(
            num_transitions, *self.maps.shape[2:]
        )
        flat_heroes = self.heroes[:num_steps_filled].reshape(num_transitions, -1)
        flat_actions = self.actions[:num_steps_filled].reshape(-1)
        flat_log_probs = self.log_probs[:num_steps_filled].reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values[:num_steps_filled].reshape(-1)

        indices = torch.randperm(num_transitions).to(self.device)

        for start_idx in range(0, actual_batch_size, actual_minibatch_size):
            end_idx = min(start_idx + actual_minibatch_size, actual_batch_size)
            if start_idx >= end_idx:
                continue
            mb_indices = indices[start_idx:end_idx]
            if len(mb_indices) == 0:
                continue
            yield {
                "maps": flat_maps[mb_indices],
                "heroes": flat_heroes[mb_indices],
                "actions": flat_actions[mb_indices],
                "old_log_probs": flat_log_probs[mb_indices],
                "advantages": flat_advantages[mb_indices],
                "returns": flat_returns[mb_indices],
                "old_values": flat_values[mb_indices],
            }

    def clear(self):
        self.ptr = 0
        self.advantages = None
        self.returns = None


# --- PPO Agent ---
class PPOAgent:
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = get_device(config.device)

        # Determine if hero tensor should be used based on whether neural critic is active
        # (Assuming neural critic only uses map, aligning agent state with reward source)
        use_hero_in_nets = not config.use_neural_critic

        self.actor = PPOPolicyNetwork(
            input_size=config.map_size,
            encoder_output_size=config.encoder_output_size,
            decoder_hidden_sizes=config.decoder_hidden_sizes,
            action_dim=config.action_dim,
            hero_tensor_size=config.hero_tensor_size,
            encoder_hidden_dims=config.encoder_hidden_dims,
            use_hero_tensor_in_encoder=use_hero_in_nets,  # PASS FLAG
        ).to(self.device)
        self.critic = ValueNetwork(
            input_size=config.map_size,
            encoder_output_size=config.encoder_output_size,
            decoder_hidden_sizes=config.decoder_hidden_sizes,
            hero_tensor_size=config.hero_tensor_size,
            encoder_hidden_dims=config.encoder_hidden_dims,
            use_hero_tensor_in_encoder=use_hero_in_nets,  # PASS FLAG
        ).to(self.device)

        all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(all_params, lr=config.learning_rate, eps=1e-5)
        self.total_steps_interacted = 0
        self.total_updates_performed = 0

    def select_action(
        self, map_obs: torch.Tensor, hero_obs: torch.Tensor, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            # Determine correct inputs based on network configuration
            hero_input_actor = (
                hero_obs if self.actor.use_hero_tensor_in_encoder else None
            )
            hero_input_critic = (
                hero_obs if self.critic.use_hero_tensor_in_encoder else None
            )

            action_logits = self.actor(map_obs, hero_input_actor)
            value = self.critic(map_obs, hero_input_critic).squeeze(-1)

            if temperature > 0:
                scaled_logits = action_logits / max(temperature, 1e-8)
            else:
                scaled_logits = action_logits
            probs = F.softmax(scaled_logits, dim=-1)
            dist = Categorical(probs=probs)
            if temperature > 0:
                action = dist.sample()
            else:
                action = torch.argmax(probs, dim=-1)
            log_prob = dist.log_prob(action)
        self.actor.train()
        self.critic.train()
        return action, log_prob, value

    def evaluate_actions(
        self, map_obs: torch.Tensor, hero_obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Determine correct inputs based on network configuration
        hero_input_actor = hero_obs if self.actor.use_hero_tensor_in_encoder else None
        hero_input_critic = hero_obs if self.critic.use_hero_tensor_in_encoder else None

        action_logits = self.actor(map_obs, hero_input_actor)
        value = self.critic(map_obs, hero_input_critic).squeeze(-1)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs=probs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy

    def update(self, memory: PPOMemory) -> Dict[str, float]:
        if memory.advantages is None or memory.returns is None:
            raise RuntimeError(
                "memory.compute_gae_returns() must be called before update()"
            )
        all_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        for _ in range(self.config.num_epochs_per_update):
            minibatch_generator = memory.get_minibatches(
                self.config.batch_size, self.config.minibatch_size
            )
            for batch in minibatch_generator:
                mb_maps, mb_heroes = batch["maps"], batch["heroes"]
                mb_actions, mb_old_log_probs = batch["actions"], batch["old_log_probs"]
                mb_advantages, mb_returns = batch["advantages"], batch["returns"]
                # mb_old_values = batch["old_values"]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )
                new_log_probs, new_values, entropy = self.evaluate_actions(
                    mb_maps, mb_heroes, mb_actions
                )

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * F.mse_loss(new_values, mb_returns)
                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    - self.config.entropy_coef * entropy_loss
                    + self.config.vf_coef * value_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                all_metrics["policy_loss"].append(policy_loss.item())
                all_metrics["value_loss"].append(value_loss.item())
                all_metrics["entropy"].append(entropy_loss.item())
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1.0) > self.config.clip_epsilon).float()
                    ).item()
                    all_metrics["approx_kl"].append(approx_kl)
                    all_metrics["clip_fraction"].append(clip_fraction)

        self.total_updates_performed += 1
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items() if v}
        return avg_metrics

    def save_checkpoint(self, path: str, episode: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + ".tmp"
        try:
            checkpoint = {
                "episode": episode,
                "total_steps_interacted": self.total_steps_interacted,
                "total_updates_performed": self.total_updates_performed,
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                # Add reward RMS state if needed for resuming normalization accurately
                # "reward_rms_mean": self.reward_rms.mean, # Assuming trainer passes it
                # "reward_rms_var": self.reward_rms.var,
                # "reward_rms_count": self.reward_rms.count,
            }
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, path)
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint to {path}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            print(
                f"[Warning] Checkpoint file not found at {path}. Starting from episode 0."
            )
            return 0
        try:
            checkpoint = torch.load(path, map_location=self.device)
            chk_config = checkpoint.get("config")
            if isinstance(chk_config, PPOConfig):
                mismatched_params = []
                if chk_config.action_dim != self.config.action_dim:
                    mismatched_params.append(
                        f"ActionDim (Chk: {chk_config.action_dim}, Cur: {self.config.action_dim})"
                    )
                if chk_config.mode != self.config.mode:
                    mismatched_params.append(
                        f"Mode (Chk: {chk_config.mode}, Cur: {self.config.mode})"
                    )
                if chk_config.map_size != self.config.map_size:
                    mismatched_params.append(
                        f"MapSize (Chk: {chk_config.map_size}, Cur: {self.config.map_size})"
                    )
                # Check if hero tensor usage mismatches between checkpoint and current config
                chk_use_hero = not chk_config.use_neural_critic
                cur_use_hero = not self.config.use_neural_critic
                if chk_use_hero != cur_use_hero:
                    mismatched_params.append(
                        f"HeroTensorUsage (Chk: {chk_use_hero}, Cur: {cur_use_hero})"
                    )

                if mismatched_params:
                    print(f"[Warning] Config mismatch detected! Checkpoint vs Current:")
                    for param in mismatched_params:
                        print(f"  - {param}")
                    print("Loading weights anyway, but behavior may be unpredictable.")
            else:
                print(
                    "[Warning] Checkpoint does not contain PPOConfig or it's invalid."
                )

            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.total_steps_interacted = checkpoint.get("total_steps_interacted", 0)
            self.total_updates_performed = checkpoint.get("total_updates_performed", 0)
            # Load reward RMS state if saved
            # if "reward_rms_mean" in checkpoint:
            #    self.reward_rms.mean = checkpoint["reward_rms_mean"]
            #    self.reward_rms.var = checkpoint["reward_rms_var"]
            #    self.reward_rms.count = checkpoint["reward_rms_count"]

            start_episode = checkpoint.get("episode", -1) + 1
            print(
                f"Checkpoint loaded from {path}. Resuming from episode {start_episode}."
            )
            print(f"  -> Total Steps Interacted: {self.total_steps_interacted:,}")
            print(f"  -> Total Updates Performed: {self.total_updates_performed:,}")
            return start_episode
        except KeyError as e:
            print(
                f"[ERROR] Missing key in checkpoint {path}: {e}. Starting from scratch."
            )
            return 0
        except Exception as e:
            print(
                f"[ERROR] Failed to load checkpoint from {path}: {e}. Starting from scratch."
            )
            return 0


# --- Visualization ---
def print_map(console: Console, map_tensor: torch.Tensor, title: str = "Generated Map"):
    if map_tensor.ndim == 4 and map_tensor.shape[0] == 1 and map_tensor.shape[1] == 1:
        map_tensor = map_tensor.squeeze(0).squeeze(0)
    elif map_tensor.ndim == 3 and map_tensor.shape[0] == 1:
        map_tensor = map_tensor.squeeze(0)
    elif map_tensor.ndim == 3 and map_tensor.shape[1] == 1:
        map_tensor = map_tensor[0].squeeze(0)
    elif map_tensor.ndim == 2:
        pass
    else:
        console.print(
            f"[red]Error: Invalid map tensor shape for printing: {map_tensor.shape}.[/red]"
        )
        return
    if map_tensor.device != torch.device("cpu"):
        map_tensor = map_tensor.cpu()
    try:
        map_np = map_tensor.numpy().astype(int)
    except Exception as e:
        console.print(f"[red]Error converting map tensor to numpy: {e}[/red]")
        return
    map_size_x, map_size_y = map_np.shape
    colors = {
        0: "dim grey50",
        1: "white",
        6: "bright_green",
        2: "bright_red",
        3: "red",
        4: "dark_red",
        5: "red3",
    }
    default_color = "magenta"
    char_width = 2
    table = Table(
        title=title,
        show_header=False,
        show_edge=False,
        box=None,
        padding=(0, 0),
        expand=False,
    )
    for _ in range(map_size_y):
        table.add_column(justify="center", width=char_width, style="dim")
    for r in range(map_size_x):
        row_cells = []
        for tile in map_np[r]:
            color = colors.get(tile, default_color)
            cell_text = f"[{color}]{tile:>{char_width - 1}} [/]"
            row_cells.append(cell_text)
        table.add_row(*row_cells)
    console.print(table)


# --- Training Orchestrator ---
class PPOTrainer:
    def __init__(self, config: PPOConfig, base_critic_func: Callable):
        self.config = config
        self.base_critic_func = base_critic_func
        self.device = get_device(config.device)
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self._setup_logging()

        self.neural_critic_model: Optional[CriticApproximatorMLP] = None
        self.critic_func_to_use: Callable = self.base_critic_func

        if config.use_neural_critic:
            ckpt_path = config.neural_critic_checkpoint_path
            self.logger.info(f"Attempting to load Neural Critic from: {ckpt_path}")
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.abspath(ckpt_path)
            try:
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(
                        f"Neural critic checkpoint not found at: {ckpt_path}"
                    )
                critic_checkpoint = torch.load(
                    ckpt_path, map_location=self.device, weights_only=False
                )
                if not isinstance(critic_checkpoint, dict):
                    raise TypeError("Loaded critic checkpoint is not a dictionary.")
                if "config" not in critic_checkpoint:
                    raise KeyError("Critic checkpoint missing 'config'.")
                if "model_state_dict" not in critic_checkpoint:
                    raise KeyError("Critic checkpoint missing 'model_state_dict'.")
                critic_config_data = critic_checkpoint["config"]
                if isinstance(critic_config_data, dict):
                    self.logger.warning("Critic config loaded as dict, reconstructing.")
                    try:
                        critic_config = CriticConfig(**critic_config_data)
                    except Exception as config_e:
                        raise TypeError(
                            f"Failed to reconstruct CriticConfig from dict: {config_e}"
                        )
                elif isinstance(critic_config_data, CriticConfig):
                    critic_config = critic_config_data
                else:
                    raise TypeError(
                        f"Expected CriticConfig or dict, got {type(critic_config_data)}"
                    )
                self.logger.info(
                    f"Loaded critic config: MapSize={critic_config.map_size}, Hidden={critic_config.mlp_hidden_sizes}"
                )

                critic_config.map_size = tuple(critic_config.map_size)  # Ensure tuple
                if critic_config.map_size != self.config.map_size:
                    msg = f"CRITICAL MAP SIZE MISMATCH! PPO: {self.config.map_size}, Critic: {critic_config.map_size}."
                    self.logger.error(msg)
                    self.console.print(f"[bold red]{msg}[/bold red]")
                    raise ValueError(msg)

                self.neural_critic_model = CriticApproximatorMLP(
                    input_size=critic_config.map_size[0] * critic_config.map_size[1],
                    hidden_sizes=critic_config.mlp_hidden_sizes,
                    output_size=1,
                    dropout_prob=critic_config.mlp_dropout_prob,
                ).to(self.device)
                self.neural_critic_model.load_state_dict(
                    critic_checkpoint["model_state_dict"]
                )
                self.neural_critic_model.eval()
                self.logger.info(
                    f"[bold green]Successfully loaded Neural Critic model.[/bold green]"
                )

                def neural_critic_wrapper(
                    map_tensor: torch.Tensor, hero_tensor: torch.Tensor
                ) -> torch.Tensor:
                    map_tensor = map_tensor.to(self.device)
                    if map_tensor.dim() == 3:
                        map_tensor = map_tensor.unsqueeze(1)
                    map_tensor = map_tensor.float()
                    with torch.no_grad():
                        scores = self.neural_critic_model(map_tensor)
                    return scores.squeeze(-1)

                self.critic_func_to_use = neural_critic_wrapper
                self.console.print(
                    "Neural Critic [bold green]loaded and active[/bold green]."
                )
            except Exception as e:
                self.logger.error(
                    f"[bold red]Failed to load Neural Critic:[/bold red] {e}",
                    exc_info=True,
                )
                self.console.print(
                    f"[bold red]Error loading Neural Critic:[/bold red] {e}. Falling back to symbolic."
                )
                self.critic_func_to_use = self.base_critic_func
                self.neural_critic_model = None
        else:
            self.logger.info(
                "Using original symbolic critic function (neural critic disabled)."
            )
            self.console.print(
                "Using original symbolic critic ([dim]neural critic disabled[/dim])."
            )
            self.critic_func_to_use = self.base_critic_func

        self.env = BatchedEnvSimulator(config, self.critic_func_to_use)
        self.agent = PPOAgent(config)
        self.memory = PPOMemory(
            config.num_envs,
            config.n_steps_per_rollout,
            config.map_size,
            config.hero_tensor_size,
            self.device,
        )
        self.start_episode = 0
        self.reward_baseline = 0.0
        self.is_first_episode_for_baseline = True
        self.reward_rms = RunningMeanStd(shape=())  # Initialize reward normalizer
        # Load RMS state if loading checkpoint (add this to agent.load_checkpoint logic if needed)

    def _setup_logging(self):
        log_dir = self.config.checkpoint_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, self.config.log_file_name)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info(f"--- Logging started for run: {self.config.run_name} ---")
        self.logger.info(
            f"PPO Config: Mode={self.config.mode}, MapSize={self.config.map_size}, Device={self.config.device}, Seed={self.config.seed}"
        )
        self.logger.info(
            f"Training: Episodes={self.config.num_episodes}, EpLength={self.config.episode_length}, NumEnvs={self.config.num_envs}"
        )
        self.logger.info(
            f"PPO: RolloutSteps={self.config.n_steps_per_rollout}, Epochs={self.config.num_epochs_per_update}, Minibatch={self.config.minibatch_size}"
        )
        self.logger.info(
            f"Hyperparams: LR={self.config.learning_rate}, Gamma={self.config.gamma}, Clip={self.config.clip_epsilon}, VF={self.config.vf_coef}, Ent={self.config.entropy_coef}"
        )
        self.logger.info(
            f"Neural Critic: {self.config.use_neural_critic}, Path: {self.config.neural_critic_checkpoint_path if self.config.use_neural_critic else 'N/A'}"
        )
        self.logger.info(
            f"Reward Normalization: {self.config.normalize_rewards}, Clip: {self.config.reward_clip_value if self.config.normalize_rewards else 'N/A'}"
        )

    def train(self):
        cfg = self.config
        critic_desc = (
            "[bold green]Neural Approximator[/]"
            if cfg.use_neural_critic and self.neural_critic_model
            else "[bold blue]Symbolic Critic[/]"
        )
        if cfg.use_neural_critic and self.neural_critic_model is None:
            critic_desc += " ([red]Load Failed![/red])"
        self.console.print(
            Panel.fit(
                f"Starting PPO: mode='{cfg.mode}', run='{cfg.run_name}'\nReward Critic: {critic_desc}",
                title="Setup",
                border_style="blue",
            )
        )
        self.console.print(
            f"Device: [cyan]{self.device}[/], Episodes: {cfg.num_episodes}, Steps/Ep: {cfg.episode_length}"
        )
        self.console.print(
            f"Envs: {cfg.num_envs}, Steps/Rollout: [bold yellow]{cfg.n_steps_per_rollout}[/], PPO Epochs: {cfg.num_epochs_per_update}, Minibatch: {cfg.minibatch_size}"
        )
        self.console.print(
            f"Total Transitions/Update: {cfg.batch_size}, Reward Norm: {cfg.normalize_rewards}"
        )
        self.console.print(f"Checkpoints & Logs: [green]{cfg.checkpoint_dir}[/green]")
        self.logger.info("Training setup complete.")

        latest_checkpoint_path = os.path.join(
            cfg.checkpoint_dir, "latest_checkpoint.pth"
        )
        if os.path.exists(latest_checkpoint_path):
            self.logger.info(
                f"Loading PPO agent checkpoint from {latest_checkpoint_path}"
            )
            self.start_episode = self.agent.load_checkpoint(latest_checkpoint_path)
            # Note: reward_rms state should be loaded in load_checkpoint if saved
            self.is_first_episode_for_baseline = self.start_episode == 0
            self.logger.info(f"Resuming training from episode {self.start_episode}")
            if self.start_episode >= cfg.num_episodes:
                msg = f"Checkpoint indicates training already completed ({self.start_episode}/{cfg.num_episodes}). Exiting."
                self.console.print(f"[yellow]{msg}[/yellow]")
                self.logger.warning(msg)
                return
        else:
            self.logger.info("No PPO agent checkpoint found. Starting from scratch.")
            self.start_episode = 0

        episode_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            TextColumn(
                "[bold]Metrics:[/]{task.fields[metrics]}", justify="left", style="white"
            ),
            console=self.console,
            transient=False,
        )
        map_obs, hero_obs = None, None

        with episode_progress:
            episode_task = episode_progress.add_task(
                "[cyan]Training Episodes",
                total=cfg.num_episodes,
                completed=self.start_episode,
                metrics=" Starting...",
            )
            for episode in range(self.start_episode, cfg.num_episodes):
                map_obs, hero_obs, initial_maps_cpu = self.env.reset()
                original_episode_rewards = []  # Store original rewards for logging
                steps_this_episode = 0
                last_update_metrics = {}

                if (
                    episode == self.start_episode
                    or (episode + 1) % cfg.print_maps_freq == 0
                ):
                    num_maps_to_print = min(3, cfg.num_envs)
                    episode_progress.console.print(
                        Panel(
                            f"--- Ep {episode + 1}: Initial Maps (First {num_maps_to_print}) ---",
                            expand=False,
                            border_style="dim",
                        )
                    )
                    for i in range(num_maps_to_print):
                        print_map(
                            episode_progress.console,
                            initial_maps_cpu[i],
                            title=f"Ep {episode + 1} Init (Env {i})",
                        )

                step_task_desc = f"Ep {episode + 1}/{cfg.num_episodes} Steps"
                step_task = episode_progress.add_task(
                    step_task_desc, total=cfg.episode_length, visible=True, metrics=""
                )
                rollout_step = 0

                while steps_this_episode < cfg.episode_length:
                    if rollout_step == 0:
                        self.memory.clear()
                    steps_to_run = min(
                        cfg.n_steps_per_rollout - rollout_step,
                        cfg.episode_length - steps_this_episode,
                    )
                    if steps_to_run <= 0:
                        break

                    for _ in range(steps_to_run):
                        if steps_this_episode >= cfg.episode_length:
                            break
                        action, log_prob, value = self.agent.select_action(
                            map_obs, hero_obs, cfg.temperature
                        )
                        next_obs_tuple, reward, done, info = self.env.step(action)
                        next_map_obs, next_hero_obs = next_obs_tuple

                        original_episode_rewards.append(
                            reward.mean().item()
                        )  # Log original reward
                        reward_to_store = reward  # Default to original reward

                        # --- Reward Normalization ---
                        if cfg.normalize_rewards:
                            reward_np = reward.cpu().numpy()
                            self.reward_rms.update(reward_np)
                            normalized_reward = (
                                reward_np - self.reward_rms.mean
                            ) / np.sqrt(self.reward_rms.var + 1e-8)
                            clipped_reward = np.clip(
                                normalized_reward,
                                -cfg.reward_clip_value,
                                cfg.reward_clip_value,
                            )
                            reward_tensor = torch.tensor(
                                clipped_reward, dtype=torch.float32, device=self.device
                            )
                            # Ensure shape consistency (e.g., if num_envs=1)
                            if reward_tensor.shape != (cfg.num_envs,):
                                if cfg.num_envs == 1 and reward_tensor.ndim == 0:
                                    reward_tensor = reward_tensor.reshape(cfg.num_envs)
                                else:  # Should not happen if RMS handles shapes correctly
                                    self.logger.warning(
                                        f"Unexpected reward shape after normalization: {reward_tensor.shape}"
                                    )
                                    reward_tensor = (
                                        reward  # Fallback to original if shape weird
                                    )
                            reward_to_store = reward_tensor

                        self.memory.add(
                            map_obs,
                            hero_obs,
                            action,
                            log_prob,
                            reward_to_store,
                            done,
                            value,
                        )  # Add possibly normalized reward
                        map_obs, hero_obs = next_map_obs, next_hero_obs
                        self.agent.total_steps_interacted += cfg.num_envs
                        steps_this_episode += 1
                        rollout_step += 1
                        episode_progress.update(
                            step_task,
                            advance=1,
                            description=f"{step_task_desc} ({rollout_step}/{cfg.n_steps_per_rollout})",
                        )

                    if rollout_step == cfg.n_steps_per_rollout:
                        with torch.no_grad():
                            hero_input_critic = (
                                hero_obs
                                if self.agent.critic.use_hero_tensor_in_encoder
                                else None
                            )
                            last_value = self.agent.critic(
                                map_obs, hero_input_critic
                            ).squeeze(-1)
                        self.memory.compute_gae_returns(
                            last_value, cfg.gamma, cfg.gae_lambda
                        )
                        update_metrics = self.agent.update(self.memory)
                        last_update_metrics = update_metrics
                        if update_metrics:
                            p_loss = update_metrics.get("policy_loss", float("nan"))
                            v_loss = update_metrics.get("value_loss", float("nan"))
                            ent = update_metrics.get("entropy", float("nan"))
                            kl = update_metrics.get("approx_kl", float("nan"))
                            clip_frac = update_metrics.get(
                                "clip_fraction", float("nan")
                            )
                            self.logger.info(
                                f"Ep: {episode + 1}, Upd: {self.agent.total_updates_performed}, Step: {self.agent.total_steps_interacted}, P_Loss: {p_loss:.4f}, V_Loss: {v_loss:.4f}, Ent: {ent:.4f}, KL: {kl:.4f}, Clip: {clip_frac:.3f}"
                            )
                        else:
                            self.logger.warning(
                                f"Ep: {episode + 1}, Update: {self.agent.total_updates_performed} no metrics."
                            )
                        rollout_step = 0

                avg_ep_reward = (
                    np.mean(original_episode_rewards)
                    if original_episode_rewards
                    else 0.0
                )
                if self.is_first_episode_for_baseline:
                    self.reward_baseline = avg_ep_reward
                    self.is_first_episode_for_baseline = False
                else:
                    self.reward_baseline = (
                        cfg.reward_baseline_alpha * avg_ep_reward
                        + (1 - cfg.reward_baseline_alpha) * self.reward_baseline
                    )
                self.logger.info(
                    f"Ep: {episode + 1}/{cfg.num_episodes} fin. Steps: {steps_this_episode}. AvgReward(orig): {avg_ep_reward:.4f}, Baseline(EMA): {self.reward_baseline:.4f}, TotalSteps: {self.agent.total_steps_interacted:,}"
                )

                p_loss_str = (
                    f"{last_update_metrics.get('policy_loss', float('nan')):>7.3f}"
                )
                v_loss_str = (
                    f"{last_update_metrics.get('value_loss', float('nan')):>7.3f}"
                )
                ent_str = f"{last_update_metrics.get('entropy', float('nan')):>6.3f}"
                metrics_str = f"AvgRew:[yellow]{avg_ep_reward:>7.3f}[/]| Base:[cyan]{self.reward_baseline:>7.3f}[/]| P:[red]{p_loss_str}[/]| V:[magenta]{v_loss_str}[/]| E:[blue]{ent_str}[/]"
                episode_progress.update(episode_task, advance=1, metrics=metrics_str)
                episode_progress.remove_task(step_task)

                if (
                    episode + 1
                ) % cfg.print_maps_freq == 0 or episode == cfg.num_episodes - 1:
                    num_maps_to_print = min(3, cfg.num_envs)
                    final_maps_cpu = self.env.maps.detach().cpu()
                    episode_progress.console.print(
                        Panel(
                            f"--- Episode {episode + 1}: Final Maps (First {num_maps_to_print}) ---",
                            expand=False,
                            border_style="dim",
                        )
                    )
                    for i in range(num_maps_to_print):
                        print_map(
                            episode_progress.console,
                            final_maps_cpu[i],
                            title=f"Ep {episode + 1} Final (Env {i})",
                        )

                if (
                    episode + 1
                ) % cfg.save_checkpoint_freq == 0 or episode == cfg.num_episodes - 1:
                    chk_path = os.path.join(
                        cfg.checkpoint_dir, f"checkpoint_ep_{episode + 1}.pth"
                    )
                    self.agent.save_checkpoint(chk_path, episode)
                    self.logger.info(
                        f"Periodic checkpoint saved to {chk_path} after ep {episode + 1}"
                    )
                    latest_path = os.path.join(
                        cfg.checkpoint_dir, "latest_checkpoint.pth"
                    )
                    self.agent.save_checkpoint(latest_path, episode)
                    self.logger.info(f"Latest checkpoint updated to {latest_path}")

        msg = f"Training finished: {cfg.num_episodes} eps, {self.agent.total_steps_interacted:,} steps, {self.agent.total_updates_performed:,} updates."
        self.console.print(Panel(msg, title="Complete", border_style="green"))
        self.logger.info(msg)
        final_chk_path = os.path.join(cfg.checkpoint_dir, "final_checkpoint.pth")
        self.agent.save_checkpoint(final_chk_path, cfg.num_episodes - 1)
        self.console.print(f"Final checkpoint saved: [green]{final_chk_path}[/green]")
        self.logger.info(f"Final checkpoint saved to {final_chk_path}")
        latest_path = os.path.join(cfg.checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(final_chk_path):
            try:
                if os.path.exists(latest_path):
                    os.remove(latest_path)
                import shutil

                shutil.copyfile(final_chk_path, latest_path)
                self.logger.info(
                    f"Latest checkpoint updated to final state: {latest_path}"
                )
            except Exception as link_e:
                self.logger.error(
                    f"Failed to update latest checkpoint link/copy: {link_e}"
                )


# --- Main Execution ---
if __name__ == "__main__":
    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__), "ppo_episodic_config.yaml"
    )
    parser = argparse.ArgumentParser(
        description="Train PPO Agent with Optional Neural Critic"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to PPO config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()
    config_path = args.config

    if os.path.exists(config_path):
        print(f"Loading PPO configuration from {config_path}")
        config = configure_from_yaml(config_path)
    else:
        print(
            f"Config file '{config_path}' not found. Using default PPOConfig settings."
        )
        config = PPOConfig()

    symbolic_critic = actual_critic
    trainer = PPOTrainer(config, symbolic_critic)

    try:
        trainer.train()
    except KeyboardInterrupt:
        msg = "\nTraining interrupted. Saving final checkpoint..."
        trainer.console.print(f"[yellow]{msg}[/yellow]")
        trainer.logger.warning(msg)
        latest_episode = -1
        latest_checkpoint_path = os.path.join(
            trainer.config.checkpoint_dir, "latest_checkpoint.pth"
        )
        if os.path.exists(latest_checkpoint_path):
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
                latest_episode = checkpoint.get("episode", -1)
            except Exception as e:
                trainer.logger.error(
                    f"Could not read latest ckpt episode on interrupt: {e}"
                )
        interrupted_chk_path = os.path.join(
            trainer.config.checkpoint_dir, "final_checkpoint_interrupted.pth"
        )
        if hasattr(trainer, "agent"):
            trainer.agent.save_checkpoint(interrupted_chk_path, latest_episode)
            final_msg = f"Interrupted state ckpt saved to {interrupted_chk_path} (ep {latest_episode + 1})"
            print(final_msg)
            trainer.logger.info(final_msg)
        else:
            print("Agent not initialized, cannot save interrupt checkpoint.")
    except Exception as e:
        trainer.console.print("\n[bold red]Critical error during training:[/bold red]")
        trainer.console.print_exception(show_locals=False)
        trainer.logger.error("Critical error during training.", exc_info=True)
        print("[bold red]Attempting to save error state checkpoint...[/bold red]")
        latest_episode = -1
        latest_checkpoint_path = os.path.join(
            trainer.config.checkpoint_dir, "latest_checkpoint.pth"
        )
        if os.path.exists(latest_checkpoint_path):
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
                latest_episode = checkpoint.get("episode", -1)
            except Exception as load_e:
                trainer.logger.error(
                    f"Could not read latest ckpt episode during error handling: {load_e}"
                )
        error_chk_path = os.path.join(
            trainer.config.checkpoint_dir, "final_checkpoint_error.pth"
        )
        if hasattr(trainer, "agent"):
            trainer.agent.save_checkpoint(error_chk_path, latest_episode)
            error_msg = (
                f"Error state ckpt saved to {error_chk_path} (ep {latest_episode + 1})"
            )
            print(error_msg)
            trainer.logger.info(error_msg)
        else:
            print("Agent not initialized, cannot save error checkpoint.")

    # --- Optional Post-Training Test ---
    console = Console()
    print("\n--- Optional: Testing Network Forward Pass ---")
    try:
        if "trainer" in locals() and hasattr(trainer, "agent"):
            test_agent = trainer.agent
            current_config = trainer.config
            batch_size = 2
            test_map_shape = (
                batch_size,
                1,
                current_config.map_size[0],
                current_config.map_size[1],
            )
            test_hero_shape = (batch_size, current_config.hero_tensor_size)
            test_map = torch.randint(0, 7, size=test_map_shape, dtype=torch.float32).to(
                current_config.device
            )
            test_hero = torch.rand(test_hero_shape, dtype=torch.float32).to(
                current_config.device
            )
            test_agent.actor.eval()
            test_agent.critic.eval()
            with torch.no_grad():
                hero_input_actor = (
                    test_hero if test_agent.actor.use_hero_tensor_in_encoder else None
                )
                hero_input_critic = (
                    test_hero if test_agent.critic.use_hero_tensor_in_encoder else None
                )
                action_logits = test_agent.actor(test_map, hero_input_actor)
                action_probs = F.softmax(action_logits, dim=-1)
                state_values = test_agent.critic(test_map, hero_input_critic)
            console.print(
                f"Mode: {current_config.mode}, Action Dim: {current_config.action_dim}, Uses Hero Tensor: {test_agent.actor.use_hero_tensor_in_encoder}"
            )
            console.print(f"Input Map: {test_map.shape}, Input Hero: {test_hero.shape}")
            console.print(
                f"Output Actor Logits: {action_logits.shape}, Probs: {action_probs.shape}"
            )
            console.print(f"Output Critic Values: {state_values.shape}")
            assert action_logits.shape == (batch_size, current_config.action_dim), (
                "Actor shape mismatch!"
            )
            assert state_values.shape == (batch_size, 1), "Critic shape mismatch!"
            console.print("[green]Network forward pass test successful.[/green]")
        else:
            console.print(
                "[yellow]Trainer/agent not initialized, skipping network test.[/yellow]"
            )
    except AttributeError as ae:
        console.print(f"[yellow]Attribute error during test: {ae}. Skipping.[/yellow]")
    except Exception as e:
        console.print(f"[red]Network forward pass test failed: {e}[/red]")
        console.print_exception(show_locals=False)
