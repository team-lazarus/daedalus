# --- START OF FILE ppo_agentsv4_refined.py ---

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
import shutil

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
# Assume running from base_directory, adjust path if necessary
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
    print("Attempting relative import based on assumed structure...")
    try:
        # Adjust path assuming agents/ and critics/ are siblings relative to base_directory
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
    """Enumeration for hero entry points."""

    TOP = 0
    LEFT = 1
    BOTTOM = 2
    RIGHT = 3


# --- Model Definition ---
class PolicyNetworkEncoder(nn.Module):
    """Encodes the map and hero state into a latent representation using an MLP."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (12, 12),
        output_size: int = 1024,
        hero_tensor_size: int = 5,
        hidden_dims: List[int] = [1024, 2048],
    ):
        """
        Initializes the PolicyNetworkEncoder.

        Args:
            input_size: Dimensions of the map (height, width).
            output_size: Dimension of the output latent vector.
            hero_tensor_size: Size of the hero state vector.
            hidden_dims: List of hidden layer dimensions.
        """
        super().__init__()
        self.input_size_x, self.input_size_y = input_size
        map_flat_dim = self.input_size_x * self.input_size_y
        input_dim = map_flat_dim + hero_tensor_size

        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_size))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, hero_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.

        Args:
            x: Map tensor (N, C, H, W) or (N, H, W).
            hero_tensor: Hero state tensor (N, hero_tensor_size).

        Returns:
            Latent representation tensor (N, output_size).
        """
        if x.dim() == 4:  # (N, C, H, W)
             x = x.squeeze(1) # Remove channel dim if present -> (N, H, W)

        x_flat = torch.flatten(x, start_dim=1)
        hero_tensor = hero_tensor.float()
        combined = torch.cat([x_flat, hero_tensor], dim=1)
        latent = self.net(combined)
        return latent


class PolicyNetworkDecoder(nn.Module):
    """Decodes latent representation into action logits or state values."""

    def __init__(
        self,
        input_size: int = 1024,
        hidden_sizes: List[int] = [512, 256],
        output_size: int = 7,
    ):
        """
        Initializes the PolicyNetworkDecoder.

        Args:
            input_size: Dimension of the input latent vector.
            hidden_sizes: List of hidden layer dimensions.
            output_size: Dimension of the output (e.g., action logits or value).
        """
        super().__init__()
        layers = []
        current_dim = input_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder.

        Args:
            x: Latent representation tensor (N, input_size).

        Returns:
            Output tensor (N, output_size).
        """
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
    ):
        """
        Initializes the PPOPolicyNetwork (Actor).

        Args:
            input_size: Map dimensions (height, width).
            encoder_output_size: Output size of the encoder.
            decoder_hidden_sizes: Hidden layer sizes for the actor decoder.
            action_dim: Number of possible actions.
            hero_tensor_size: Size of the hero state vector.
            encoder_hidden_dims: Hidden layer sizes for the encoder.
        """
        super().__init__()
        self.encoder = PolicyNetworkEncoder(
            input_size=input_size,
            output_size=encoder_output_size,
            hero_tensor_size=hero_tensor_size,
            hidden_dims=encoder_hidden_dims,
        )
        self.actor_decoder = PolicyNetworkDecoder(
            input_size=encoder_output_size,
            hidden_sizes=decoder_hidden_sizes,
            output_size=action_dim,
        )

    def forward(
        self, map_tensor: torch.Tensor, hero_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass returning action logits.

        Args:
            map_tensor: Map tensor (N, H, W) or (N, 1, H, W).
            hero_tensor: Hero state tensor (N, hero_tensor_size).

        Returns:
            Action logits tensor (N, action_dim).
        """
        latent = self.encoder(map_tensor, hero_tensor)
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
    ):
        """
        Initializes the ValueNetwork (Critic).

        Args:
            input_size: Map dimensions (height, width).
            encoder_output_size: Output size of the encoder.
            decoder_hidden_sizes: Hidden layer sizes for the value decoder.
            hero_tensor_size: Size of the hero state vector.
            encoder_hidden_dims: Hidden layer sizes for the encoder.
        """
        super().__init__()
        self.encoder = PolicyNetworkEncoder(
            input_size=input_size,
            output_size=encoder_output_size,
            hero_tensor_size=hero_tensor_size,
            hidden_dims=encoder_hidden_dims,
        )
        self.value_decoder = PolicyNetworkDecoder(
            input_size=encoder_output_size,
            hidden_sizes=decoder_hidden_sizes,
            output_size=1,  # Output a single value
        )

    def forward(
        self, map_tensor: torch.Tensor, hero_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass returning state value prediction.

        Args:
            map_tensor: Map tensor (N, H, W) or (N, 1, H, W).
            hero_tensor: Hero state tensor (N, hero_tensor_size).

        Returns:
            State value tensor (N, 1).
        """
        latent = self.encoder(map_tensor, hero_tensor)
        value = self.value_decoder(latent)
        return value


# --- Configuration ---
@dataclass
class PPOConfig:
    """Configuration class for PPO training."""

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
    neural_critic_checkpoint_path: str = "critics/neural_critic_checkpoints/critic_MLP_latest/latest_checkpoint.pth"

    batch_size: int = field(init=False)
    action_dim: int = field(init=False)

    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.batch_size = self.num_envs * self.n_steps_per_rollout
        if self.minibatch_size <= 0:
             print(f"Warning: Invalid minibatch_size ({self.minibatch_size}), setting to batch_size ({self.batch_size}).")
             self.minibatch_size = self.batch_size
        elif self.batch_size % self.minibatch_size != 0:
            print(
                f"Warning: Minibatch size ({self.minibatch_size}) is not a divisor of the "
                f"total batch size ({self.batch_size}). This can lead to incomplete "
                "batches during update."
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
    """
    Loads configuration from a YAML file, filtering unknown keys.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        An instance of PPOConfig.
    """
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
                "true", "1", "yes", "y",
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
    """
    Gets the torch device object based on the provided string.

    Args:
        device_str: "cuda" or "cpu".

    Returns:
        A torch.device object.
    """
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def initialize_map_hero(
    config: PPOConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Initializes a random map, hero tensor, and agent start position.

    Performs a random walk to create initial map structure.

    Args:
        config: The PPOConfig object.

    Returns:
        A tuple containing:
            - map_tensor: Initial map (H, W) as a torch tensor on CPU.
            - hero_tensor: Initial hero state vector as a torch tensor on CPU.
            - start_pos: Tuple (x, y) of the agent's starting position.
    """
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

    # Assumes tile values: 0:Unused, 1:Empty, 6:Door, 2-5:Enemy types
    # Values should align with critic expectations
    for _ in range(config.initial_map_walk_steps):
        is_empty = random.random() < config.initial_map_empty_prob
        if is_empty:
            tile_value = 1 if random.random() < 0.9 else 6 # Small chance of placing a door
        else:
            tile_value = random.randint(2, 5) # Place an enemy type

        map_tensor[current_pos] = tile_value

        # Move to adjacent or diagonal tile
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        dx, dy = random.choice(moves)
        next_x = max(0, min(size_x - 1, current_pos[0] + dx))
        next_y = max(0, min(size_y - 1, current_pos[1] + dy))
        current_pos = (next_x, next_y)

    # Ensure the agent's starting position is walkable (tile value 1)
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
    """
    Simulates multiple environments (map generation processes) in parallel.

    Handles state updates based on agent actions and calculates rewards using
    a provided critic function.
    """

    def __init__(self, config: PPOConfig, critic_function_to_use: Callable):
        """
        Initializes the BatchedEnvSimulator.

        Args:
            config: The PPOConfig object.
            critic_function_to_use: The function to call to get rewards for maps.
                                    Takes (map_tensor, hero_tensor) and returns scores.
        """
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
        # Agent positions are tracked on CPU for simplicity in action logic
        self.agent_positions = [(0, 0)] * self.num_envs

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resets all environments to new initial states.

        Returns:
            A tuple containing:
                - map_obs: Initial map observations for the policy (N, 1, H, W) on device.
                - hero_obs: Initial hero observations for the policy (N, hero_size) on device.
                - initial_maps_cpu_batch: The raw initial maps (N, H, W) on CPU.
        """
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
        """
        Formats the current state into observations suitable for the policy network.

        Returns:
            A tuple containing:
                - map_obs: Formatted map tensor (N, 1, H, W) as float on device.
                - hero_obs: Formatted hero tensor (N, hero_size) as float on device.
        """
        map_obs = self.maps.unsqueeze(1).float()
        hero_obs = self.heroes.float()
        return map_obs, hero_obs

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, List[Dict]
    ]:
        """
        Applies actions to all environments and advances their state.

        Args:
            actions: Tensor of actions selected by the policy (N,).

        Returns:
            A tuple containing:
                - next_obs: Tuple (next_map_obs, next_hero_obs) for the policy.
                - rewards: Tensor of rewards received (N,).
                - dones: Tensor indicating if episodes terminated (N,). Always False here.
                - infos: List of info dictionaries (one per environment). Currently empty.
        """
        actions_device = actions.to(self.device)

        for i in range(self.num_envs):
            action_val = actions_device[i].item()
            pos_x, pos_y = self.agent_positions[i]
            current_map = self.maps[i]
            mode = self.config.mode

            if mode == "narrow":
                # Action 0-6: Place tile type (0-6) at current pos, then move randomly
                if 0 <= action_val <= 6:
                    current_map[pos_x, pos_y] = action_val
                dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                next_x = max(0, min(self.map_size_x - 1, pos_x + dx))
                next_y = max(0, min(self.map_size_y - 1, pos_y + dy))
                self.agent_positions[i] = (next_x, next_y)

            elif mode == "turtle":
                # Action 0-6: Place tile; 7-10: Move turtle (Up, Left, Down, Right)
                if 0 <= action_val <= 6:
                    current_map[pos_x, pos_y] = action_val
                elif action_val == 7: pos_x = max(0, pos_x - 1)
                elif action_val == 8: pos_y = max(0, pos_y - 1)
                elif action_val == 9: pos_x = min(self.map_size_x - 1, pos_x + 1)
                elif action_val == 10: pos_y = min(self.map_size_y - 1, pos_y + 1)
                self.agent_positions[i] = (pos_x, pos_y)

            elif mode == "wide":
                # Action determines tile type and location across the entire map
                num_tile_types = 7
                n_tiles_total = self.map_size_x * self.map_size_y
                tile_type = action_val // n_tiles_total
                flat_index = action_val % n_tiles_total
                target_x = flat_index // self.map_size_y
                target_y = flat_index % self.map_size_y

                if 0 <= tile_type < num_tile_types:
                    if 0 <= target_x < self.map_size_x and 0 <= target_y < self.map_size_y:
                        current_map[target_x, target_y] = tile_type
                # Agent position typically doesn't change in 'wide' mode

            else:
                raise ValueError(f"Unknown mode in step: {mode}")

        # Calculate rewards using the provided critic function
        # Critic function should handle tensor formatting if needed
        rewards = self.critic_func(self.maps, self.heroes)
        rewards = rewards.to(self.device).float().squeeze()
        if rewards.shape != (self.num_envs,):
            raise ValueError(
                f"Critic function returned unexpected rewards shape: {rewards.shape}. Expected ({self.num_envs},)"
            )

        # Episodes have fixed length, termination handled by trainer
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        next_map_obs, next_hero_obs = self._get_observation()
        infos = [{} for _ in range(self.num_envs)]

        return (next_map_obs, next_hero_obs), rewards, dones, infos


# --- PPO Memory ---
class PPOMemory:
    """Stores transitions collected during rollouts for PPO updates."""

    def __init__(
        self,
        num_envs: int,
        n_steps: int,
        map_shape: Tuple[int, int],
        hero_shape: int,
        device: torch.device,
    ):
        """
        Initializes the PPOMemory buffer.

        Args:
            num_envs: Number of parallel environments.
            n_steps: Number of steps per rollout per environment.
            map_shape: Map dimensions (height, width).
            hero_shape: Size of the hero state vector.
            device: The torch device to store tensors on.
        """
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
        self.dones = torch.zeros(
            (n_steps, num_envs), dtype=torch.bool, device=device
        )
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
        """
        Adds a transition (for all environments) to the memory buffer.

        Args:
            map_obs: Map observations (N, 1, H, W).
            hero_obs: Hero observations (N, hero_size).
            action: Actions taken (N,).
            log_prob: Log probabilities of actions (N,).
            reward: Rewards received (N,).
            done: Done flags (N,).
            value: Value estimates (N,).
        """
        if self.ptr >= self.n_steps:
            raise IndexError(
                "Memory buffer overflow. Ensure clear() is called before starting new rollout."
            )

        self.maps[self.ptr] = map_obs.to(self.device)
        self.heroes[self.ptr] = hero_obs.to(self.device)
        self.actions[self.ptr] = action.to(self.device)
        self.log_probs[self.ptr] = log_prob.to(self.device)
        self.rewards[self.ptr] = reward.to(self.device)
        self.dones[self.ptr] = done.to(self.device)
        self.values[self.ptr] = value.to(self.device)
        self.ptr += 1

    def compute_gae_returns(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ):
        """
        Computes Generalized Advantage Estimation (GAE) and returns.

        Args:
            last_value: Value estimate of the state reached after the last step (N,).
            gamma: Discount factor.
            gae_lambda: GAE lambda factor.
        """
        if self.ptr != self.n_steps:
            print(
                f"Warning: Computing GAE on incomplete buffer (ptr={self.ptr}, n_steps={self.n_steps})."
            )
            if self.ptr == 0:
                self.advantages = torch.zeros(0, self.num_envs, device=self.device)
                self.returns = torch.zeros(0, self.num_envs, device=self.device)
                return

        last_value = last_value.to(self.device).squeeze()
        if last_value.shape != (self.num_envs,):
            raise ValueError(
                f"Expected last_value shape ({self.num_envs},), got {last_value.shape}"
            )

        num_steps_filled = self.ptr
        self.advantages = torch.zeros(
            (num_steps_filled, self.num_envs), dtype=torch.float32, device=self.device
        )
        last_gae_lam = 0

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
        """
        Generates minibatches of transitions from the collected rollout data.

        Requires compute_gae_returns() to have been called first.

        Args:
            batch_size: The total number of transitions to sample from (usually n_steps * num_envs).
            minibatch_size: The size of each minibatch to yield.

        Yields:
            Dictionaries containing minibatch tensors for:
            'maps', 'heroes', 'actions', 'old_log_probs', 'advantages', 'returns', 'old_values'.
        """
        if self.advantages is None or self.returns is None:
            raise ValueError("Advantages/returns must be computed before get_minibatches.")

        num_steps_filled = self.advantages.shape[0]
        num_transitions = num_steps_filled * self.num_envs
        if num_transitions == 0:
            return

        if num_transitions < batch_size:
            print(
                f"Warning: Requested batch_size {batch_size} > available transitions "
                f"{num_transitions}. Using available transitions."
            )
            batch_size = num_transitions

        if minibatch_size <= 0 or minibatch_size > batch_size:
             print(f"Warning: Invalid minibatch_size ({minibatch_size}), adjusting to batch_size ({batch_size}).")
             minibatch_size = batch_size

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

        for start_idx in range(0, batch_size, minibatch_size):
            end_idx = min(start_idx + minibatch_size, num_transitions)
            if start_idx >= end_idx: continue
            mb_indices = indices[start_idx:end_idx]
            if len(mb_indices) == 0: continue

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
        """Resets the buffer pointer and clears computed advantages/returns."""
        self.ptr = 0
        self.advantages = None
        self.returns = None


# --- PPO Agent ---
class PPOAgent:
    """The PPO Agent containing policy/value networks and update logic."""

    def __init__(self, config: PPOConfig):
        """
        Initializes the PPOAgent.

        Args:
            config: The PPOConfig object.
        """
        self.config = config
        self.device = get_device(config.device)

        self.actor = PPOPolicyNetwork(
            input_size=config.map_size,
            encoder_output_size=config.encoder_output_size,
            decoder_hidden_sizes=config.decoder_hidden_sizes,
            action_dim=config.action_dim,
            hero_tensor_size=config.hero_tensor_size,
            encoder_hidden_dims=config.encoder_hidden_dims,
        ).to(self.device)
        self.critic = ValueNetwork(
            input_size=config.map_size,
            encoder_output_size=config.encoder_output_size,
            decoder_hidden_sizes=config.decoder_hidden_sizes,
            hero_tensor_size=config.hero_tensor_size,
            encoder_hidden_dims=config.encoder_hidden_dims,
        ).to(self.device)

        all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(all_params, lr=config.learning_rate, eps=1e-5)

        self.total_steps_interacted = 0
        self.total_updates_performed = 0

    def select_action(
        self, map_obs: torch.Tensor, hero_obs: torch.Tensor, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects actions based on policy for the given observations.

        Args:
            map_obs: Map observations (N, 1, H, W).
            hero_obs: Hero observations (N, hero_size).
            temperature: Temperature for sampling actions (higher = more random).

        Returns:
            A tuple containing:
                - action: Selected actions (N,).
                - log_prob: Log probabilities of the selected actions (N,).
                - value: Value estimates for the current states (N,).
        """
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            action_logits = self.actor(map_obs, hero_obs)
            value = self.critic(map_obs, hero_obs).squeeze(-1) # Shape (N,)

            if temperature > 1e-8: # Avoid division by zero or near-zero
                scaled_logits = action_logits / temperature
            else:
                scaled_logits = action_logits

            probs = F.softmax(scaled_logits, dim=-1)
            dist = Categorical(probs=probs)

            if temperature > 1e-8:
                action = dist.sample()
            else:
                action = torch.argmax(probs, dim=-1) # Deterministic

            log_prob = dist.log_prob(action)

        self.actor.train()
        self.critic.train()
        return action, log_prob, value

    def evaluate_actions(
        self, map_obs: torch.Tensor, hero_obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates given actions against the current policy for PPO updates.

        Args:
            map_obs: Map observations from the minibatch (MB, 1, H, W).
            hero_obs: Hero observations from the minibatch (MB, hero_size).
            actions: Actions taken in the minibatch (MB,).

        Returns:
            A tuple containing:
                - log_prob: Log probabilities of the provided actions (MB,).
                - value: Value estimates for the provided states (MB,).
                - entropy: Entropy of the action distribution for the states (MB,).
        """
        action_logits = self.actor(map_obs, hero_obs)
        value = self.critic(map_obs, hero_obs).squeeze(-1) # Shape (MB,)

        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs=probs)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value, entropy

    def update(self, memory: PPOMemory) -> Dict[str, float]:
        """
        Performs PPO update using data collected in the memory buffer.

        Args:
            memory: The PPOMemory buffer containing collected transitions.

        Returns:
            A dictionary containing average metrics from the update process.
        """
        if memory.advantages is None or memory.returns is None:
            raise RuntimeError("memory.compute_gae_returns() must be called before update()")

        all_metrics = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "approx_kl": [], "clip_fraction": [],
        }

        for _ in range(self.config.num_epochs_per_update):
            minibatch_generator = memory.get_minibatches(
                self.config.batch_size, self.config.minibatch_size
            )

            for batch in minibatch_generator:
                mb_maps, mb_heroes = batch["maps"], batch["heroes"]
                mb_actions, mb_old_log_probs = batch["actions"], batch["old_log_probs"]
                mb_advantages, mb_returns = batch["advantages"], batch["returns"]

                # Normalize advantages per minibatch
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                new_log_probs, new_values, entropy = self.evaluate_actions(
                    mb_maps, mb_heroes, mb_actions
                )

                # Policy Loss (Clipped Surrogate Objective)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss (MSE)
                value_loss = 0.5 * F.mse_loss(new_values, mb_returns)

                # Entropy Loss
                entropy_loss = entropy.mean()

                # Total Loss
                loss = (
                    policy_loss
                    - self.config.entropy_coef * entropy_loss
                    + self.config.vf_coef * value_loss
                )

                # Optimization Step
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                # Log Minibatch Metrics
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
        """
        Saves model, optimizer state, and training progress to a file.

        Args:
            path: The file path to save the checkpoint.
            episode: The last completed episode number.
        """
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
                "config": self.config, # Save config for reference
            }
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, path) # Atomic replace
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint to {path}: {e}")
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except OSError: pass

    def load_checkpoint(self, path: str) -> int:
        """
        Loads model and optimizer state from a checkpoint file.

        Args:
            path: The file path to load the checkpoint from.

        Returns:
            The episode number to start training from (last completed episode + 1).
        """
        if not os.path.exists(path):
            print(f"[Warning] Checkpoint file not found at {path}. Starting from episode 0.")
            return 0
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Basic Compatibility Check
            chk_config = checkpoint.get("config")
            if isinstance(chk_config, PPOConfig):
                mismatched = []
                if chk_config.action_dim != self.config.action_dim: mismatched.append("ActionDim")
                if chk_config.mode != self.config.mode: mismatched.append("Mode")
                if chk_config.map_size != self.config.map_size: mismatched.append("MapSize")
                # Add more checks for network architecture parameters if needed
                if mismatched:
                    print(f"[Warning] Config mismatch! Checkpoint vs Current: {', '.join(mismatched)}")
                    print("Loading weights anyway, but behavior may be unpredictable.")
            else:
                print("[Warning] Checkpoint PPOConfig missing or invalid. Cannot check compatibility.")

            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self.total_steps_interacted = checkpoint.get("total_steps_interacted", 0)
            self.total_updates_performed = checkpoint.get("total_updates_performed", 0)
            start_episode = checkpoint.get("episode", -1) + 1

            print(f"Checkpoint loaded from {path}. Resuming from episode {start_episode}.")
            print(f"  -> Steps Interacted: {self.total_steps_interacted:,}, Updates Performed: {self.total_updates_performed:,}")
            return start_episode
        except KeyError as e:
            print(f"[ERROR] Missing key in checkpoint {path}: {e}. Starting from scratch.")
            return 0
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint from {path}: {e}. Starting from scratch.")
            return 0


# --- Visualization ---
def print_map(console: Console, map_tensor: torch.Tensor, title: str = "Generated Map"):
    """
    Prints a map tensor to the console using Rich table formatting.

    Args:
        console: The Rich Console object.
        map_tensor: The map tensor (H, W), (1, H, W), (N, 1, H, W), or (1, 1, H, W).
        title: The title for the map display.
    """
    original_shape = map_tensor.shape
    if map_tensor.ndim == 4: map_tensor = map_tensor.squeeze(0).squeeze(0) # (1, 1, H, W) -> (H, W)
    elif map_tensor.ndim == 3 and map_tensor.shape[0] == 1: map_tensor = map_tensor.squeeze(0) # (1, H, W) -> (H, W)
    elif map_tensor.ndim == 3 and map_tensor.shape[1] == 1: map_tensor = map_tensor[0].squeeze(0) # (N, 1, H, W) -> First map (H, W)
    elif map_tensor.ndim == 2: pass # Already (H, W)
    else:
        console.print(f"[red]Error: Cannot print map with shape {original_shape}.[/red]")
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
        0: "dim grey50", 1: "white", 6: "bright_green",
        2: "bright_red", 3: "red", 4: "dark_red", 5: "red3",
    }
    default_color = "magenta"
    char_width = 2

    table = Table(title=title, show_header=False, show_edge=False, box=None, padding=(0, 0), expand=False)
    for _ in range(map_size_y):
        table.add_column(justify="center", width=char_width, style="dim")

    for r in range(map_size_x):
        row_cells = [f"[{colors.get(tile, default_color)}]{tile:>{char_width-1}} [/]" for tile in map_np[r]]
        table.add_row(*row_cells)

    console.print(table)


# --- Training Orchestrator ---
class PPOTrainer:
    """Orchestrates the PPO training process with episodic resets and rollouts."""

    def __init__(self, config: PPOConfig, base_critic_func: Callable):
        """
        Initializes the PPOTrainer.

        Args:
            config: The PPOConfig object.
            base_critic_func: The symbolic critic function (used if neural critic fails or is disabled).
        """
        self.config = config
        self.base_critic_func = base_critic_func
        self.device = get_device(config.device)
        self.console = Console()
        self.logger = logging.getLogger(__name__)

        self._seed_everything(config.seed)
        self._setup_logging()
        self.critic_func_to_use = self._initialize_critic()

        self.env = BatchedEnvSimulator(config, self.critic_func_to_use)
        self.agent = PPOAgent(config)
        self.memory = PPOMemory(
            config.num_envs, config.n_steps_per_rollout, config.map_size,
            config.hero_tensor_size, self.device
        )

        self.start_episode = 0
        self.reward_baseline = 0.0
        self.is_first_episode_for_baseline = True

    def _seed_everything(self, seed: int):
        """Sets random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Consider setting these for full determinism, but may impact performance
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        self.logger.info(f"Seeding complete with seed: {seed}")

    def _setup_logging(self):
        """Configures logging to file within the run's checkpoint directory."""
        log_dir = self.config.checkpoint_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, self.config.log_file_name)

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"--- Logging started for run: {self.config.run_name} ---")
        self.logger.info(f"Config: {self.config}")


    def _initialize_critic(self) -> Callable:
        """Loads the neural critic if configured, otherwise returns the base critic."""
        if not self.config.use_neural_critic:
            self.logger.info("Using the original symbolic critic function.")
            self.console.print("Using original symbolic critic ([dim]neural critic disabled[/dim]).")
            return self.base_critic_func

        ckpt_path = self.config.neural_critic_checkpoint_path
        self.logger.info(f"Attempting to load Neural Critic from: {ckpt_path}")
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(ckpt_path)
            self.logger.info(f"Resolved relative path to: {ckpt_path}")

        try:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Neural critic checkpoint not found at: {ckpt_path}")

            critic_checkpoint = torch.load(ckpt_path, map_location=self.device)

            if not isinstance(critic_checkpoint, dict) or "config" not in critic_checkpoint or "model_state_dict" not in critic_checkpoint:
                 raise TypeError("Invalid critic checkpoint format. Expected dict with 'config' and 'model_state_dict'.")

            critic_config_data = critic_checkpoint["config"]
            if isinstance(critic_config_data, dict):
                self.logger.warning("Critic config loaded as dict, reconstructing CriticConfig.")
                try:
                    critic_config = CriticConfig(**critic_config_data)
                except Exception as config_e:
                    raise TypeError(f"Failed to reconstruct CriticConfig: {config_e}")
            elif isinstance(critic_config_data, CriticConfig):
                 critic_config = critic_config_data
            else:
                raise TypeError(f"Unexpected type for critic config in checkpoint: {type(critic_config_data)}")

            self.logger.info(f"Loaded critic config: MapSize={critic_config.map_size}, Hidden={critic_config.mlp_hidden_sizes}")

            # Critical Map Size Check
            critic_config.map_size = tuple(critic_config.map_size)
            if critic_config.map_size != self.config.map_size:
                msg = (f"CRITICAL MAP SIZE MISMATCH! PPO Config: {self.config.map_size}, "
                       f"Loaded Critic Config: {critic_config.map_size}. Cannot proceed.")
                self.logger.error(msg)
                self.console.print(f"[bold red]{msg}[/bold red]")
                raise ValueError(msg)

            self.neural_critic_model = CriticApproximatorMLP(
                input_size=critic_config.map_size[0] * critic_config.map_size[1],
                hidden_sizes=critic_config.mlp_hidden_sizes,
                output_size=1,
                dropout_prob=critic_config.mlp_dropout_prob,
            ).to(self.device)

            self.neural_critic_model.load_state_dict(critic_checkpoint["model_state_dict"])
            self.neural_critic_model.eval()

            self.logger.info("Successfully loaded and initialized Neural Critic model.")

            def neural_critic_wrapper(map_tensor: torch.Tensor, hero_tensor: torch.Tensor) -> torch.Tensor:
                """Wrapper function for the loaded neural critic MLP."""
                map_tensor = map_tensor.to(self.device)
                if map_tensor.dim() == 3: map_tensor = map_tensor.unsqueeze(1) # (N, H, W) -> (N, 1, H, W)
                map_tensor = map_tensor.float()

                with torch.no_grad():
                    # The MLP critic likely only uses the map
                    scores = self.neural_critic_model(map_tensor) # Shape (N, 1)
                return scores.squeeze(-1) # Shape (N,)

            self.console.print("Neural Critic [bold green]loaded and active[/bold green].")
            return neural_critic_wrapper

        except Exception as e:
            self.logger.error(f"Failed to load Neural Critic: {e}", exc_info=True)
            self.console.print(f"[bold red]Error loading Neural Critic:[/bold red] {e}")
            self.console.print("Falling back to the original symbolic critic function.")
            self.neural_critic_model = None # Ensure it's None
            return self.base_critic_func


    def train(self):
        """Runs the main PPO training loop."""
        cfg = self.config

        critic_desc = ("[bold green]Neural Approximator[/]" if cfg.use_neural_critic and hasattr(self, 'neural_critic_model') and self.neural_critic_model
                       else "[bold blue]Symbolic Critic[/]")
        if cfg.use_neural_critic and (not hasattr(self, 'neural_critic_model') or not self.neural_critic_model):
             critic_desc += " ([red]Load Failed![/red])"

        self.console.print(Panel.fit(f"Starting PPO Training: mode='{cfg.mode}', run='{cfg.run_name}'\nReward Critic: {critic_desc}", title="Setup", border_style="blue"))
        self.console.print(f"Device: [cyan]{self.device}[/], Episodes: {cfg.num_episodes:,}, Steps/Episode: {cfg.episode_length}")
        self.console.print(f"Envs: {cfg.num_envs}, Update Freq (Rollout Steps): [bold yellow]{cfg.n_steps_per_rollout}[/], PPO Epochs: {cfg.num_epochs_per_update}, Minibatch: {cfg.minibatch_size}")
        self.console.print(f"Total Transitions per Update: {cfg.batch_size:,}")
        self.console.print(f"Checkpoints & Logs Dir: [green]{cfg.checkpoint_dir}[/green]")
        self.logger.info("Training setup complete.")

        # Load Agent Checkpoint if exists
        latest_checkpoint_path = os.path.join(cfg.checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(latest_checkpoint_path):
            self.logger.info(f"Loading PPO agent checkpoint from {latest_checkpoint_path}")
            self.start_episode = self.agent.load_checkpoint(latest_checkpoint_path)
            self.is_first_episode_for_baseline = self.start_episode == 0
            self.logger.info(f"Resuming training from episode {self.start_episode}")
            if self.start_episode >= cfg.num_episodes:
                msg = f"Checkpoint indicates training already completed ({self.start_episode}/{cfg.num_episodes}). Exiting."
                self.console.print(f"[yellow]{msg}[/yellow]")
                self.logger.warning(msg)
                return
        else:
            self.logger.info("No PPO agent checkpoint found. Starting training from scratch.")
            self.start_episode = 0


        episode_progress = Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(), TextColumn("ETA:"), TimeRemainingColumn(),
            TextColumn("[bold]Metrics:[/]{task.fields[metrics]}", justify="left", style="white"),
            console=self.console, transient=False,
        )

        map_obs, hero_obs = None, None # Initialized in first env.reset()

        with episode_progress:
            episode_task = episode_progress.add_task(
                "[cyan]Training Episodes", total=cfg.num_episodes,
                completed=self.start_episode, metrics=" Starting..."
            )

            for episode in range(self.start_episode, cfg.num_episodes):
                if map_obs is None: # First episode or after resuming
                     map_obs, hero_obs, initial_maps_cpu = self.env.reset()
                else: # Subsequent episodes, state carries over if envs aren't reset per ep
                    # Standard PPO with fixed rollouts doesn't reset envs every episode,
                    # but here the outer loop *is* episodes. We reset each episode.
                    map_obs, hero_obs, initial_maps_cpu = self.env.reset()

                episode_rewards = []
                steps_this_episode = 0
                last_update_metrics = {}
                rollout_step_count = 0 # Steps collected in the current rollout buffer

                if episode == self.start_episode or (episode + 1) % cfg.print_maps_freq == 0:
                    num_maps_to_print = min(3, cfg.num_envs)
                    episode_progress.console.print(Panel(f"--- Episode {episode + 1}: Initial Maps (First {num_maps_to_print}) ---", expand=False, border_style="dim"))
                    for i in range(num_maps_to_print):
                        print_map(episode_progress.console, initial_maps_cpu[i], title=f"Ep {episode + 1} Initial (Env {i})")

                step_task_desc = f"Ep {episode + 1}/{cfg.num_episodes} Steps"
                step_task = episode_progress.add_task(step_task_desc, total=cfg.episode_length, visible=True, metrics="") # Initialize metrics

                while steps_this_episode < cfg.episode_length:
                    # Select action
                    action, log_prob, value = self.agent.select_action(map_obs, hero_obs, cfg.temperature)

                    # Step environment
                    next_obs_tuple, reward, done, _ = self.env.step(action) # Ignore info
                    next_map_obs, next_hero_obs = next_obs_tuple

                    # Store transition
                    # Need map_obs before unsqueeze for memory if network adds channel dim
                    map_obs_for_memory = map_obs # Should already be (N, 1, H, W) from _get_observation
                    self.memory.add(map_obs_for_memory, hero_obs, action, log_prob, reward, done, value)

                    # Update state
                    map_obs, hero_obs = next_map_obs, next_hero_obs
                    episode_rewards.append(reward.mean().item()) # Track average reward across envs

                    # Increment counters
                    self.agent.total_steps_interacted += cfg.num_envs
                    steps_this_episode += 1
                    rollout_step_count += 1

                    # Update step progress bar
                    episode_progress.update(step_task, advance=1, description=f"{step_task_desc} ({rollout_step_count}/{cfg.n_steps_per_rollout} rollout)")

                    # Check if rollout buffer is full
                    if rollout_step_count == cfg.n_steps_per_rollout:
                        # Compute GAE and returns for the completed rollout
                        with torch.no_grad():
                            last_value = self.agent.critic(map_obs, hero_obs).squeeze(-1)
                        self.memory.compute_gae_returns(last_value, cfg.gamma, cfg.gae_lambda)

                        # Perform PPO update
                        update_metrics = self.agent.update(self.memory)
                        last_update_metrics = update_metrics

                        # Log update metrics
                        if update_metrics:
                             p_loss = update_metrics.get('policy_loss', float('nan'))
                             v_loss = update_metrics.get('value_loss', float('nan'))
                             ent = update_metrics.get('entropy', float('nan'))
                             kl = update_metrics.get('approx_kl', float('nan'))
                             clip_frac = update_metrics.get('clip_fraction', float('nan'))
                             self.logger.info(f"Ep: {episode+1}, Update: {self.agent.total_updates_performed}, Step: {self.agent.total_steps_interacted:,}, "
                                              f"P_Loss: {p_loss:.4f}, V_Loss: {v_loss:.4f}, Entropy: {ent:.4f}, KL: {kl:.4f}, ClipFrac: {clip_frac:.3f}")
                        else:
                             self.logger.warning(f"Ep: {episode+1}, Update: {self.agent.total_updates_performed} returned no metrics.")

                        # Clear memory and reset rollout counter
                        self.memory.clear()
                        rollout_step_count = 0


                # --- End of Episode ---
                avg_ep_reward = np.mean(episode_rewards) if episode_rewards else 0.0
                if self.is_first_episode_for_baseline:
                    self.reward_baseline = avg_ep_reward
                    self.is_first_episode_for_baseline = False
                else:
                    alpha = cfg.reward_baseline_alpha
                    self.reward_baseline = alpha * avg_ep_reward + (1 - alpha) * self.reward_baseline

                self.logger.info(f"Ep: {episode + 1}/{cfg.num_episodes} finished. Steps: {steps_this_episode}. AvgReward: {avg_ep_reward:.4f}, "
                                 f"RewardBaseline(EMA): {self.reward_baseline:.4f}, TotalEnvSteps: {self.agent.total_steps_interacted:,}")

                p_loss_str = f"{last_update_metrics.get('policy_loss', float('nan')):>7.3f}"
                v_loss_str = f"{last_update_metrics.get('value_loss', float('nan')):>7.3f}"
                ent_str = f"{last_update_metrics.get('entropy', float('nan')):>6.3f}"
                metrics_str = (f"AvgRew:[yellow]{avg_ep_reward:>7.3f}[/]| Baseline:[cyan]{self.reward_baseline:>7.3f}[/]| "
                               f"P:[red]{p_loss_str}[/]| V:[magenta]{v_loss_str}[/]| E:[blue]{ent_str}[/]")
                episode_progress.update(episode_task, advance=1, metrics=metrics_str)
                episode_progress.remove_task(step_task)

                if (episode + 1) % cfg.print_maps_freq == 0 or episode == cfg.num_episodes - 1:
                    num_maps_to_print = min(3, cfg.num_envs)
                    final_maps_cpu = self.env.maps.detach().cpu()
                    episode_progress.console.print(Panel(f"--- Episode {episode + 1}: Final Maps (First {num_maps_to_print}) ---", expand=False, border_style="dim"))
                    for i in range(num_maps_to_print):
                         print_map(episode_progress.console, final_maps_cpu[i], title=f"Ep {episode + 1} Final (Env {i})")


                # Save Checkpoint Periodically and Latest
                save_now = (episode + 1) % cfg.save_checkpoint_freq == 0
                if save_now or episode == cfg.num_episodes - 1: # Save on last episode too
                    latest_path = os.path.join(cfg.checkpoint_dir, "latest_checkpoint.pth")
                    self.agent.save_checkpoint(latest_path, episode)
                    self.logger.info(f"Latest checkpoint updated to {latest_path} after episode {episode + 1}")

                    if save_now and episode < cfg.num_episodes - 1 : # Avoid duplicate save on last episode if freq aligns
                        chk_path = os.path.join(cfg.checkpoint_dir, f"checkpoint_ep_{episode + 1}.pth")
                        self.agent.save_checkpoint(chk_path, episode)
                        self.logger.info(f"Periodic checkpoint saved to {chk_path}")

        # --- End of Training ---
        msg = (f"Training finished after {cfg.num_episodes} episodes ({self.agent.total_steps_interacted:,} total env steps, "
               f"{self.agent.total_updates_performed:,} PPO updates).")
        self.console.print(Panel(msg, title="Complete", border_style="green"))
        self.logger.info(msg)

        # Save final state labeled explicitly
        final_chk_path = os.path.join(cfg.checkpoint_dir, "final_checkpoint.pth")
        self.agent.save_checkpoint(final_chk_path, cfg.num_episodes - 1)
        self.console.print(f"Final checkpoint saved to: [green]{final_chk_path}[/green]")
        self.logger.info(f"Final checkpoint saved to {final_chk_path}")

        # Ensure latest points to the final one
        latest_path = os.path.join(cfg.checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(final_chk_path):
            try:
                if os.path.exists(latest_path): os.remove(latest_path)
                shutil.copyfile(final_chk_path, latest_path)
                self.logger.info(f"Latest checkpoint ensured to point to final state: {latest_path}")
            except Exception as link_e:
                self.logger.error(f"Failed to update latest checkpoint link/copy: {link_e}")


# --- Main Execution ---
if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    default_config_path = os.path.join(script_dir, "ppo_episodic_config.yaml") # Assumes config is sibling to script

    parser = argparse.ArgumentParser(description="Train PPO Agent with Optional Neural Critic")
    parser.add_argument("--config", type=str, default=default_config_path,
                        help=f"Path to PPO config YAML file (default: {default_config_path})")
    args = parser.parse_args()
    config_path = args.config

    if os.path.exists(config_path):
        print(f"Loading PPO configuration from {config_path}")
        ppo_config = configure_from_yaml(config_path)
    else:
        print(f"Config file '{config_path}' not found. Using default PPOConfig settings.")
        ppo_config = PPOConfig()

    # Use the imported symbolic critic as the base/fallback
    symbolic_critic_func = actual_critic

    # Create and run the trainer
    trainer = PPOTrainer(ppo_config, symbolic_critic_func)
    try:
        trainer.train()
    except KeyboardInterrupt:
        msg = "\nTraining interrupted by user. Saving final checkpoint..."
        trainer.console.print(f"[yellow]{msg}[/yellow]")
        trainer.logger.warning(msg)
        last_completed_episode = -1
        # Try to get last saved episode from latest checkpoint
        latest_checkpoint_path = os.path.join(trainer.config.checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(latest_checkpoint_path):
             try:
                 checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
                 last_completed_episode = checkpoint.get("episode", -1)
             except Exception as e:
                 trainer.logger.error(f"Could not read episode from latest checkpoint on interrupt: {e}")

        # Save current state as interrupted
        interrupted_chk_path = os.path.join(trainer.config.checkpoint_dir, "final_checkpoint_interrupted.pth")
        if hasattr(trainer, 'agent'):
            trainer.agent.save_checkpoint(interrupted_chk_path, last_completed_episode)
            final_msg = f"Interrupted state checkpoint saved to {interrupted_chk_path} (based on last completed ep {last_completed_episode})"
            print(final_msg)
            trainer.logger.info(final_msg)
        else:
             print("Agent not fully initialized, cannot save interrupt checkpoint.")

    except Exception as e:
        trainer.console.print("\n[bold red]An critical error occurred during training:[/bold red]")
        trainer.console.print_exception(show_locals=False)
        trainer.logger.error("A critical error occurred during training.", exc_info=True)

        print("[bold red]Attempting to save error state checkpoint...[/bold red]")
        last_completed_episode = -1
        latest_checkpoint_path = os.path.join(trainer.config.checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(latest_checkpoint_path):
             try:
                 checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
                 last_completed_episode = checkpoint.get("episode", -1)
             except Exception as load_e:
                 trainer.logger.error(f"Could not read episode from latest checkpoint during error handling: {load_e}")

        error_chk_path = os.path.join(trainer.config.checkpoint_dir, "final_checkpoint_error.pth")
        if hasattr(trainer, 'agent'):
            trainer.agent.save_checkpoint(error_chk_path, last_completed_episode)
            error_msg = f"Error state checkpoint saved to {error_chk_path} (based on last completed ep {last_completed_episode})"
            print(error_msg)
            trainer.logger.info(error_msg)
        else:
            print("Agent not fully initialized, cannot save error checkpoint.")