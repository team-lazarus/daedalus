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
import sys  # <<< Added for path modification
import time
import logging
import datetime
from typing import List, Tuple, Dict, Any, Callable, Optional, Union, Generator
from dataclasses import dataclass, field, fields

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
)
from tqdm import tqdm

from enum import Enum

# --- Path Modification to import from parent directory ---
# Add the base directory (parent of 'agents') to the Python path
# This allows importing from 'critics' and 'daedalus'
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, ".."))
if base_dir not in sys.path:
    sys.path.append(base_dir)
# --------------------------------------------------------

# --- Import Critic Related Components ---
try:
    # Import the original critic function
    from daedalus.critics import level_critic as original_level_critic

    # Import the neural critic model classes and config
    from critics.critic_approximator import (
        CriticApproximatorCNN,
        CriticApproximatorMLP,
        CriticConfig as NeuralCriticConfig,  # Use alias to avoid name conflict
        configure_critic_from_yaml as configure_neural_critic_from_yaml,  # If needed
    )

    CRITIC_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Error importing critic components: {e}")
    print(
        "Ensure 'critics' and 'daedalus' directories are adjacent to 'agents' and contain the necessary files."
    )
    CRITIC_IMPORTS_SUCCESSFUL = False

    # Define dummy classes/functions if import fails, so the rest of the script might load
    class CriticApproximatorCNN(nn.Module):
        pass

    class CriticApproximatorMLP(nn.Module):
        pass

    class NeuralCriticConfig:
        pass

    def original_level_critic(*args, **kwargs):
        raise NotImplementedError("Original critic import failed")
# ----------------------------------------


# --- Mock Entry Enum (Replace with actual import if available) ---
class Entry(Enum):
    TOP = 0
    LEFT = 1
    BOTTOM = 2
    RIGHT = 3


# --- Model Definition (Encoder Changed) ---
# PolicyNetworkEncoder, PolicyNetworkDecoder, PPOPolicyNetwork, ValueNetwork remain unchanged
class PolicyNetworkEncoder(nn.Module):
    """Encodes the map and hero state into a latent representation using a NN (MLP)."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (12, 12),
        output_size: int = 1024,
        hero_tensor_size: int = 5,
        hidden_dims: List[int] = [512, 512],
    ):
        super().__init__()
        self.input_size = self.input_size_x, self.input_size_y = input_size
        self.map_flat_dim = self.input_size_x * self.input_size_y
        self.hero_tensor_size = hero_tensor_size
        self.input_dim = self.map_flat_dim + self.hero_tensor_size
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

    def forward(self, x: torch.Tensor, hero_tensor: torch.Tensor) -> torch.Tensor:
        x_flat = torch.flatten(x, start_dim=1)
        hero_tensor = hero_tensor.float()
        combined = torch.cat([x_flat, hero_tensor], dim=1)
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
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_out = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


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
    # <<< Force CUDA >>>
    device: str = "cuda"  # Changed default to cuda

    # Training Loop Control
    num_episodes: int = 1000
    episode_length: int = 256

    # PPO Core Parameters
    num_envs: int = 128
    n_steps_per_rollout: int = 64
    num_epochs_per_update: int = 4
    minibatch_size: int = 64

    # PPO Algorithm Hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Action Space & Exploration
    temperature: float = 0.5

    # Logging & Saving
    save_checkpoint_freq: int = 100
    print_maps_freq: int = 50
    checkpoint_dir: str = "ppo_checkpoints_episodic_metrics"
    run_name: Optional[str] = None
    log_file_name: str = "ppo_training_log.log"

    # Environment Initialization
    initial_map_walk_steps: int = 36
    initial_map_empty_prob: float = 0.75

    # Baseline parameters
    reward_baseline_alpha: float = 0.05

    # Model architecture
    encoder_output_size: int = 1024
    decoder_hidden_sizes: List[int] = field(default_factory=lambda: [512, 256])
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 512])

    # Critic Configuration
    use_neural_critic: bool = True
    neural_critic_base_dir: str = "critics/neural_critic_checkpoints"
    neural_critic_run_name: str = "critic_MLP_latest"  # <<< Placeholder - CHANGE THIS
    fallback_to_original_critic_on_fail: bool = False

    # Calculated properties
    batch_size: int = field(init=False)
    action_dim: int = field(init=False)
    neural_critic_checkpoint_path: str = field(init=False)

    def __post_init__(self):
        """Calculate derived properties and validate CUDA after initialization."""
        # <<< Validate CUDA if requested >>>
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Configuration specifies CUDA device, but CUDA is not available."
            )
        elif self.device != "cuda":
            print(f"Warning: Device is set to '{self.device}', not 'cuda'.")
            # Optionally, fall back to CPU if needed, but current request is to force CUDA
            # self.device = "cpu" # Example fallback

        # --- Remaining calculations ---
        self.batch_size = self.num_envs * self.n_steps_per_rollout
        if self.minibatch_size > self.batch_size:
            print(
                f"Warning: Minibatch size ({self.minibatch_size}) > Rollout batch size ({self.batch_size}). Adjusting minibatch size."
            )
            self.minibatch_size = self.batch_size

        n_tiles = self.map_size[0] * self.map_size[1]
        mode_actions = {"narrow": 7, "turtle": 10, "wide": 7 * n_tiles}
        if self.mode not in mode_actions:
            raise ValueError(f"Unknown mode: {self.mode}")
        self.action_dim = mode_actions[self.mode]

        if self.run_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            critic_type = "NeuralCrit" if self.use_neural_critic else "OrigCrit"
            device_tag = "gpu" if self.device == "cuda" else "cpu"
            self.run_name = f"ppo_{self.mode}_{critic_type}_{device_tag}_{timestamp}"

        self.checkpoint_dir = os.path.join(
            script_dir, self.checkpoint_dir, self.run_name
        )
        self.log_file_name = f"ppo_training_{self.run_name}.log"

        # Construct the full path to the neural critic checkpoint
        # <<< Path targets 'latest_checkpoint.pth' specifically >>>
        relative_chkpt_path = os.path.join(
            self.neural_critic_base_dir,
            self.neural_critic_run_name,
            "latest_checkpoint.pth",  # <<< Explicitly uses this filename
        )
        self.neural_critic_checkpoint_path = os.path.join(base_dir, relative_chkpt_path)


def configure_from_yaml(yaml_path: str) -> PPOConfig:
    """Loads configuration from a YAML file, filtering unknown keys."""
    try:
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        valid_keys = {f.name for f in fields(PPOConfig) if f.init}
        # <<< Force device to cuda if specified in YAML, otherwise use class default >>>
        if "device" in yaml_config and yaml_config["device"] != "cuda":
            print(
                f"Warning: YAML specified device '{yaml_config['device']}', overriding to 'cuda' as requested."
            )
        # Force 'cuda' unless YAML explicitly sets something else (and isn't 'cuda')
        # Retain the YAML value if it's *not* 'cuda', otherwise stick with 'cuda' from default/YAML
        yaml_device = yaml_config.get(
            "device", "cuda"
        )  # Default to cuda if not in yaml
        if yaml_device != "cuda":
            print(
                f"Warning: YAML config specified device '{yaml_device}'. Script requires 'cuda'. Forcing 'cuda'."
            )
            # Override YAML value if it's not cuda
            filtered_config = {
                k: v
                for k, v in yaml_config.items()
                if k in valid_keys and k != "device"
            }
            filtered_config["device"] = "cuda"  # Force cuda
        else:
            # If YAML has 'cuda' or no device field, the default 'cuda' will be used or confirmed
            filtered_config = {k: v for k, v in yaml_config.items() if k in valid_keys}
            if (
                "device" not in filtered_config
            ):  # Ensure device is in config if not in yaml
                filtered_config["device"] = "cuda"

        # Ensure boolean flags are parsed correctly from YAML
        if "use_neural_critic" in filtered_config:
            filtered_config["use_neural_critic"] = bool(
                filtered_config["use_neural_critic"]
            )
        if "fallback_to_original_critic_on_fail" in filtered_config:
            filtered_config["fallback_to_original_critic_on_fail"] = bool(
                filtered_config["fallback_to_original_critic_on_fail"]
            )

        return PPOConfig(**filtered_config)
    except FileNotFoundError:
        print(
            f"Warning: YAML config file not found at {yaml_path}. Using default config (forcing CUDA)."
        )
        return PPOConfig()  # Default already forces CUDA and runs validation
    except Exception as e:
        print(f"Error loading YAML config: {e}. Using default config (forcing CUDA).")
        return PPOConfig()  # Default already forces CUDA and runs validation


# --- Environment Utilities (Unchanged) ---
def get_device(device_str: str) -> torch.device:
    # Note: Validation happens in PPOConfig.__post_init__
    return torch.device(device_str)


def initialize_map_hero(
    config: PPOConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    map_tensor = torch.zeros(config.map_size, dtype=torch.int64)
    size_x, size_y = config.map_size
    start_pos = (random.randint(0, size_x - 1), random.randint(0, size_y - 1))
    current_pos = start_pos
    for _ in range(config.initial_map_walk_steps):
        tile_value = (
            1
            if random.random() < config.initial_map_empty_prob
            else random.randint(2, 5)  # Assuming original tile values 2-5
        )
        map_tensor[current_pos] = tile_value
        dx, dy = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        next_x = max(0, min(size_x - 1, current_pos[0] + dx))
        next_y = max(0, min(size_y - 1, current_pos[1] + dy))
        current_pos = (next_x, next_y)
    if map_tensor[start_pos] == 0:
        map_tensor[start_pos] = 1  # Ensure start is walkable
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


# --- Environment Simulation (Unchanged) ---
class BatchedEnvSimulator:
    def __init__(self, config: PPOConfig, critic_func: Callable):
        self.config = config
        self.num_envs = config.num_envs
        self.map_size_x, self.map_size_y = config.map_size
        self.critic_func = critic_func
        self.device = get_device(config.device)  # Uses validated device
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
            map_i, hero_i, pos_i = initialize_map_hero(self.config)
            self.maps[i] = map_i.to(self.device)
            self.heroes[i] = hero_i.to(self.device)
            self.agent_positions[i] = pos_i
            initial_maps_list_cpu.append(map_i.clone())
        initial_maps_cpu = torch.stack(initial_maps_list_cpu)
        map_obs, hero_obs = self._get_observation()
        return map_obs, hero_obs, initial_maps_cpu

    def _get_observation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        map_obs = self.maps.unsqueeze(1).float()  # Add channel dim for NN input
        hero_obs = self.heroes.float()
        return map_obs, hero_obs

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        actions_np = actions.cpu().numpy()
        current_maps_before_step = self.maps.clone()

        for i in range(self.num_envs):
            action = actions_np[i]
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
                if 0 <= action <= 5:
                    current_map[pos_x, pos_y] = action
                elif action == 6:  # UP
                    pos_x = (pos_x - 1 + self.map_size_x) % self.map_size_x
                elif action == 7:  # LEFT
                    pos_y = (pos_y - 1 + self.map_size_y) % self.map_size_y
                elif action == 8:  # DOWN
                    pos_x = (pos_x + 1) % self.map_size_x
                elif action == 9:  # RIGHT
                    pos_y = (pos_y + 1) % self.map_size_y
                self.agent_positions[i] = (pos_x, pos_y)
            elif mode == "wide":
                num_tile_types = 7
                n_tiles_total = self.map_size_x * self.map_size_y
                if self.config.action_dim != num_tile_types * n_tiles_total:
                    raise ValueError(
                        f"Action dim mismatch for wide mode. Check config."
                    )
                tile_type = action // n_tiles_total
                flat_index = action % n_tiles_total
                target_x = flat_index // self.map_size_y
                target_y = flat_index % self.map_size_y
                if 0 <= tile_type < num_tile_types:
                    current_map[target_x, target_y] = tile_type

        # Calculate rewards using the provided critic function
        rewards = self.critic_func(self.maps.long(), self.heroes.long())

        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        next_obs = self._get_observation()
        infos = {}
        return next_obs, rewards, dones, infos


# --- PPO Memory (Unchanged) ---
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
            raise IndexError("Memory buffer full")
        self.maps[self.ptr] = map_obs
        self.heroes[self.ptr] = hero_obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae_returns(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ):
        last_gae_lam = 0
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0
                next_values = last_value
            else:
                next_non_terminal = 1.0
                next_values = self.values[t + 1]
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        self.returns = self.advantages + self.values
        self.ptr = 0

    def get_minibatches(
        self, batch_size: int, minibatch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        if self.advantages is None or self.returns is None:
            raise ValueError("Advantages/returns not computed.")
        num_transitions = self.ptr * self.num_envs
        if num_transitions == 0:
            return
        if minibatch_size > num_transitions:
            print(
                f"Warning: Minibatch size ({minibatch_size}) > available transitions ({num_transitions}). Using {num_transitions} as minibatch size."
            )
            minibatch_size = num_transitions

        indices = torch.randperm(num_transitions).to(self.device)
        valid_maps = self.maps[: self.ptr].reshape(
            num_transitions, *self.maps.shape[2:]
        )
        valid_heroes = self.heroes[: self.ptr].reshape(num_transitions, -1)
        valid_actions = self.actions[: self.ptr].reshape(-1)
        valid_log_probs = self.log_probs[: self.ptr].reshape(-1)
        valid_advantages = self.advantages[: self.ptr].reshape(-1)
        valid_returns = self.returns[: self.ptr].reshape(-1)
        valid_values = self.values[: self.ptr].reshape(-1)

        for start_idx in range(0, num_transitions, minibatch_size):
            end_idx = start_idx + minibatch_size
            mb_indices = indices[start_idx:end_idx]
            if len(mb_indices) == 0:
                continue

            yield {
                "maps": valid_maps[mb_indices],
                "heroes": valid_heroes[mb_indices],
                "actions": valid_actions[mb_indices],
                "old_log_probs": valid_log_probs[mb_indices],
                "advantages": valid_advantages[mb_indices],
                "returns": valid_returns[mb_indices],
                "old_values": valid_values[mb_indices],
            }

    def clear(self):
        self.ptr = 0


# --- PPO Agent (Unchanged) ---
class PPOAgent:
    """The PPO Agent class containing policy/value networks and update logic."""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = get_device(config.device)  # Uses validated device
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
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
            eps=1e-5,
        )
        self.total_steps_interacted = 0
        self.total_updates_performed = 0

    def select_action(
        self, map_obs: torch.Tensor, hero_obs: torch.Tensor, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            action_logits = self.actor(map_obs, hero_obs)
            value = self.critic(map_obs, hero_obs).squeeze(-1)
            if temperature > 0:
                scaled_logits = action_logits / max(temperature, 1e-8)
            else:
                scaled_logits = action_logits
            probs = F.softmax(scaled_logits, dim=-1)
            dist = Categorical(probs=probs)
            action = dist.sample() if temperature > 0 else torch.argmax(probs, dim=-1)
            log_prob = dist.log_prob(action)
        self.actor.train()
        self.critic.train()
        return action, log_prob, value

    def evaluate_actions(
        self, map_obs: torch.Tensor, hero_obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits = self.actor(map_obs, hero_obs)
        value = self.critic(map_obs, hero_obs).squeeze(-1)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs=probs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy

    def update(self, memory: PPOMemory) -> Dict[str, float]:
        if memory.ptr == 0:
            return {}

        with torch.no_grad():
            last_map_obs = memory.maps[memory.ptr - 1]
            last_hero_obs = memory.heroes[memory.ptr - 1]
            last_value = self.critic(last_map_obs, last_hero_obs).squeeze(-1)
        memory.compute_gae_returns(
            last_value, self.config.gamma, self.config.gae_lambda
        )

        all_metrics = {"policy_loss": [], "value_loss": [], "entropy": []}

        for _ in range(self.config.num_epochs_per_update):
            minibatch_generator = memory.get_minibatches(
                memory.ptr * self.config.num_envs, self.config.minibatch_size
            )
            for batch in minibatch_generator:
                mb_maps, mb_heroes, mb_actions = (
                    batch["maps"],
                    batch["heroes"],
                    batch["actions"],
                )
                mb_old_log_probs, mb_advantages, mb_returns = (
                    batch["old_log_probs"],
                    batch["advantages"],
                    batch["returns"],
                )

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )
                new_log_probs, new_values, entropy = self.evaluate_actions(
                    mb_maps, mb_heroes, mb_actions
                )

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                clip_adv = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * mb_advantages
                )
                policy_loss = -(torch.min(ratio * mb_advantages, clip_adv)).mean()
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

        self.total_updates_performed += 1
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items() if v}
        return avg_metrics

    def save_checkpoint(self, path: str, episode: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "episode": episode,
            "total_steps_interacted": self.total_steps_interacted,
            "total_updates_performed": self.total_updates_performed,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            print(
                f"Warning: Checkpoint file not found at {path}. Starting from episode 0."
            )
            return 0
        try:
            # <<< Load directly to the agent's device >>>
            checkpoint = torch.load(path, map_location=self.device)
            chk_config = checkpoint.get("config")
            if chk_config and isinstance(chk_config, PPOConfig):
                if (
                    chk_config.action_dim != self.config.action_dim
                    or chk_config.mode != self.config.mode
                ):
                    print(
                        f"Warning: Config mismatch (ActionDim/Mode) in checkpoint vs current. Loading weights anyway."
                    )
                # <<< Check if loaded checkpoint device matches current config >>>
                if (
                    hasattr(chk_config, "device")
                    and chk_config.device != self.config.device
                ):
                    print(
                        f"Warning: Checkpoint was saved for device '{chk_config.device}', but current config is '{self.config.device}'. Loading anyway."
                    )
            elif chk_config:
                print(
                    "Warning: Config loaded from checkpoint is not a PPOConfig instance. Skipping detailed checks."
                )
            else:
                print("Warning: No config found in checkpoint.")

            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.total_steps_interacted = checkpoint.get("total_steps_interacted", 0)
            self.total_updates_performed = checkpoint.get("total_updates_performed", 0)
            start_episode = checkpoint.get("episode", -1) + 1

            print(
                f"Checkpoint loaded from {path}. Resuming from episode {start_episode} "
                # <<< Corrected reference to self.total_steps_interacted >>>
                f"(Updates: {self.total_updates_performed}, Steps: {self.total_steps_interacted:,})."
            )
            return start_episode
        except Exception as e:
            print(f"Error loading checkpoint from {path}: {e}. Starting from scratch.")
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=self.config.learning_rate,
                eps=1e-5,
            )
            self.total_steps_interacted = 0
            self.total_updates_performed = 0
            return 0


# --- Visualization (Unchanged) ---
def print_map(console: Console, map_tensor: torch.Tensor, title: str = "Generated Map"):
    if map_tensor.ndim == 3 and map_tensor.shape[0] == 1:
        map_tensor = map_tensor.squeeze(0)
    elif map_tensor.ndim == 4 and map_tensor.shape[0] == 1 and map_tensor.shape[1] == 1:
        map_tensor = map_tensor.squeeze(0).squeeze(0)
    elif map_tensor.ndim == 2:
        pass
    else:
        console.print(
            f"[red]Warning: Unexpected map dimensions for printing: {map_tensor.shape}[/red]"
        )
        return

    if map_tensor.device != torch.device("cpu"):
        map_tensor = map_tensor.cpu()
    map_np = map_tensor.numpy().astype(int)
    map_size_x, map_size_y = map_np.shape
    colors = {
        0: "grey50",
        1: "white",
        2: "bright_red",
        3: "red",
        4: "dark_red",
        5: "red3",
        6: "bright_green",
    }
    default_color, char_width = "magenta", 2
    table = Table(
        title=title,
        show_header=False,
        show_edge=True,
        box=None,
        padding=0,
        expand=False,
    )
    for _ in range(map_size_y):
        table.add_column(justify="center", width=char_width)
    for r in range(map_size_x):
        row_cells = [
            f"[{colors.get(tile, default_color)}]{tile:>{char_width - 1}}[/]"
            for tile in map_np[r]
        ]
        table.add_row(*row_cells)
    console.print(table)


# --- Training Orchestrator (Unchanged Internally) ---
class PPOTrainer:
    """Orchestrates the PPO training process with episodic resets."""

    def __init__(self, config: PPOConfig, critic_func: Callable):
        self.config = config
        self.critic_func = critic_func
        self.device = get_device(config.device)  # Uses validated device
        self.console = Console()
        self.logger = logging.getLogger(__name__)

        # Seed everything
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        # <<< Seed CUDA specifically if using it >>>
        if self.device.type == "cuda":
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)  # for multi-GPU
            # Potentially add determinism settings (can impact performance)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

        # Initialize components
        self.env = BatchedEnvSimulator(config, critic_func)
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

        self._setup_logging()

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
        self.logger.info(f"Full Config: {self.config}")

    def train(self):
        cfg = self.config
        self.console.print(
            Panel.fit(
                f"Starting PPO Episodic Training: mode='{cfg.mode}', run='{cfg.run_name}'",
                title="Setup",
                border_style="blue",
            )
        )
        critic_source = "Neural Critic" if cfg.use_neural_critic else "Original Critic"
        if cfg.use_neural_critic:
            critic_source += f" ({os.path.basename(cfg.neural_critic_run_name)})"
        self.console.print(f"Environment Critic: [bold yellow]{critic_source}[/]")
        # <<< Confirm CUDA device >>>
        self.console.print(
            f"Device: [bold cyan]{self.device}[/], Episodes: {cfg.num_episodes}, Steps/Episode: {cfg.episode_length}"
        )
        self.console.print(
            f"Envs: {cfg.num_envs}, Steps/Rollout: [bold yellow]{cfg.n_steps_per_rollout}[/], PPO Epochs: {cfg.num_epochs_per_update}, Minibatch: {cfg.minibatch_size}"
        )
        self.console.print(f"Total Transitions per Update: {cfg.batch_size}")
        self.console.print(
            f"Checkpoints & Logs Dir: [green]{cfg.checkpoint_dir}[/green]"
        )
        self.logger.info("Training setup complete.")

        latest_checkpoint_path = os.path.join(
            cfg.checkpoint_dir, "latest_checkpoint.pth"
        )
        if os.path.exists(latest_checkpoint_path):
            self.logger.info(
                f"Loading PPO agent checkpoint from {latest_checkpoint_path}"
            )
            self.start_episode = self.agent.load_checkpoint(latest_checkpoint_path)
            self.logger.info(f"Resuming PPO training from episode {self.start_episode}")
            if self.start_episode >= cfg.num_episodes:
                msg = f"Checkpoint indicates training already completed ({self.start_episode}/{cfg.num_episodes} episodes)."
                self.console.print(f"[yellow]{msg} Exiting.[/yellow]")
                self.logger.warning(msg)
                return

        episode_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            TextColumn("[bold]Metrics:[/]{task.fields[metrics]}", justify="left"),
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
                episode_rewards = []
                steps_this_episode = 0
                last_update_metrics = {}

                if (
                    episode + 1
                ) % cfg.print_maps_freq == 0 or episode == self.start_episode:
                    num_maps_to_print = min(3, cfg.num_envs)
                    episode_progress.console.print(
                        Panel(
                            f"--- Episode {episode + 1}: Initial Maps ---", expand=False
                        )
                    )
                    for i in range(num_maps_to_print):
                        print_map(
                            episode_progress.console,
                            initial_maps_cpu[i],
                            title=f"Ep {episode + 1} Initial (Env {i})",
                        )
                    episode_progress.console.print("-" * (cfg.map_size[1] * 3 + 4))

                steps_collected_in_episode = 0
                while steps_collected_in_episode < cfg.episode_length:
                    self.memory.clear()
                    rollout_start_step = steps_collected_in_episode
                    steps_to_run_this_rollout = min(
                        cfg.n_steps_per_rollout, cfg.episode_length - rollout_start_step
                    )
                    if steps_to_run_this_rollout <= 0:
                        break

                    for _ in range(steps_to_run_this_rollout):
                        action, log_prob, value = self.agent.select_action(
                            map_obs, hero_obs, cfg.temperature
                        )
                        next_obs_tuple, reward, done, info = self.env.step(action)
                        next_map_obs, next_hero_obs = next_obs_tuple
                        self.memory.add(
                            map_obs, hero_obs, action, log_prob, reward, done, value
                        )
                        map_obs, hero_obs = next_map_obs, next_hero_obs
                        episode_rewards.append(reward.mean().item())
                        self.agent.total_steps_interacted += cfg.num_envs
                        steps_collected_in_episode += 1

                    if self.memory.ptr > 0:
                        update_metrics = self.agent.update(self.memory)
                        last_update_metrics = update_metrics
                        if update_metrics:
                            self.logger.info(
                                f"Ep: {episode + 1}, Rollout Steps: {rollout_start_step + 1}-{steps_collected_in_episode}, "
                                f"Update#: {self.agent.total_updates_performed}, "
                                f"P_Loss: {update_metrics.get('policy_loss', float('nan')):.4f}, "
                                f"V_Loss (PPO): {update_metrics.get('value_loss', float('nan')):.4f}, "
                                f"Entropy: {update_metrics.get('entropy', float('nan')):.4f}"
                            )
                        else:
                            self.logger.warning(
                                f"Ep: {episode + 1} - PPO update skipped, no data (ptr={self.memory.ptr})."
                            )

                avg_ep_reward = np.mean(episode_rewards) if episode_rewards else 0
                if self.is_first_episode_for_baseline:
                    self.reward_baseline = avg_ep_reward
                    self.is_first_episode_for_baseline = False
                else:
                    self.reward_baseline = (
                        cfg.reward_baseline_alpha * avg_ep_reward
                        + (1 - cfg.reward_baseline_alpha) * self.reward_baseline
                    )

                self.logger.info(
                    f"Ep: {episode + 1}/{cfg.num_episodes} finished. AvgReward (Env Critic): {avg_ep_reward:.4f}, "
                    f"RewardBaseline: {self.reward_baseline:.4f}, TotalSteps: {self.agent.total_steps_interacted:,}"
                )

                p_loss = last_update_metrics.get("policy_loss", float("nan"))
                v_loss = last_update_metrics.get("value_loss", float("nan"))
                ent = last_update_metrics.get("entropy", float("nan"))
                metrics_str = (
                    f"AvgRew:[yellow]{avg_ep_reward:>7.3f}[/]| Baseline:[cyan]{self.reward_baseline:>7.3f}[/]| "
                    f"P_Loss:[red]{p_loss:>7.3f}[/]| V_Loss(PPO):[magenta]{v_loss:>7.3f}[/]| Ent:[blue]{ent:>6.3f}[/]"
                )
                episode_progress.update(episode_task, advance=1, metrics=metrics_str)

                if (
                    episode + 1
                ) % cfg.print_maps_freq == 0 or episode == cfg.num_episodes - 1:
                    num_maps_to_print = min(3, cfg.num_envs)
                    final_maps_cpu = self.env.maps.detach().cpu()
                    episode_progress.console.print(
                        Panel(
                            f"--- Episode {episode + 1}: Final Maps ---", expand=False
                        )
                    )
                    for i in range(num_maps_to_print):
                        print_map(
                            episode_progress.console,
                            final_maps_cpu[i],
                            title=f"Ep {episode + 1} Final (Env {i})",
                        )
                    episode_progress.console.print("-" * (cfg.map_size[1] * 3 + 4))

                if (
                    episode + 1
                ) % cfg.save_checkpoint_freq == 0 or episode == cfg.num_episodes - 1:
                    chk_path = os.path.join(
                        cfg.checkpoint_dir, f"checkpoint_ep_{episode + 1}.pth"
                    )
                    self.agent.save_checkpoint(chk_path, episode)
                    self.logger.info(
                        f"PPO agent checkpoint saved to {chk_path} at episode {episode + 1}"
                    )
                    latest_path = os.path.join(
                        cfg.checkpoint_dir, "latest_checkpoint.pth"
                    )
                    temp_latest_path = latest_path + ".tmp"
                    try:
                        self.agent.save_checkpoint(temp_latest_path, episode)
                        os.replace(temp_latest_path, latest_path)
                        self.logger.debug(
                            f"Updated latest checkpoint link to episode {episode + 1}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to save latest checkpoint atomically: {e}"
                        )
                        if os.path.exists(temp_latest_path):
                            os.remove(temp_latest_path)

        msg = f"Training finished after {cfg.num_episodes} episodes ({self.agent.total_steps_interacted:,} total steps, {self.agent.total_updates_performed:,} updates)."
        self.console.print(Panel(msg, title="Complete", border_style="green"))
        self.logger.info(msg)
        final_chk_path = os.path.join(cfg.checkpoint_dir, "latest_checkpoint.pth")
        self.console.print(
            f"Final PPO agent state saved in: [green]{final_chk_path}[/green]"
        )
        self.logger.info(f"Final PPO agent state saved in {final_chk_path}")


# --- Neural Critic Loading and Wrapping ---
def load_neural_critic(
    checkpoint_path: str, device: torch.device
) -> Optional[Tuple[nn.Module, NeuralCriticConfig]]:
    """Loads a pre-trained neural critic model (CNN or MLP) from a checkpoint."""
    console = Console()
    if not CRITIC_IMPORTS_SUCCESSFUL:
        console.print(
            "[bold red]Cannot load neural critic: Required imports failed.[/bold red]"
        )
        return None
    # <<< Use the provided checkpoint_path which already points to latest_checkpoint.pth >>>
    if not os.path.exists(checkpoint_path):
        console.print(
            f"[bold red]Error:[/bold red] Neural critic checkpoint not found at: {checkpoint_path}"
        )
        return None

    try:
        console.print(
            f"Loading neural critic checkpoint from: [cyan]{checkpoint_path}[/cyan]"
        )
        # <<< Load directly to the target device (CUDA) >>>
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "config" not in checkpoint or not isinstance(
            checkpoint["config"], NeuralCriticConfig
        ):
            console.print(
                "[bold red]Error:[/bold red] Checkpoint does not contain a valid NeuralCriticConfig."
            )
            return None
        nc_config: NeuralCriticConfig = checkpoint["config"]
        console.print(
            f"  Loaded Neural Critic Config: Type={'MLP' if nc_config.use_mlp_critic else 'CNN'}, MapSize={nc_config.map_size}"
        )

        model_type = "MLP" if nc_config.use_mlp_critic else "CNN"
        if nc_config.use_mlp_critic:
            map_h, map_w = nc_config.map_size
            model = CriticApproximatorMLP(
                input_size=map_h * map_w,
                hidden_sizes=nc_config.mlp_hidden_sizes,
                output_size=1,
                dropout_prob=nc_config.mlp_dropout_prob,
            )
            console.print(f"  Instantiated CriticApproximatorMLP.")
        else:
            model = CriticApproximatorCNN(
                channels=nc_config.cnn_channels,
                input_size=nc_config.map_size,
                hidden_fc_size=nc_config.cnn_fc_hidden_size,
                output_size=1,
            )
            console.print(f"  Instantiated CriticApproximatorCNN.")

        model.load_state_dict(checkpoint["model_state_dict"])
        # <<< Ensure model is on the correct device (CUDA) >>>
        model.to(device)
        model.eval()

        console.print(
            f"[green]Successfully loaded neural critic model ({model_type}) weights to {device}.[/green]"
        )
        return model, nc_config

    except Exception as e:
        console.print(f"[bold red]Error loading neural critic checkpoint:[/bold red]")
        console.print_exception(show_locals=True)
        return None


def create_neural_critic_wrapper(
    neural_critic_model: nn.Module, device: torch.device
) -> Callable:
    """Creates a wrapper function compatible with BatchedEnvSimulator's critic_func."""

    def wrapped_critic(
        maps_batch: torch.Tensor, heroes_batch: torch.Tensor
    ) -> torch.Tensor:
        neural_critic_model.eval()
        with torch.no_grad():
            # <<< Ensure input maps are moved to CUDA device >>>
            maps_batch_device = maps_batch.to(device)
            if maps_batch_device.ndim == 3:
                maps_nn_input = maps_batch_device.unsqueeze(1).float()
            elif maps_batch_device.ndim == 4 and maps_batch_device.shape[1] == 1:
                maps_nn_input = maps_batch_device.float()
            else:
                raise ValueError(
                    f"Unexpected map shape for neural critic wrapper: {maps_batch.shape}"
                )
            scores = neural_critic_model(maps_nn_input)
            scores = scores.squeeze(-1)
            # <<< Scores are already on the correct device (CUDA) >>>
            return scores

    return wrapped_critic


# --- Main Execution ---
if __name__ == "__main__":
    console = Console()

    # --- Configuration Loading (Forces CUDA, validates) ---
    config_path = "ppo_episodic_config.yaml"
    config_path_abs = os.path.join(script_dir, config_path)
    try:
        if os.path.exists(config_path_abs):
            console.print(
                f"Loading PPO configuration from [cyan]{config_path_abs}[/cyan]"
            )
            config = configure_from_yaml(config_path_abs)
        else:
            console.print(
                f"[yellow]PPO configuration file '{config_path_abs}' not found. Using default settings (forcing CUDA).[/yellow]"
            )
            config = PPOConfig()  # Will validate CUDA in __post_init__
    except RuntimeError as e:  # Catch CUDA validation error from PPOConfig
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error loading configuration:[/bold red] {e}")
        sys.exit(1)

    # --- Initialize Critic ---
    actual_critic = None
    if not CRITIC_IMPORTS_SUCCESSFUL and (
        config.use_neural_critic or not config.fallback_to_original_critic_on_fail
    ):
        console.print(
            "[bold red]Cannot proceed: Critic imports failed. Exiting.[/bold red]"
        )
        sys.exit(1)

    # <<< Get the validated CUDA device >>>
    cuda_device = get_device(config.device)
    console.print(f"Ensured CUDA device: [bold cyan]{cuda_device}[/]")

    if config.use_neural_critic:
        console.print("Attempting to load and use the Neural Critic...")
        # <<< Pass the validated CUDA device to loader >>>
        # config.neural_critic_checkpoint_path already points to the specific 'latest_checkpoint.pth'
        load_result = load_neural_critic(
            config.neural_critic_checkpoint_path, cuda_device
        )

        if load_result:
            neural_critic_model, loaded_nc_config = load_result
            if loaded_nc_config.map_size != config.map_size:
                console.print(
                    f"[bold red]CRITICAL ERROR:[/bold red] PPO config map size {config.map_size} "
                    f"does not match loaded Neural Critic map size {loaded_nc_config.map_size}."
                )
                sys.exit(1)
            # <<< Pass validated CUDA device to wrapper >>>
            actual_critic = create_neural_critic_wrapper(
                neural_critic_model, cuda_device
            )
            console.print(
                "[green]Neural Critic loaded and wrapped successfully for CUDA.[/green]"
            )
        else:
            console.print("[bold red]Failed to load the Neural Critic.[/bold red]")
            if config.fallback_to_original_critic_on_fail:
                console.print(
                    "[yellow]Falling back to the original level_critic function.[/yellow]"
                )
                actual_critic = original_level_critic
            else:
                console.print("[bold red]Fallback disabled. Exiting.[/bold red]")
                sys.exit(1)
    else:
        console.print("Using the original [cyan]level_critic[/cyan] function.")
        actual_critic = original_level_critic

    if actual_critic is None:
        console.print(
            "[bold red]Critic function could not be initialized. Exiting.[/bold red]"
        )
        sys.exit(1)

    # --- Create and Run Trainer ---
    trainer = PPOTrainer(config, actual_critic)  # Config already validated for CUDA
    try:
        trainer.train()
    except KeyboardInterrupt:
        # --- Keyboard Interrupt Handling (Ensure saving happens) ---
        msg = "\nTraining interrupted by user. Saving final PPO agent checkpoint..."
        trainer.console.print(f"[yellow]{msg}[/yellow]")
        trainer.logger.warning(msg)
        latest_episode = -1
        latest_checkpoint_path = os.path.join(
            trainer.config.checkpoint_dir, "latest_checkpoint.pth"
        )
        # Try loading to CPU to avoid CUDA issues during shutdown
        if os.path.exists(latest_checkpoint_path):
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
                latest_episode = checkpoint.get("episode", -1)
            except Exception as e:
                trainer.logger.error(
                    f"Could not read latest checkpoint episode on interrupt: {e}"
                )
        interrupted_chk_path = os.path.join(
            trainer.config.checkpoint_dir, "final_checkpoint_interrupted.pth"
        )
        try:
            trainer.agent.save_checkpoint(interrupted_chk_path, latest_episode)
            final_msg = f"Interrupted state PPO agent checkpoint saved to {interrupted_chk_path} (ep {latest_episode + 1})"
            print(final_msg)
            trainer.logger.info(final_msg)
        except Exception as save_e:
            final_msg = f"Failed to save interrupted checkpoint: {save_e}"
            print(f"[bold red]{final_msg}[/bold red]")
            trainer.logger.error(final_msg)
    except Exception as e:
        # --- Generic Exception Handling (Ensure saving happens) ---
        trainer.console.print(
            "[bold red]An unexpected error occurred during training![/bold red]"
        )
        trainer.console.print_exception(show_locals=True)
        trainer.logger.error("An error occurred during training.", exc_info=True)
        print(
            "\n[red]Attempting to save final PPO agent checkpoint due to error...[/red]"
        )
        latest_episode = -1
        latest_checkpoint_path = os.path.join(
            trainer.config.checkpoint_dir, "latest_checkpoint.pth"
        )
        # Try loading to CPU to avoid CUDA issues during shutdown
        if os.path.exists(latest_checkpoint_path):
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
                latest_episode = checkpoint.get("episode", -1)
            except Exception as load_e:
                trainer.logger.error(
                    f"Could not read latest checkpoint episode during error handling: {load_e}"
                )
        error_chk_path = os.path.join(
            trainer.config.checkpoint_dir, "final_checkpoint_error.pth"
        )
        try:
            trainer.agent.save_checkpoint(error_chk_path, latest_episode)
            error_msg = f"Error state PPO agent checkpoint saved to {error_chk_path} (ep {latest_episode + 1})"
            print(error_msg)
            trainer.logger.info(error_msg)
        except Exception as save_e:
            error_msg = f"Failed to save error checkpoint: {save_e}"
            print(f"[bold red]{error_msg}[/bold red]")
            trainer.logger.error(error_msg)

    # --- Optional Post-Training Test (PPO Agent Networks) ---
    console.print(
        "\n--- Optional: Testing PPO Network Forward Pass (Final State on CUDA) ---"
    )
    try:
        test_agent = trainer.agent  # Agent models should already be on CUDA device
        test_device = trainer.device  # Should be cuda
        console.print(f"Testing on device: [cyan]{test_device}[/]")

        test_map_obs = torch.rand((2, 1, config.map_size[0], config.map_size[1])).to(
            test_device
        )
        test_hero_obs = torch.rand((2, config.hero_tensor_size)).to(test_device)

        test_agent.actor.eval()
        test_agent.critic.eval()
        with torch.no_grad():
            logits = test_agent.actor(test_map_obs, test_hero_obs)
            values = test_agent.critic(test_map_obs, test_hero_obs)
            probs = F.softmax(logits, dim=-1)

        console.print(f"PPO Mode: {config.mode}, Action Dim: {config.action_dim}")
        console.print(
            f"Input Map Obs: {test_map_obs.shape}, Input Hero Obs: {test_hero_obs.shape}"
        )
        console.print(f"Output Logits: {logits.shape}, Output Probs: {probs.shape}")
        console.print(f"Output PPO Values: {values.shape}")
        assert logits.shape == (2, config.action_dim), "Actor output shape mismatch!"
        assert values.shape == (2, 1), "PPO Critic output shape mismatch!"
        console.print(
            "[green]PPO Network forward pass test successful on CUDA.[/green]"
        )

    except AttributeError:
        console.print(
            "[yellow]Trainer object or agent not fully initialized, skipping PPO network test.[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]PPO Network forward pass test failed: {e}[/red]")
        console.print_exception(show_locals=False)
