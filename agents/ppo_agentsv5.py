# --- START OF FILE ppo_agentsv4.py --- # Changed name slightly for clarity

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
import argparse  # Added for config path argument

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
    TaskID,  # Added for type hinting
)
# from tqdm import tqdm # Removed tqdm as Rich progress bar is used for steps now

# --- Import Critic Components ---
# Assume running from base_directory, adjust path if necessary
try:
    # Import the original symbolic critic
    from daedalus.critics import level_critic as actual_critic

    # Import the neural critic approximation components
    # Assuming critic_approximator.py is findable via PYTHONPATH or relative structure
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
        # Adjust path assuming agents/ and critics/ are siblings relative to base_directory
        critics_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "critics")
        )
        # Need base directory path as well if daedalus is not installed
        base_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Add both to sys.path, prioritizing critics for direct import
        sys.path.insert(0, critics_path)
        sys.path.insert(0, base_dir_path)  # For 'from daedalus.critics...'

        # Retry imports
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


from enum import Enum


# --- Mock Entry Enum (Keep as is) ---
class Entry(Enum):
    TOP = 0
    LEFT = 1
    BOTTOM = 2
    RIGHT = 3


# --- Model Definition (Keep as is) ---
# PolicyNetworkEncoder, PolicyNetworkDecoder, PPOPolicyNetwork, ValueNetwork
class PolicyNetworkEncoder(nn.Module):
    """Encodes the map and hero state into a latent representation using a NN (MLP)."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (12, 12),
        output_size: int = 1024,
        hero_tensor_size: int = 5,
        hidden_dims: List[int] = [1024, 2048],
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
        """Forward pass for the NN encoder."""
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
        layers = []
        current_dim = input_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_size))
        self.net = nn.Sequential(*layers)
        # Using Sequential is often cleaner than defining individual layers if linear flow
        # self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        # self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        # self.fc_out = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the decoder."""
        return self.net(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc_out(x)
        # return x


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
        """Forward pass returning action logits."""
        # Ensure map_tensor has channel dim: (N, 1, H, W)
        if map_tensor.dim() == 3:
            map_tensor = map_tensor.unsqueeze(1)

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
        """Forward pass returning state value prediction."""
        # Ensure map_tensor has channel dim: (N, 1, H, W)
        if map_tensor.dim() == 3:
            map_tensor = map_tensor.unsqueeze(1)

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

    # >> New << Training Loop Control
    num_episodes: int = 1000
    episode_length: int = 256

    # PPO Core Parameters
    num_envs: int = 128
    n_steps_per_rollout: int = 64
    num_epochs_per_update: int = 4
    minibatch_size: int = 64  # Recommended to be a divisor of batch_size

    # PPO Algorithm Hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Action Space & Exploration
    temperature: float = 0.5  # For sampling during rollouts

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
    reward_baseline_alpha: float = 0.05  # EMA smoothing for reward baseline logging

    # Model architecture
    encoder_output_size: int = 1024
    decoder_hidden_sizes: List[int] = field(default_factory=lambda: [512, 256])
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 512])

    # --- Neural Critic Integration ---
    use_neural_critic: bool = True
    # Default path relative to the base directory (where you run the script)
    # MAKE SURE THIS PATH IS CORRECT FOR YOUR SETUP
    neural_critic_checkpoint_path: str = "critics/neural_critic_checkpoints/critic_MLP_20250422_113333/latest_checkpoint.pth"

    # Calculated properties
    batch_size: int = field(init=False)
    action_dim: int = field(init=False)

    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.batch_size = self.num_envs * self.n_steps_per_rollout
        if self.batch_size % self.minibatch_size != 0:
            print(
                f"Warning: Minibatch size ({self.minibatch_size}) is not a divisor of the total batch size ({self.batch_size}). This can lead to incomplete batches during update."
            )
            # Optionally adjust minibatch_size here, e.g., find closest divisor or default to batch_size
            # self.minibatch_size = self.batch_size # Example: Use full batch if not divisible

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
    """Loads configuration from a YAML file, filtering unknown keys."""
    try:
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        if not isinstance(yaml_config, dict):
            raise TypeError(f"YAML file {yaml_path} did not parse into a dictionary.")

        valid_keys = {f.name for f in fields(PPOConfig) if f.init}
        filtered_config = {k: v for k, v in yaml_config.items() if k in valid_keys}

        # Ensure boolean conversion if loading from YAML
        if "use_neural_critic" in filtered_config:
            val = filtered_config["use_neural_critic"]
            filtered_config["use_neural_critic"] = str(val).lower() in [
                "true",
                "1",
                "yes",
                "y",
            ]

        # Convert map_size if it's a list in YAML
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


# --- Environment Utilities (Identical) ---
def get_device(device_str: str) -> torch.device:
    """Gets the torch device object."""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def initialize_map_hero(
    config: PPOConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Initializes a random map, hero tensor, and agent start position."""
    map_tensor = torch.zeros(config.map_size, dtype=torch.int64)
    size_x, size_y = config.map_size
    if size_x <= 0 or size_y <= 0:
        return (
            map_tensor,
            torch.zeros(config.hero_tensor_size, dtype=torch.int64),
            (0, 0),
        )  # Handle invalid size

    start_pos = (random.randint(0, size_x - 1), random.randint(0, size_y - 1))
    current_pos = start_pos

    # Perform random walk to generate initial map structure
    for _ in range(config.initial_map_walk_steps):
        # Tile values: 1 (empty), 2-5 (enemies - check critic alignment later)
        tile_value = (
            (1 if random.random() < 0.9 else 6)
            if random.random() < config.initial_map_empty_prob
            else random.randint(2, 5)  # Assuming 2-5 are valid enemy types for now
        )
        map_tensor[current_pos] = tile_value

        # Move to adjacent or diagonal tile
        dx, dy = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        next_x = max(0, min(size_x - 1, current_pos[0] + dx))
        next_y = max(0, min(size_y - 1, current_pos[1] + dy))
        current_pos = (next_x, next_y)

    # Ensure the agent's starting position is walkable (tile value 1)
    if map_tensor[start_pos] == 0:  # If start pos wasn't visited by walk
        map_tensor[start_pos] = 1

    # Generate random hero tensor
    hero_tensor = torch.tensor(
        [
            random.randint(1, 10),  # Stat 1
            random.choice([0, 1]),  # Stat 2 (binary)
            random.choice([0, 1]),  # Stat 3 (binary)
            random.choice([e.value for e in Entry]),  # Entry direction
            random.randint(0, 10),  # Stat 5
        ],
        dtype=torch.int64,
    )
    return map_tensor, hero_tensor, start_pos


# --- Environment Simulation (Modified __init__) ---
class BatchedEnvSimulator:
    """Simulates multiple environments in parallel."""

    def __init__(self, config: PPOConfig, critic_function_to_use: Callable):
        self.config = config
        self.num_envs = config.num_envs
        self.map_size_x, self.map_size_y = config.map_size
        self.critic_func = critic_function_to_use  # Use the provided function
        self.device = get_device(config.device)

        # Initialize tensors on the target device
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
        # Store agent positions as a list of tuples (CPU is fine for this)
        self.agent_positions = [(0, 0)] * self.num_envs

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resets all environments and returns initial observations."""
        initial_maps_list_cpu = []  # Keep initial maps on CPU for potential display
        for i in range(self.num_envs):
            # Generate map/hero on CPU first
            map_i_cpu, hero_i_cpu, pos_i = initialize_map_hero(self.config)
            # Move to target device for simulation state
            self.maps[i] = map_i_cpu.to(self.device)
            self.heroes[i] = hero_i_cpu.to(self.device)
            self.agent_positions[i] = pos_i
            initial_maps_list_cpu.append(map_i_cpu)  # Store CPU version

        # Stack CPU maps for return (e.g., for visualization)
        initial_maps_cpu_batch = torch.stack(initial_maps_list_cpu)
        # Get initial observations (processed for NN input)
        map_obs, hero_obs = self._get_observation()
        return map_obs, hero_obs, initial_maps_cpu_batch

    def _get_observation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Formats the current state into observations suitable for the policy network."""
        # Map shape: (N, H, W) -> (N, 1, H, W), dtype float
        map_obs = self.maps.unsqueeze(1).float()
        # Hero shape: (N, hero_size), dtype float
        hero_obs = self.heroes.float()
        return map_obs, hero_obs

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, List[Dict]
    ]:
        """Applies actions to environments, returns next state, rewards, dones, infos."""
        # Ensure actions are on CPU for numpy conversion if needed, or process directly on device
        actions_device = actions.to(self.device)  # Keep actions on device if possible

        # --- Apply Actions and Update State ---
        for i in range(self.num_envs):
            action = actions_device[i].item()  # Get scalar action value
            pos_x, pos_y = self.agent_positions[i]
            current_map = self.maps[i]  # Direct reference to device tensor
            mode = self.config.mode

            # --- Action Interpretation based on Mode ---
            if mode == "narrow":
                # Action 0-6: Place tile type (Check critic alignment: 0:?, 1:Empty, 6:Door, 3,4,5:Enemy)
                if 0 <= action <= 6:
                    current_map[pos_x, pos_y] = action
                # Randomly move agent
                dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                next_x = max(0, min(self.map_size_x - 1, pos_x + dx))
                next_y = max(0, min(self.map_size_y - 1, pos_y + dy))
                self.agent_positions[i] = (next_x, next_y)

            elif mode == "turtle":
                # Actions 0-5: Place tile
                # Actions 6-9: Move Turtle (Up, Left, Down, Right - Bounded)
                if 0 <= action <= 6:
                    current_map[pos_x, pos_y] = action
                elif action == 7:
                    pos_x = max(0, pos_x - 1)  # Up
                elif action == 8:
                    pos_y = max(0, pos_y - 1)  # Left
                elif action == 9:
                    pos_x = min(self.map_size_x - 1, pos_x + 1)  # Down
                elif action == 10:
                    pos_y = min(self.map_size_y - 1, pos_y + 1)  # Right
                self.agent_positions[i] = (pos_x, pos_y)

            elif mode == "wide":
                num_tile_types = 7  # Assumes tiles 0-6
                n_tiles_total = self.map_size_x * self.map_size_y
                # Check if action dimension matches expectation (can be flexible if needed)
                # if self.config.action_dim != num_tile_types * n_tiles_total:
                #     print(f"Warning: Action dim mismatch for wide mode...")

                tile_type = action // n_tiles_total
                flat_index = action % n_tiles_total
                target_x = flat_index // self.map_size_y
                target_y = flat_index % self.map_size_y

                if 0 <= tile_type < num_tile_types:
                    # Check bounds for target coordinates just in case
                    if (
                        0 <= target_x < self.map_size_x
                        and 0 <= target_y < self.map_size_y
                    ):
                        current_map[target_x, target_y] = tile_type
                # Agent position typically doesn't change in 'wide' mode per action

            else:
                raise ValueError(f"Unknown mode in step: {mode}")

        # --- Calculate Rewards using the provided critic function ---
        # Pass device tensors directly to the critic function
        # The critic_func handles tensor formatting (e.g., unsqueeze, float) if it's the neural wrapper
        rewards = self.critic_func(self.maps, self.heroes)  # Pass device tensors
        # Ensure rewards are on the correct device and have shape (num_envs,)
        rewards = rewards.to(self.device).float().squeeze()
        if rewards.shape != (self.num_envs,):
            raise ValueError(
                f"Critic function returned unexpected rewards shape: {rewards.shape}. Expected ({self.num_envs},)"
            )

        # --- Determine Done Status ---
        # In this setup, episodes have fixed length, so 'dones' are always False during the episode.
        # Termination is handled by the training loop based on episode_length.
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- Get Next Observations ---
        next_map_obs, next_hero_obs = self._get_observation()

        # --- Info Dictionary (Placeholder) ---
        infos = [{} for _ in range(self.num_envs)]  # List of dicts, one per env

        return (next_map_obs, next_hero_obs), rewards, dones, infos


# --- PPO Memory (Identical) ---
class PPOMemory:
    """Stores transitions collected during rollouts for PPO updates."""

    def __init__(
        self,
        num_envs: int,
        n_steps: int,  # Steps per rollout per environment
        map_shape: Tuple[int, int],
        hero_shape: int,
        device: torch.device,
    ):
        self.n_steps, self.num_envs, self.device = n_steps, num_envs, device
        map_c, map_h, map_w = 1, map_shape[0], map_shape[1]  # Assume 1 channel for maps

        # Initialize buffers on the target device
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
        )  # Store dones for GAE calculation
        self.values = torch.zeros(
            (n_steps, num_envs), dtype=torch.float32, device=device
        )

        # Placeholders for calculated advantages and returns
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None
        self.ptr = 0  # Pointer to the current position in the buffer

    def add(
        self,
        map_obs: torch.Tensor,  # (num_envs, 1, H, W) float
        hero_obs: torch.Tensor,  # (num_envs, hero_size) float
        action: torch.Tensor,  # (num_envs,) int64
        log_prob: torch.Tensor,  # (num_envs,) float
        reward: torch.Tensor,  # (num_envs,) float
        done: torch.Tensor,  # (num_envs,) bool
        value: torch.Tensor,  # (num_envs,) float
    ):
        """Adds a transition for all environments to the memory buffer."""
        if self.ptr >= self.n_steps:
            # This should ideally not happen if cleared correctly, but good safety check
            raise IndexError(
                "Memory buffer overflow. Ensure clear() is called before starting new rollout collection."
            )

        # Store tensors at the current pointer position
        self.maps[self.ptr] = map_obs.to(self.device)  # Ensure on correct device
        self.heroes[self.ptr] = hero_obs.to(self.device)
        self.actions[self.ptr] = action.to(self.device)
        self.log_probs[self.ptr] = log_prob.to(self.device)
        self.rewards[self.ptr] = reward.to(self.device)
        self.dones[self.ptr] = done.to(self.device)
        self.values[self.ptr] = value.to(self.device)

        self.ptr += 1  # Increment pointer

    def compute_gae_returns(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ):
        """
        Computes Generalized Advantage Estimation (GAE) and returns (discounted rewards).
        'last_value' is the value estimate of the state reached *after* the last step in the buffer.
        """
        if self.ptr != self.n_steps:
            print(
                f"Warning: Computing GAE on incomplete buffer (ptr={self.ptr}, n_steps={self.n_steps})."
            )
            # Decide behavior: proceed with partial data or raise error? Proceeding for now.
            if self.ptr == 0:
                self.advantages = torch.zeros_like(self.rewards[: self.ptr])
                self.returns = torch.zeros_like(self.rewards[: self.ptr])
                return  # Nothing to compute

        # Ensure last_value is on the correct device
        last_value = last_value.to(self.device).squeeze()
        if last_value.shape != (self.num_envs,):
            raise ValueError(
                f"Expected last_value shape ({self.num_envs},), got {last_value.shape}"
            )

        last_gae_lam = 0
        # Allocate tensors for advantages and returns (only for the filled part of the buffer)
        num_steps_filled = self.ptr
        self.advantages = torch.zeros(
            (num_steps_filled, self.num_envs), dtype=torch.float32, device=self.device
        )

        # Iterate backwards through the collected steps
        for t in reversed(range(num_steps_filled)):
            if t == num_steps_filled - 1:
                # For the last step in the buffer, the 'next state' is the one whose value is 'last_value'
                next_non_terminal = (
                    1.0 - self.dones[t].float()
                )  # 1 if not done, 0 if done
                next_values = last_value
            else:
                # For other steps, the 'next state' is the state at step t+1
                next_non_terminal = (
                    1.0 - self.dones[t + 1].float()
                )  # Use done state from *next* step
                next_values = self.values[t + 1]

            # Calculate TD error (delta)
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            # Calculate GAE recursively
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam

        # Calculate returns by adding advantages to values
        self.returns = self.advantages + self.values[:num_steps_filled]

        # Reset pointer - buffer is now processed and ready for get_minibatches or clearing
        # It's generally better practice to let the caller decide when to clear.
        # self.ptr = 0

    def get_minibatches(
        self, batch_size: int, minibatch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Generates minibatches of transitions from the collected rollout data."""
        if self.advantages is None or self.returns is None:
            raise ValueError(
                "Advantages and returns must be computed before calling get_minibatches."
            )

        # Total number of transitions available (steps_filled * num_envs)
        num_steps_filled = self.advantages.shape[0]
        num_transitions = num_steps_filled * self.num_envs
        if num_transitions == 0:
            return  # No data to provide

        if num_transitions < batch_size:
            print(
                f"Warning: Requested batch_size {batch_size} > available transitions {num_transitions}. Using available transitions."
            )
            # This shouldn't happen if batch_size = n_steps * num_envs and buffer is full
            batch_size = num_transitions

        if minibatch_size > batch_size:
            print(
                f"Warning: Minibatch size ({minibatch_size}) > effective batch size ({batch_size}). Adjusting minibatch size."
            )
            minibatch_size = batch_size

        # Flatten the data buffers (select only the filled steps)
        # Reshape from (n_steps, num_envs, ...) to (batch_size, ...)
        flat_maps = self.maps[:num_steps_filled].reshape(
            num_transitions, *self.maps.shape[2:]
        )
        flat_heroes = self.heroes[:num_steps_filled].reshape(num_transitions, -1)
        flat_actions = self.actions[:num_steps_filled].reshape(-1)
        flat_log_probs = self.log_probs[:num_steps_filled].reshape(-1)
        flat_advantages = self.advantages.reshape(-1)  # Already computed
        flat_returns = self.returns.reshape(-1)  # Already computed
        flat_values = self.values[:num_steps_filled].reshape(-1)  # Old values

        # Generate random indices for minibatch sampling
        indices = torch.randperm(num_transitions).to(self.device)

        # Yield minibatches
        for start_idx in range(0, batch_size, minibatch_size):
            end_idx = start_idx + minibatch_size
            # Ensure end_idx does not exceed the number of available transitions
            actual_end_idx = min(end_idx, num_transitions)
            if start_idx >= actual_end_idx:
                continue  # Skip if start index is beyond available data

            mb_indices = indices[start_idx:actual_end_idx]

            if len(mb_indices) == 0:
                continue

            yield {
                "maps": flat_maps[mb_indices],
                "heroes": flat_heroes[mb_indices],
                "actions": flat_actions[mb_indices],
                "old_log_probs": flat_log_probs[mb_indices],
                "advantages": flat_advantages[mb_indices],
                "returns": flat_returns[mb_indices],
                "old_values": flat_values[
                    mb_indices
                ],  # Include old values for potential value clipping
            }

    def clear(self):
        """Resets the buffer pointer, making it ready for new data collection."""
        self.ptr = 0
        # Optionally, explicitly release computed advantages/returns
        self.advantages = None
        self.returns = None
        # Zeroing out tensors is usually not necessary unless memory reuse is critical
        # self.maps.zero_()
        # ... etc ...


# --- PPO Agent (Identical) ---
class PPOAgent:
    """The PPO Agent class containing policy/value networks and update logic."""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = get_device(config.device)

        # Initialize Actor and Critic networks
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

        # Combine parameters from both networks for the optimizer
        all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(
            all_params,
            lr=config.learning_rate,
            eps=1e-5,  # Epsilon for numerical stability
        )

        # Track training progress
        self.total_steps_interacted = 0
        self.total_updates_performed = 0

    def select_action(
        self, map_obs: torch.Tensor, hero_obs: torch.Tensor, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects actions based on policy, returns action, log_prob, and value estimate."""
        self.actor.eval()  # Set networks to evaluation mode for inference
        self.critic.eval()
        with torch.no_grad():  # Disable gradient calculations for inference
            # Get action logits from actor and value estimate from critic
            action_logits = self.actor(map_obs, hero_obs)
            value = self.critic(map_obs, hero_obs).squeeze(-1)  # Shape (num_envs,)

            # Apply temperature scaling for exploration
            if temperature > 0:
                scaled_logits = action_logits / max(
                    temperature, 1e-8
                )  # Avoid division by zero
            else:
                scaled_logits = (
                    action_logits  # Use original logits for argmax if temp <= 0
                )

            # Create action distribution
            probs = F.softmax(scaled_logits, dim=-1)
            dist = Categorical(probs=probs)

            # Sample action or take argmax based on temperature
            if temperature > 0:
                action = dist.sample()
            else:
                action = torch.argmax(probs, dim=-1)  # Deterministic action

            # Calculate log probability of the chosen action
            log_prob = dist.log_prob(action)

        self.actor.train()  # Set networks back to training mode
        self.critic.train()
        return action, log_prob, value

    def evaluate_actions(
        self, map_obs: torch.Tensor, hero_obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates actions given state, returns log_probs, values, and entropy."""
        # Get action logits and value estimates for the given states/actions
        action_logits = self.actor(map_obs, hero_obs)
        value = self.critic(map_obs, hero_obs).squeeze(-1)  # Shape (batch_size,)

        # Create action distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs=probs)

        # Calculate log probability of the *provided* actions
        log_prob = dist.log_prob(actions)
        # Calculate entropy of the action distribution (for entropy bonus)
        entropy = dist.entropy()

        return log_prob, value, entropy

    def update(self, memory: PPOMemory) -> Dict[str, float]:
        """Performs PPO update using data collected in the memory buffer."""
        if memory.advantages is None or memory.returns is None:
            raise RuntimeError(
                "memory.compute_gae_returns() must be called before update()"
            )

        # --- PPO Optimization Loop ---
        all_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        # Iterate multiple epochs over the collected data
        for _ in range(self.config.num_epochs_per_update):
            # Get minibatches from the memory buffer
            minibatch_generator = memory.get_minibatches(
                self.config.batch_size, self.config.minibatch_size
            )

            for batch in minibatch_generator:
                # Extract data from the minibatch dictionary
                mb_maps, mb_heroes = batch["maps"], batch["heroes"]
                mb_actions, mb_old_log_probs = batch["actions"], batch["old_log_probs"]
                mb_advantages, mb_returns = batch["advantages"], batch["returns"]
                # mb_old_values = batch["old_values"] # Uncomment if using value clipping

                # --- Normalize Advantages (per minibatch) ---
                # This is a common practice that can stabilize training
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # --- Evaluate current policy on minibatch data ---
                new_log_probs, new_values, entropy = self.evaluate_actions(
                    mb_maps, mb_heroes, mb_actions
                )

                # --- Policy Loss (Clipped Surrogate Objective) ---
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

                # --- Value Loss (MSE) ---
                # Optional: Value function clipping (uncomment relevant lines if needed)
                # value_loss_unclipped = F.mse_loss(new_values, mb_returns)
                # clipped_values = mb_old_values + torch.clamp(new_values - mb_old_values, -self.config.clip_epsilon, self.config.clip_epsilon)
                # value_loss_clipped = F.mse_loss(clipped_values, mb_returns)
                # value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped)

                # Standard MSE Value Loss:
                value_loss = 0.5 * F.mse_loss(new_values, mb_returns)

                # --- Entropy Loss ---
                # Encourages exploration by maximizing the policy's entropy
                entropy_loss = entropy.mean()

                # --- Total Loss ---
                # Combine losses, scaling value and entropy losses
                loss = (
                    policy_loss
                    - self.config.entropy_coef * entropy_loss
                    + self.config.vf_coef * value_loss
                )

                # --- Optimization Step ---
                self.optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Compute gradients
                # Apply gradient clipping to prevent large updates
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()  # Update network weights

                # --- Log Metrics (from this minibatch) ---
                all_metrics["policy_loss"].append(policy_loss.item())
                all_metrics["value_loss"].append(value_loss.item())
                all_metrics["entropy"].append(entropy_loss.item())

                # Calculate and log optional metrics for debugging/monitoring
                with torch.no_grad():
                    # Approximate KL divergence between old and new policies
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    # Fraction of samples where the policy ratio was clipped
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1.0) > self.config.clip_epsilon).float()
                    ).item()
                    all_metrics["approx_kl"].append(approx_kl)
                    all_metrics["clip_fraction"].append(clip_fraction)

        # --- Aggregate Metrics (After all epochs for this update) ---
        self.total_updates_performed += 1
        # Calculate the average of metrics collected across all minibatches and epochs
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items() if v}
        return avg_metrics

    def save_checkpoint(self, path: str, episode: int):
        """Saves model, optimizer state, and training progress."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save to a temporary file first to prevent corruption if interrupted
        temp_path = path + ".tmp"
        try:
            checkpoint = {
                "episode": episode,  # Last completed episode
                "total_steps_interacted": self.total_steps_interacted,
                "total_updates_performed": self.total_updates_performed,
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,  # Save config for reference and compatibility checks
                # Add other state if needed (e.g., EMA baseline state, RNG state)
            }
            torch.save(checkpoint, temp_path)
            # Atomically replace the old checkpoint with the new one
            os.replace(temp_path, path)
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint to {path}: {e}")
            # Clean up the temporary file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def load_checkpoint(self, path: str) -> int:
        """Loads model and optimizer state, returns the next episode to start from."""
        if not os.path.exists(path):
            print(
                f"[Warning] Checkpoint file not found at {path}. Starting from episode 0."
            )
            return 0
        try:
            # Load checkpoint onto the agent's device
            checkpoint = torch.load(path, map_location=self.device)

            # --- Compatibility Check (Basic Example) ---
            chk_config = checkpoint.get("config")
            if isinstance(
                chk_config, PPOConfig
            ):  # Check if config exists and is the right type
                # Compare critical parameters that affect network architecture or behavior
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
                # Add checks for encoder/decoder sizes if they might change

                if mismatched_params:
                    print(f"[Warning] Config mismatch detected! Checkpoint vs Current:")
                    for param in mismatched_params:
                        print(f"  - {param}")
                    print("Loading weights anyway, but behavior may be unpredictable.")
            else:
                print(
                    "[Warning] Checkpoint does not contain PPOConfig or it's invalid. Cannot perform compatibility checks."
                )
            # --- End Compatibility Check ---

            # Load state dictionaries
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load training progress counters
            self.total_steps_interacted = checkpoint.get("total_steps_interacted", 0)
            self.total_updates_performed = checkpoint.get("total_updates_performed", 0)
            # Determine the next episode to start (episode saved is the last completed one)
            start_episode = checkpoint.get("episode", -1) + 1

            print(
                f"Checkpoint loaded from {path}. Resuming from episode {start_episode}."
            )
            print(f"  -> Total Steps Interacted: {self.total_steps_interacted:,}")
            print(f"  -> Total Updates Performed: {self.total_updates_performed:,}")
            return start_episode
        except KeyError as e:
            print(
                f"[ERROR] Missing key in checkpoint {path}: {e}. Cannot load checkpoint. Starting from scratch."
            )
            return 0
        except Exception as e:
            print(
                f"[ERROR] Failed to load checkpoint from {path}: {e}. Starting from scratch."
            )
            return 0


# --- Visualization (Identical) ---
def print_map(console: Console, map_tensor: torch.Tensor, title: str = "Generated Map"):
    """Prints a map tensor to the console using Rich."""
    # Handle different input tensor dimensions (e.g., from observation vs initial)
    if (
        map_tensor.ndim == 4 and map_tensor.shape[0] == 1 and map_tensor.shape[1] == 1
    ):  # (1, 1, H, W)
        map_tensor = map_tensor.squeeze(0).squeeze(0)
    elif map_tensor.ndim == 3 and map_tensor.shape[0] == 1:  # (1, H, W)
        map_tensor = map_tensor.squeeze(0)
    elif (
        map_tensor.ndim == 3 and map_tensor.shape[1] == 1
    ):  # (N, 1, H, W) -> take first map
        map_tensor = map_tensor[0].squeeze(0)
    elif map_tensor.ndim == 2:  # Already (H, W)
        pass
    else:
        console.print(
            f"[red]Error: Invalid map tensor shape for printing: {map_tensor.shape}. Expected 2D or squeezable to 2D.[/red]"
        )
        return

    # Ensure tensor is on CPU for numpy conversion
    if map_tensor.device != torch.device("cpu"):
        map_tensor = map_tensor.cpu()

    # Convert to numpy array of integers
    try:
        map_np = map_tensor.numpy().astype(int)
    except Exception as e:
        console.print(f"[red]Error converting map tensor to numpy: {e}[/red]")
        return

    map_size_x, map_size_y = map_np.shape

    # Define tile colors (adjust based on actual tile values used)
    colors = {
        0: "dim grey50",  # Wall/Unused (adjust if 0 is used differently)
        1: "white",  # Empty (Common)
        6: "bright_green",  # Door (Common in critic training)
        # Enemy colors (Check critic_approximator.py and level_critic.py conventions)
        2: "bright_red",  # Enemy type 1?
        3: "red",  # Enemy type 2? (Used in critic training)
        4: "dark_red",  # Enemy type 3? (Used in critic training)
        5: "red3",  # Enemy type 4? (Used in critic training)
    }
    default_color = "magenta"  # Color for unexpected tile values
    char_width = 2  # Width for each cell in the table

    # Create Rich table
    table = Table(
        title=title,
        show_header=False,
        show_edge=False,  # Cleaner look without cell edges
        box=None,
        padding=(0, 0),  # No padding within cells
        expand=False,
    )
    # Add columns with fixed width
    for _ in range(map_size_y):
        table.add_column(
            justify="center", width=char_width, style="dim"
        )  # Dim style for grid lines effect

    # Add rows with colored cells
    for r in range(map_size_x):
        row_cells = []
        for tile in map_np[r]:
            color = colors.get(tile, default_color)
            # Format tile value, right-aligned within cell width minus 1 for space
            cell_text = (
                f"[{color}]{tile:>{char_width - 1}} [/]"  # Add space after number
            )
            row_cells.append(cell_text)
        table.add_row(*row_cells)

    console.print(table)


# --- Training Orchestrator (Modified __init__, train) ---
class PPOTrainer:
    """Orchestrates the PPO training process with episodic resets."""

    def __init__(self, config: PPOConfig, base_critic_func: Callable):
        self.config = config
        self.base_critic_func = base_critic_func  # The original symbolic critic
        self.device = get_device(config.device)
        self.console = Console()
        self.logger = logging.getLogger(__name__)  # Get logger instance

        # --- Seed everything for reproducibility ---
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
            # Optional: Enable deterministic algorithms for debugging, may impact performance
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

        # --- Setup Logging ---
        self._setup_logging()

        # --- Initialize Neural Critic (if configured) ---
        self.neural_critic_model: Optional[CriticApproximatorMLP] = None
        self.critic_func_to_use: Callable = self.base_critic_func  # Default to symbolic

        if config.use_neural_critic:
            ckpt_path = config.neural_critic_checkpoint_path
            self.logger.info(f"Attempting to load Neural Critic from: {ckpt_path}")
            if not os.path.isabs(ckpt_path):
                # Assume path is relative to the script's execution directory (e.g., base_directory)
                ckpt_path = os.path.abspath(ckpt_path)
                self.logger.info(f"Resolved relative path to: {ckpt_path}")

            try:
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(
                        f"Neural critic checkpoint not found at resolved path: {ckpt_path}"
                    )

                # Load checkpoint (MUST contain config and state_dict)
                critic_checkpoint = torch.load(
                    ckpt_path, map_location=self.device, weights_only=False
                )

                if not isinstance(critic_checkpoint, dict):
                    raise TypeError("Loaded critic checkpoint is not a dictionary.")
                if "config" not in critic_checkpoint:
                    raise KeyError("Critic checkpoint missing 'config' field.")
                if "model_state_dict" not in critic_checkpoint:
                    raise KeyError(
                        "Critic checkpoint missing 'model_state_dict' field."
                    )

                # Extract critic's configuration
                critic_config_data = critic_checkpoint["config"]
                if not isinstance(critic_config_data, CriticConfig):
                    # If config was saved as dict, try to reconstruct CriticConfig
                    if isinstance(critic_config_data, dict):
                        self.logger.warning(
                            "Critic config loaded as dict, attempting reconstruction."
                        )
                        # Need 'configure_critic_from_yaml' or similar logic to handle potential missing fields/defaults
                        # For simplicity, assume the dict contains all necessary fields for now
                        try:
                            critic_config = CriticConfig(**critic_config_data)
                        except Exception as config_e:
                            raise TypeError(
                                f"Failed to reconstruct CriticConfig from dict: {config_e}"
                            )
                    else:
                        raise TypeError(
                            f"Expected CriticConfig or dict in checkpoint, got {type(critic_config_data)}"
                        )
                else:
                    critic_config = critic_config_data

                self.logger.info(
                    f"Loaded critic configuration: MapSize={critic_config.map_size}, Hidden={critic_config.mlp_hidden_sizes}"
                )

                # --- Sanity Check: Map Size Compatibility ---
                critic_config.map_size = tuple(critic_config.map_size)
                if critic_config.map_size != self.config.map_size:
                    # This is a critical mismatch, likely requires retraining or different checkpoint
                    msg = (
                        f"CRITICAL MAP SIZE MISMATCH! "
                        f"PPO Config: {self.config.map_size}, "
                        f"Loaded Critic Config: {critic_config.map_size}. "
                        "Cannot proceed with incompatible critic."
                    )
                    self.logger.error(msg)
                    self.console.print(f"[bold red]{msg}[/bold red]")
                    raise ValueError(msg)  # Stop execution

                # Instantiate the MLP critic model using its config
                self.neural_critic_model = CriticApproximatorMLP(
                    input_size=critic_config.map_size[0] * critic_config.map_size[1],
                    hidden_sizes=critic_config.mlp_hidden_sizes,
                    output_size=1,
                    dropout_prob=critic_config.mlp_dropout_prob,  # Use dropout from its training
                ).to(self.device)

                # Load the trained weights
                self.neural_critic_model.load_state_dict(
                    critic_checkpoint["model_state_dict"]
                )
                self.neural_critic_model.eval()  # Set to evaluation mode (disables dropout)

                self.logger.info(
                    f"[bold green]Successfully loaded and initialized Neural Critic model.[/bold green]"
                )

                # --- Create the wrapper function for the neural critic ---
                def neural_critic_wrapper(
                    map_tensor: torch.Tensor, hero_tensor: torch.Tensor
                ) -> torch.Tensor:
                    """Wrapper for the neural critic MLP"""
                    # Ensure map_tensor is on the correct device and has the right shape/type
                    map_tensor = map_tensor.to(
                        self.device
                    )  # Shape (N, H, W) or (N, 1, H, W)
                    if map_tensor.dim() == 3:
                        map_tensor = map_tensor.unsqueeze(
                            1
                        )  # Add channel dim -> (N, 1, H, W)
                    map_tensor = map_tensor.float()  # Ensure float type

                    # Call the neural critic model (no gradients needed here)
                    with torch.no_grad():
                        # Pass only the map tensor to the MLP critic
                        scores = self.neural_critic_model(
                            map_tensor
                        )  # Output shape (N, 1)
                    return scores.squeeze(-1)  # Return shape (N,)

                # Set the function to use for reward calculation
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
                    f"[bold red]Error loading Neural Critic:[/bold red] {e}. Check path and checkpoint format."
                )
                self.console.print(
                    "Falling back to the original symbolic critic function."
                )
                self.critic_func_to_use = self.base_critic_func  # Fallback
                self.neural_critic_model = None  # Ensure it's None if loading failed
        else:
            self.logger.info(
                "Using the original symbolic critic function (neural critic disabled in config)."
            )
            self.console.print(
                "Using original symbolic critic ([dim]neural critic disabled[/dim])."
            )
            self.critic_func_to_use = self.base_critic_func

        # --- Initialize other components ---
        self.env = BatchedEnvSimulator(
            config, self.critic_func_to_use
        )  # Pass the chosen critic
        self.agent = PPOAgent(config)
        self.memory = PPOMemory(
            config.num_envs,
            config.n_steps_per_rollout,
            config.map_size,
            config.hero_tensor_size,
            self.device,
        )

        # Initialize training state variables
        self.start_episode = 0
        self.reward_baseline = 0.0
        self.is_first_episode_for_baseline = True

    def _setup_logging(self):
        """Configures the logging module to log to file."""
        log_dir = self.config.checkpoint_dir  # Log file in the run's checkpoint dir
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, self.config.log_file_name)

        # Remove existing handlers to prevent duplicate logs if re-initialized
        # (Good practice, especially in notebooks or repeated runs)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        # Set logging level
        self.logger.setLevel(logging.INFO)
        # Define formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Create File Handler (Append mode 'a')
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Optional: Add a StreamHandler to log to console as well (can be verbose)
        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(formatter)
        # self.logger.addHandler(console_handler)

        self.logger.info(f"--- Logging started for run: {self.config.run_name} ---")
        # Log key configuration parameters for traceability
        self.logger.info(
            f"PPO Config: Mode={self.config.mode}, MapSize={self.config.map_size}, Device={self.config.device}"
        )
        self.logger.info(
            f"Training Params: Episodes={self.config.num_episodes}, EpLength={self.config.episode_length}, NumEnvs={self.config.num_envs}"
        )
        self.logger.info(
            f"PPO Params: RolloutSteps={self.config.n_steps_per_rollout}, Epochs={self.config.num_epochs_per_update}, Minibatch={self.config.minibatch_size}"
        )
        self.logger.info(
            f"Hyperparams: LR={self.config.learning_rate}, Gamma={self.config.gamma}, Clip={self.config.clip_epsilon}, VF_Coef={self.config.vf_coef}, Ent_Coef={self.config.entropy_coef}"
        )
        self.logger.info(f"Using Neural Critic: {self.config.use_neural_critic}")
        if self.config.use_neural_critic:
            self.logger.info(
                f"Neural Critic Path: {self.config.neural_critic_checkpoint_path}"
            )

    def train(self):
        """Runs the main PPO training loop over episodes."""
        cfg = self.config

        # --- Setup & Initial Logging ---
        critic_desc = (
            "[bold green]Neural Approximator[/]"
            if self.config.use_neural_critic and self.neural_critic_model
            else "[bold blue]Symbolic Critic[/]"
        )
        if self.config.use_neural_critic and self.neural_critic_model is None:
            critic_desc += " ([red]Load Failed![/red])"

        self.console.print(
            Panel.fit(
                f"Starting PPO Episodic Training: mode='{cfg.mode}', run='{cfg.run_name}'\nReward Critic: {critic_desc}",
                title="Setup",
                border_style="blue",
            )
        )
        self.console.print(
            f"Device: [cyan]{self.device}[/cyan], Episodes: {cfg.num_episodes}, Steps/Episode: {cfg.episode_length}"
        )
        self.console.print(
            f"Envs: {cfg.num_envs}, Steps/Rollout (Update Freq): [bold yellow]{cfg.n_steps_per_rollout}[/], PPO Epochs: {cfg.num_epochs_per_update}, Minibatch: {cfg.minibatch_size}"
        )
        self.console.print(f"Total Transitions per Update: {cfg.batch_size}")
        self.console.print(
            f"Checkpoints & Logs Dir: [green]{cfg.checkpoint_dir}[/green]"
        )
        self.logger.info("Training setup complete.")

        # --- Load Agent Checkpoint ---
        latest_checkpoint_path = os.path.join(
            cfg.checkpoint_dir, "latest_checkpoint.pth"
        )
        if os.path.exists(latest_checkpoint_path):
            self.logger.info(
                f"Loading PPO agent checkpoint from {latest_checkpoint_path}"
            )
            self.start_episode = self.agent.load_checkpoint(latest_checkpoint_path)
            # Reset baseline calculation flag if resuming from start
            self.is_first_episode_for_baseline = self.start_episode == 0
            self.logger.info(f"Resuming training from episode {self.start_episode}")
            # Check if training is already complete
            if self.start_episode >= cfg.num_episodes:
                msg = f"Checkpoint indicates training already completed ({self.start_episode}/{cfg.num_episodes} episodes)."
                self.console.print(f"[yellow]{msg} Exiting.[/yellow]")
                self.logger.warning(msg)
                return
        else:
            self.logger.info(
                "No PPO agent checkpoint found. Starting training from scratch."
            )
            self.start_episode = 0

        # --- Setup Rich Progress Bar ---
        episode_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            # This column displays metrics attached to the main episode task
            TextColumn(
                "[bold]Metrics:[/]{task.fields[metrics]}", justify="left", style="white"
            ),
            console=self.console,
            transient=False,  # Keep the bar visible after completion
            # refresh_per_second=10 # Adjust refresh rate if needed
        )

        # --- Episodic Training Loop ---
        map_obs, hero_obs = None, None  # Initialized in env.reset()

        with episode_progress:
            # Add the main task for tracking episodes
            episode_task = episode_progress.add_task(
                "[cyan]Training Episodes",
                total=cfg.num_episodes,
                completed=self.start_episode,
                metrics=" Starting...",  # Initial metrics string
            )

            for episode in range(self.start_episode, cfg.num_episodes):
                # --- Start of Episode ---
                map_obs, hero_obs, initial_maps_cpu = self.env.reset()
                episode_rewards = []
                steps_this_episode = 0
                last_update_metrics = {}  # Store metrics from the last update in the episode

                # --- Print Initial Maps Periodically ---
                if (
                    episode == self.start_episode
                    or (episode + 1) % cfg.print_maps_freq == 0
                ):
                    num_maps_to_print = min(3, cfg.num_envs)
                    episode_progress.console.print(
                        Panel(
                            f"--- Episode {episode + 1}: Initial Maps (First {num_maps_to_print}) ---",
                            expand=False,
                            border_style="dim",
                        )
                    )
                    for i in range(num_maps_to_print):
                        print_map(
                            episode_progress.console,
                            initial_maps_cpu[i],
                            title=f"Ep {episode + 1} Initial (Env {i})",
                        )

                # --- Inner Loop: Steps within an Episode ---
                step_task_desc = f"Ep {episode + 1}/{cfg.num_episodes} Steps"
                # <<< FIX: Initialize the 'metrics' field for the step_task >>>
                step_task = episode_progress.add_task(
                    step_task_desc,
                    total=cfg.episode_length,
                    visible=True,  # Make this task bar visible
                    metrics="",  # Initialize metrics field to prevent KeyError
                )

                rollout_step = 0  # Track steps within the current rollout buffer
                while steps_this_episode < cfg.episode_length:
                    # --- Collect Rollout ---
                    if rollout_step == 0:
                        self.memory.clear()  # Clear memory at the start of each new rollout

                    # Determine how many steps to run in this interaction cycle
                    steps_to_run = min(
                        cfg.n_steps_per_rollout
                        - rollout_step,  # Remaining steps in rollout buffer
                        cfg.episode_length - steps_this_episode,
                    )  # Remaining steps in episode

                    if steps_to_run <= 0:
                        break  # Safety check

                    # Interact with environment for 'steps_to_run' steps
                    for _ in range(steps_to_run):
                        if steps_this_episode >= cfg.episode_length:
                            break  # Double check episode length

                        # Select action, get log_prob and value estimate
                        action, log_prob, value = self.agent.select_action(
                            map_obs, hero_obs, cfg.temperature
                        )

                        # Step the environment
                        next_obs_tuple, reward, done, info = self.env.step(action)
                        next_map_obs, next_hero_obs = next_obs_tuple

                        # Store transition in memory
                        self.memory.add(
                            map_obs, hero_obs, action, log_prob, reward, done, value
                        )

                        # Update current observation
                        map_obs, hero_obs = next_map_obs, next_hero_obs

                        # Record rewards (use mean over envs for episode avg)
                        episode_rewards.append(reward.mean().item())

                        # Increment counters
                        self.agent.total_steps_interacted += cfg.num_envs
                        steps_this_episode += 1
                        rollout_step += 1

                        # Update step progress bar
                        episode_progress.update(
                            step_task,
                            advance=1,
                            description=f"{step_task_desc} ({rollout_step}/{cfg.n_steps_per_rollout})",
                        )

                    # --- Perform PPO Update (if rollout buffer is full) ---
                    # Condition should be met exactly when rollout_step reaches n_steps_per_rollout
                    if rollout_step == cfg.n_steps_per_rollout:
                        # Compute GAE and returns *before* the update
                        with torch.no_grad():
                            # Get value estimate for the *last* observation reached
                            last_value = self.agent.critic(map_obs, hero_obs).squeeze(
                                -1
                            )
                        self.memory.compute_gae_returns(
                            last_value, cfg.gamma, cfg.gae_lambda
                        )

                        # Perform PPO update epochs using the data in memory
                        update_metrics = self.agent.update(self.memory)
                        last_update_metrics = (
                            update_metrics  # Store for episode summary
                        )

                        # Log PPO update metrics to file
                        if update_metrics:
                            p_loss = update_metrics.get("policy_loss", float("nan"))
                            v_loss = update_metrics.get("value_loss", float("nan"))
                            ent = update_metrics.get("entropy", float("nan"))
                            kl = update_metrics.get("approx_kl", float("nan"))
                            clip_frac = update_metrics.get(
                                "clip_fraction", float("nan")
                            )
                            self.logger.info(
                                f"Ep: {episode + 1}, Update: {self.agent.total_updates_performed}, Step: {self.agent.total_steps_interacted}, "
                                f"P_Loss: {p_loss:.4f}, V_Loss: {v_loss:.4f}, Entropy: {ent:.4f}, KL: {kl:.4f}, ClipFrac: {clip_frac:.3f}"
                            )
                        else:
                            self.logger.warning(
                                f"Ep: {episode + 1}, Update: {self.agent.total_updates_performed} returned no metrics (likely empty buffer)."
                            )

                        # Reset rollout step counter for the next collection phase
                        rollout_step = 0
                        # Memory is cleared at the beginning of the next rollout phase

                # --- End of Episode ---
                avg_ep_reward = np.mean(episode_rewards) if episode_rewards else 0.0

                # Update Reward Baseline (EMA) for logging/monitoring
                if self.is_first_episode_for_baseline:
                    self.reward_baseline = avg_ep_reward
                    self.is_first_episode_for_baseline = False
                else:
                    alpha = cfg.reward_baseline_alpha
                    self.reward_baseline = (
                        alpha * avg_ep_reward + (1 - alpha) * self.reward_baseline
                    )

                # Log Episode Summary to file
                self.logger.info(
                    f"Ep: {episode + 1}/{cfg.num_episodes} finished. Steps: {steps_this_episode}. "
                    f"AvgReward: {avg_ep_reward:.4f}, RewardBaseline(EMA): {self.reward_baseline:.4f}, "
                    f"TotalEnvSteps: {self.agent.total_steps_interacted:,}"
                )

                # Update Overall Episode Progress Bar Display
                p_loss_str = (
                    f"{last_update_metrics.get('policy_loss', float('nan')):>7.3f}"
                )
                v_loss_str = (
                    f"{last_update_metrics.get('value_loss', float('nan')):>7.3f}"
                )
                ent_str = f"{last_update_metrics.get('entropy', float('nan')):>6.3f}"
                metrics_str = (
                    f"AvgRew:[yellow]{avg_ep_reward:>7.3f}[/]| "
                    f"Baseline:[cyan]{self.reward_baseline:>7.3f}[/]| "
                    f"P:[red]{p_loss_str}[/]| "  # Shorter labels
                    f"V:[magenta]{v_loss_str}[/]| "
                    f"E:[blue]{ent_str}[/]"
                )
                episode_progress.update(episode_task, advance=1, metrics=metrics_str)
                episode_progress.remove_task(
                    step_task
                )  # Remove the completed step task bar

                # --- Print Final Maps Periodically ---
                if (
                    episode + 1
                ) % cfg.print_maps_freq == 0 or episode == cfg.num_episodes - 1:
                    num_maps_to_print = min(3, cfg.num_envs)
                    final_maps_cpu = (
                        self.env.maps.detach().cpu()
                    )  # Get current maps from env
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

                # --- Save Checkpoint Periodically and Latest ---
                if (episode + 1) % cfg.save_checkpoint_freq == 0:
                    # Save periodic checkpoint (e.g., checkpoint_ep_100.pth)
                    chk_path = os.path.join(
                        cfg.checkpoint_dir, f"checkpoint_ep_{episode + 1}.pth"
                    )
                    self.agent.save_checkpoint(chk_path, episode)  # Save agent state
                    self.logger.info(
                        f"Periodic checkpoint saved to {chk_path} after episode {episode + 1}"
                    )

                    # Save latest checkpoint (overwrite latest_checkpoint.pth)
                    latest_path = os.path.join(
                        cfg.checkpoint_dir, "latest_checkpoint.pth"
                    )
                    self.agent.save_checkpoint(latest_path, episode)
                    self.logger.info(f"Latest checkpoint updated to {latest_path}")

        # --- End of Training ---
        msg = (
            f"Training finished after {cfg.num_episodes} episodes "
            f"({self.agent.total_steps_interacted:,} total environment steps, "
            f"{self.agent.total_updates_performed:,} PPO updates)."
        )
        self.console.print(Panel(msg, title="Complete", border_style="green"))
        self.logger.info(msg)

        # Save final agent state checkpoint
        final_chk_path = os.path.join(cfg.checkpoint_dir, "final_checkpoint.pth")
        self.agent.save_checkpoint(
            final_chk_path, cfg.num_episodes - 1
        )  # Save state after last episode
        self.console.print(
            f"Final checkpoint saved to: [green]{final_chk_path}[/green]"
        )
        self.logger.info(f"Final checkpoint saved to {final_chk_path}")

        # Also update latest_checkpoint to be the same as final, ensuring it points to the fully trained model
        latest_path = os.path.join(cfg.checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(final_chk_path):  # Make sure final save was successful
            try:
                # Create a symlink or copy the file
                if os.path.exists(latest_path):
                    os.remove(latest_path)  # Remove old latest if exists
                # os.symlink(os.path.basename(final_chk_path), latest_path) # Use symlink if preferred
                import shutil

                shutil.copyfile(final_chk_path, latest_path)  # Copy for robustness
                self.logger.info(
                    f"Latest checkpoint updated to final state: {latest_path}"
                )
            except Exception as link_e:
                self.logger.error(
                    f"Failed to update latest checkpoint link/copy: {link_e}"
                )


# --- Main Execution ---
if __name__ == "__main__":
    # Set default config path relative to the script location
    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__), "ppo_episodic_config.yaml"
    )

    # Basic argument parsing for config file override
    parser = argparse.ArgumentParser(
        description="Train PPO Agent with Optional Neural Critic"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the PPO configuration YAML file (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()
    config_path = args.config

    # Load PPO configuration
    if os.path.exists(config_path):
        print(f"Loading PPO configuration from {config_path}")
        config = configure_from_yaml(config_path)
    else:
        print(
            f"Configuration file '{config_path}' not found. Using default settings defined in PPOConfig."
        )
        config = PPOConfig()

    # --- Initialize Base Critic (Symbolic) ---
    # The trainer will decide whether to use this or the neural one based on config
    symbolic_critic = actual_critic

    # --- Create and Run Trainer ---
    trainer = PPOTrainer(config, symbolic_critic)
    try:
        trainer.train()
    except KeyboardInterrupt:
        # Handle graceful interruption by user (Ctrl+C)
        msg = "\nTraining interrupted by user. Saving final checkpoint..."
        trainer.console.print(f"[yellow]{msg}[/yellow]")
        trainer.logger.warning(msg)
        latest_episode = -1
        # Try to determine the last completed episode from the agent's state
        if hasattr(trainer, "agent") and hasattr(
            trainer.agent, "load_checkpoint"
        ):  # Check if agent exists
            # Attempt to read episode from the latest checkpoint if possible
            latest_checkpoint_path = os.path.join(
                trainer.config.checkpoint_dir, "latest_checkpoint.pth"
            )
            if os.path.exists(latest_checkpoint_path):
                try:
                    checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
                    latest_episode = checkpoint.get("episode", -1)
                except Exception as e:
                    trainer.logger.error(
                        f"Could not read latest checkpoint episode on interrupt: {e}"
                    )

        # Save the current state as interrupted
        interrupted_chk_path = os.path.join(
            trainer.config.checkpoint_dir, "final_checkpoint_interrupted.pth"
        )
        if hasattr(trainer, "agent"):
            trainer.agent.save_checkpoint(interrupted_chk_path, latest_episode)
            final_msg = f"Interrupted state checkpoint saved to {interrupted_chk_path} (based on episode {latest_episode + 1} starting or interrupted)"
            print(final_msg)
            trainer.logger.info(final_msg)
        else:
            print("Agent not initialized, cannot save interrupt checkpoint.")

    except Exception as e:
        # Handle unexpected errors during training
        trainer.console.print(
            "\n[bold red]An critical error occurred during training:[/bold red]"
        )
        trainer.console.print_exception(show_locals=False)  # Show traceback in console
        trainer.logger.error(
            "A critical error occurred during training.", exc_info=True
        )  # Log full traceback to file

        print("[bold red]Attempting to save error state checkpoint...[/bold red]")
        latest_episode = -1
        # Try to get last episode from checkpoint
        latest_checkpoint_path = os.path.join(
            trainer.config.checkpoint_dir, "latest_checkpoint.pth"
        )
        if os.path.exists(latest_checkpoint_path):
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
                latest_episode = checkpoint.get("episode", -1)
            except Exception as load_e:
                trainer.logger.error(
                    f"Could not read latest checkpoint episode during error handling: {load_e}"
                )

        # Save error state checkpoint
        error_chk_path = os.path.join(
            trainer.config.checkpoint_dir, "final_checkpoint_error.pth"
        )
        if hasattr(trainer, "agent"):
            trainer.agent.save_checkpoint(error_chk_path, latest_episode)
            error_msg = f"Error state checkpoint saved to {error_chk_path} (based on episode {latest_episode + 1} starting or interrupted)"
            print(error_msg)
            trainer.logger.info(error_msg)
        else:
            print("Agent not initialized, cannot save error checkpoint.")

    # --- Optional Post-Training Test (Network Forward Pass) ---
    console = Console()
    print("\n--- Optional: Testing Network Forward Pass (Final Config/State) ---")
    try:
        # Check if trainer and agent were successfully initialized
        if "trainer" in locals() and hasattr(trainer, "agent"):
            test_agent = trainer.agent
            current_config = trainer.config  # Use the config from the trainer

            # Create dummy input data matching the config
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

            test_agent.actor.eval()  # Set to eval mode
            test_agent.critic.eval()

            with torch.no_grad():
                # Test actor forward pass
                action_logits = test_agent.actor(test_map, test_hero)
                action_probs = F.softmax(action_logits, dim=-1)
                # Test critic forward pass
                state_values = test_agent.critic(test_map, test_hero)

            console.print(
                f"Mode: {current_config.mode}, Action Dim: {current_config.action_dim}"
            )
            console.print(
                f"Input Map Shape: {test_map.shape}, Input Hero Shape: {test_hero.shape}"
            )
            console.print(f"Output Actor Logits Shape: {action_logits.shape}")
            console.print(f"Output Actor Probs Shape: {action_probs.shape}")
            console.print(
                f"Output Critic Values Shape: {state_values.shape}"
            )  # Expected: (batch_size, 1)

            # Basic shape assertions
            assert action_logits.shape == (batch_size, current_config.action_dim), (
                f"Actor output shape mismatch!"
            )
            assert state_values.shape == (batch_size, 1), (
                f"Critic output shape mismatch!"
            )

            console.print("[green]Network forward pass test successful.[/green]")
        else:
            console.print(
                "[yellow]Trainer object or agent not fully initialized, skipping network test.[/yellow]"
            )

    except AttributeError as ae:
        console.print(
            f"[yellow]Attribute error during test (likely incomplete initialization): {ae}. Skipping network test.[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Network forward pass test failed: {e}[/red]")
        console.print_exception(show_locals=False)
