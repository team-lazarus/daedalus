# --- Imports ---
# TODO: Parth, add NN encoder
# TODO: Parth, ensure metrics for each episode are saved
# TODO: Parth, add stable baseline
# TODO: Parth, add arg parse (without stable baseline, with stable baseline, dqn [if possible])
# TODO: Parth, validation

# TODO: Parth, there are three representation (narrow, wide, turtle)
# TODO: Parth, add batches

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import random
import yaml
import os
import time
from typing import List, Tuple, Dict, Any, Callable, Optional, Union, Generator
from dataclasses import dataclass, field, fields  # Import fields for config loading

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
from tqdm import tqdm  # Import tqdm

# Assume the critic function and Entry enum are available
from daedalus.critics import level_critic

from enum import Enum


# --- Mock Entry Enum (Replace with actual import if available) ---
class Entry(Enum):
    TOP = 0
    LEFT = 1
    BOTTOM = 2
    RIGHT = 3


# --- Model Definition (Identical to previous version) ---
class PolicyNetworkEncoder(nn.Module):
    """Encodes the map and hero state into a latent representation."""

    def __init__(
        self,
        channels: List[int] = [1, 4, 16, 64, 256],
        input_size: Tuple[int, int] = (12, 12),
        output_size: int = 1024,
        hero_tensor_size: int = 5,
    ):
        super().__init__()
        self.input_size = self.input_size_x, self.input_size_y = input_size
        downsampled_size_x = self.input_size_x // 4
        downsampled_size_y = self.input_size_y // 4
        self.cnn_output_dim = downsampled_size_x * downsampled_size_y * channels[-1]
        self.output_size = output_size
        self.hero_tensor_size = hero_tensor_size

        self.conv1 = nn.Conv2d(channels[0], channels[1], 3, padding="same")
        self.conv2 = nn.Conv2d(channels[1], channels[2], 3, padding="same")
        self.conv3 = nn.Conv2d(channels[2], channels[3], 3, padding="same")
        self.conv4 = nn.Conv2d(channels[3], channels[4], 3, padding="same")
        self.down1 = nn.Conv2d(
            channels[2], channels[2], kernel_size=3, stride=2, padding=1
        )
        self.down2 = nn.Conv2d(
            channels[4], channels[4], kernel_size=3, stride=2, padding=1
        )
        self.linear = nn.Linear(
            self.cnn_output_dim + self.hero_tensor_size, self.output_size, bias=True
        )

    def forward(self, x: torch.Tensor, hero_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.down1(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.down2(x))
        x = torch.flatten(x, start_dim=1)
        hero_tensor = hero_tensor.float()  # Ensure float type
        combined = torch.cat([x, hero_tensor], dim=1)
        x = F.relu(self.linear(combined))
        return x


class PolicyNetworkDecoder(nn.Module):
    """Decodes latent representation into action probabilities or state values."""

    def __init__(
        self,
        input_size: int = 1024,
        hidden_sizes: List[int] = [512, 256],
        output_size: int = 7,  # Default, adjusted by PPOConfig
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_out = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the decoder."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)  # Output raw logits
        return x


class PPOPolicyNetwork(nn.Module):
    """Combined PPO Actor Network using Encoder-Decoder structure."""

    def __init__(
        self,
        channels: List[int] = [1, 4, 16, 64, 256],
        input_size: Tuple[int, int] = (12, 12),
        encoder_output_size: int = 1024,
        decoder_hidden_sizes: List[int] = [512, 256],
        action_dim: int = 7,  # Default, adjusted by PPOConfig
        hero_tensor_size: int = 5,
    ):
        super().__init__()
        self.encoder = PolicyNetworkEncoder(
            channels=channels,
            input_size=input_size,
            output_size=encoder_output_size,
            hero_tensor_size=hero_tensor_size,
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
        latent = self.encoder(map_tensor, hero_tensor)
        action_logits = self.actor_decoder(latent)
        return action_logits


class ValueNetwork(nn.Module):
    """PPO Critic Network using Encoder-Decoder structure."""

    def __init__(
        self,
        channels: List[int] = [1, 4, 16, 64, 256],
        input_size: Tuple[int, int] = (12, 12),
        encoder_output_size: int = 1024,
        decoder_hidden_sizes: List[int] = [512, 256],
        hero_tensor_size: int = 5,
    ):
        super().__init__()
        self.encoder = PolicyNetworkEncoder(
            channels=channels,
            input_size=input_size,
            output_size=encoder_output_size,
            hero_tensor_size=hero_tensor_size,
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
    num_episodes: int = 1000  # Total episodes to run (Adjustable)
    episode_length: int = 256  # Steps per episode (Adjustable)

    # PPO Core Parameters
    num_envs: int = 128  # Number of parallel environments/maps (Adjustable)
    n_steps_per_rollout: int = (
        64  # <<<<< UPDATE FREQUENCY: Update policy every N steps per env
    )
    num_epochs_per_update: int = 4  # PPO epochs over collected data
    minibatch_size: int = 64  # Should be <= num_envs * n_steps_per_rollout

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

    # Logging & Saving (relative to episodes now)
    save_checkpoint_freq: int = 100  # Save checkpoint every N episodes
    print_maps_freq: int = 50  # Print maps every N episodes
    checkpoint_dir: str = "ppo_checkpoints_episodic_metrics"
    run_name: Optional[str] = None

    # Environment Initialization
    initial_map_walk_steps: int = 36
    initial_map_empty_prob: float = 0.75

    # Calculated properties
    batch_size: int = field(init=False)  # Rollout batch size
    action_dim: int = field(init=False)

    def __post_init__(self):
        """Calculate derived properties after initialization."""
        # Batch size for PPO update is still based on rollouts
        self.batch_size = self.num_envs * self.n_steps_per_rollout
        if self.minibatch_size > self.batch_size:
            print(
                f"Warning: Minibatch size ({self.minibatch_size}) > Rollout batch size ({self.batch_size}). Adjusting minibatch size."
            )
            self.minibatch_size = self.batch_size

        n_tiles = self.map_size[0] * self.map_size[1]
        mode_actions = {
            "narrow": 7,
            "turtle": 10,  # 6 mod + 4 move
            # For wide, action = tile_type * num_tiles + flat_index
            # Assuming 7 tile types (0-6) like narrow
            "wide": 7 * n_tiles,
        }
        if self.mode not in mode_actions:
            raise ValueError(f"Unknown mode: {self.mode}")
        self.action_dim = mode_actions[self.mode]

        if self.run_name is None:
            self.run_name = f"ppo_{self.mode}_ep_{time.strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.run_name)


def configure_from_yaml(yaml_path: str) -> PPOConfig:
    """Loads configuration from a YAML file, filtering unknown keys."""
    try:
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        # Filter to only include keys defined in PPOConfig
        valid_keys = {f.name for f in fields(PPOConfig) if f.init}  # Use fields()
        filtered_config = {k: v for k, v in yaml_config.items() if k in valid_keys}
        return PPOConfig(**filtered_config)
    except FileNotFoundError:
        print(
            f"Warning: YAML config file not found at {yaml_path}. Using default config."
        )
        return PPOConfig()
    except Exception as e:
        print(f"Error loading YAML config: {e}. Using default config.")
        return PPOConfig()


# --- Environment Utilities (Identical) ---
def get_device(device_str: str) -> torch.device:
    """Gets the torch device."""
    return torch.device(device_str)


def initialize_map_hero(
    config: PPOConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Creates a random initial map via random walk and a random hero tensor."""
    map_tensor = torch.zeros(config.map_size, dtype=torch.int64)
    size_x, size_y = config.map_size
    start_pos = (random.randint(0, size_x - 1), random.randint(0, size_y - 1))
    current_pos = start_pos

    for _ in range(config.initial_map_walk_steps):
        tile_value = (
            1
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
        map_tensor[start_pos] = 1  # Ensure start is walkable
    hero_tensor = torch.tensor(
        [
            random.randint(1, 10),  # health
            random.choice([0, 1]),  # item1
            random.choice([0, 1]),  # item2
            random.choice([e.value for e in Entry]),  # direction
            random.randint(0, 10),  # rooms_left
        ],
        dtype=torch.int64,
    )
    return map_tensor, hero_tensor, start_pos


# --- Environment Simulation (Identical except reset logic clarified) ---
class BatchedEnvSimulator:
    """Simulates multiple environments in parallel. Resets map/hero/pos together."""

    def __init__(self, config: PPOConfig, critic_func: Callable):
        self.config = config
        self.num_envs = config.num_envs
        self.map_size_x, self.map_size_y = config.map_size
        self.critic_func = critic_func
        self.device = get_device(config.device)

        # Initialize tensors on the correct device
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
        self.agent_positions = [
            (0, 0)
        ] * self.num_envs  # Store agent positions (CPU is fine)

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resets all environments (maps, heroes, positions) for a new episode.
        Returns initial observations and the initial maps (CPU tensor)."""
        initial_maps_list_cpu = []
        for i in range(self.num_envs):
            map_i, hero_i, pos_i = initialize_map_hero(self.config)
            self.maps[i] = map_i.to(self.device)
            self.heroes[i] = hero_i.to(self.device)
            self.agent_positions[i] = pos_i
            initial_maps_list_cpu.append(map_i.clone())  # Keep a CPU copy for printing

        initial_maps_cpu = torch.stack(initial_maps_list_cpu)
        map_obs, hero_obs = self._get_observation()
        return map_obs, hero_obs, initial_maps_cpu

    def _get_observation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the current observation (maps, heroes) ready for the network."""
        map_obs = self.maps.unsqueeze(1).float()  # Add channel dim, convert to float
        hero_obs = self.heroes.float()  # Convert to float
        return map_obs, hero_obs

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """Applies actions to environments, calculates rewards, and returns results."""
        actions_np = actions.cpu().numpy()  # Actions needed for indexing/logic on CPU

        for i in range(self.num_envs):
            action = actions_np[i]
            pos_x, pos_y = self.agent_positions[i]
            current_map = self.maps[i]  # Get reference to map on device
            mode = self.config.mode

            if mode == "narrow":
                if 0 <= action <= 6:
                    current_map[pos_x, pos_y] = action  # 7 is no-op
                dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                next_x = max(0, min(self.map_size_x - 1, pos_x + dx))
                next_y = max(0, min(self.map_size_y - 1, pos_y + dy))
                self.agent_positions[i] = (next_x, next_y)
            elif mode == "turtle":
                if 0 <= action <= 5:
                    current_map[pos_x, pos_y] = action
                elif action == 6:
                    pos_x = (pos_x - 1 + self.map_size_x) % self.map_size_x  # Up
                elif action == 7:
                    pos_y = (pos_y - 1 + self.map_size_y) % self.map_size_y  # Left
                elif action == 8:
                    pos_x = (pos_x + 1) % self.map_size_x  # Down
                elif action == 9:
                    pos_y = (pos_y + 1) % self.map_size_y  # Right
                self.agent_positions[i] = (pos_x, pos_y)
            elif mode == "wide":
                num_tile_types = 7  # Assuming 0-6
                n_tiles_total = self.map_size_x * self.map_size_y
                if self.config.action_dim != num_tile_types * n_tiles_total:
                    print(
                        f"Warning: Action dim mismatch for wide mode. Check config. Expected {num_tile_types * n_tiles_total}, got {self.config.action_dim}"
                    )
                tile_type = action // n_tiles_total
                flat_index = action % n_tiles_total
                target_x = flat_index // self.map_size_y
                target_y = flat_index % self.map_size_y
                if 0 <= tile_type < num_tile_types:
                    current_map[target_x, target_y] = tile_type

        # Reward Calculation
        rewards = self.critic_func(self.maps.clone(), self.heroes.clone())

        # Done Signal: Not naturally generated, PPO handles termination by rollout length
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        next_obs = self._get_observation()
        infos = {}
        return next_obs, rewards, dones, infos


# --- PPO Memory (Identical) ---
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
        """Adds a transition step to the memory."""
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
        """Calculates advantages and returns using GAE."""
        last_gae_lam = 0
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                # In episodic setting, 'done' doesn't mean terminal state for value bootstrapping
                # We always bootstrap from the last collected value if the episode didn't truly end there.
                # Since our env step doesn't produce true terminal 'dones', next_non_terminal is always 1 here.
                next_non_terminal = (
                    1.0  # Assume non-terminal unless externally specified
                )
                next_values = last_value
            else:
                # Same logic applies for intermediate steps
                next_non_terminal = 1.0  # Assume non-terminal
                next_values = self.values[t + 1]
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        self.returns = self.advantages + self.values
        self.ptr = 0  # Reset pointer, ready for next rollout collection or reuse

    def get_minibatches(
        self, batch_size: int, minibatch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Generates minibatches from the stored transitions."""
        if self.advantages is None or self.returns is None:
            raise ValueError("Advantages/returns not computed.")
        # Use self.ptr to get actual filled size
        num_transitions = self.ptr * self.num_envs
        if num_transitions == 0:
            return  # No data to process
        if minibatch_size > num_transitions:
            # This can happen if the last rollout in an episode is smaller than minibatch_size
            print(
                f"Warning: Minibatch size ({minibatch_size}) > available transitions ({num_transitions}). Using {num_transitions} as minibatch size."
            )
            minibatch_size = num_transitions

        indices = torch.randperm(num_transitions).to(self.device)
        # Flatten and select only filled data using self.ptr
        valid_maps = self.maps[: self.ptr].reshape(
            num_transitions, *self.maps.shape[2:]
        )
        valid_heroes = self.heroes[: self.ptr].reshape(num_transitions, -1)
        valid_actions = self.actions[: self.ptr].reshape(-1)
        valid_log_probs = self.log_probs[: self.ptr].reshape(-1)
        valid_advantages = self.advantages[: self.ptr].reshape(-1)
        valid_returns = self.returns[: self.ptr].reshape(-1)
        valid_values = self.values[: self.ptr].reshape(-1)  # Get corresponding values

        for start_idx in range(0, num_transitions, minibatch_size):
            end_idx = start_idx + minibatch_size
            mb_indices = indices[start_idx:end_idx]
            # Check if mb_indices is empty which can happen if num_transitions % minibatch_size != 0
            # and we are at the very last partial batch, but this should be handled by the range end condition.
            # Double check slice logic: end_idx might exceed num_transitions if num_transitions % minibatch != 0
            # but slicing handles this gracefully.
            if len(mb_indices) == 0:
                continue

            yield {
                "maps": valid_maps[mb_indices],
                "heroes": valid_heroes[mb_indices],
                "actions": valid_actions[mb_indices],
                "old_log_probs": valid_log_probs[mb_indices],
                "advantages": valid_advantages[mb_indices],
                "returns": valid_returns[mb_indices],
                "old_values": valid_values[
                    mb_indices
                ],  # Add old values for clipped value loss (optional but common)
            }

    def clear(self):
        """Resets the memory buffer pointer."""
        self.ptr = 0


# --- PPO Agent (Identical except checkpointing logic might adapt) ---
class PPOAgent:
    """The PPO Agent class containing policy/value networks and update logic."""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = get_device(config.device)
        self.actor = PPOPolicyNetwork(
            action_dim=config.action_dim,
            hero_tensor_size=config.hero_tensor_size,
            input_size=config.map_size,  # Pass map size if needed by underlying layers
        ).to(self.device)
        self.critic = ValueNetwork(
            hero_tensor_size=config.hero_tensor_size,
            input_size=config.map_size,  # Pass map size if needed by underlying layers
        ).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
            eps=1e-5,
        )
        self.total_steps_interacted = 0  # Total env steps across all episodes
        self.total_updates_performed = 0  # Total PPO updates across all episodes

    def select_action(
        self, map_obs: torch.Tensor, hero_obs: torch.Tensor, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects actions based on policy, returns action, log_prob, and value."""
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            action_logits = self.actor(map_obs, hero_obs)
            value = self.critic(map_obs, hero_obs).squeeze(-1)
            if temperature > 0:
                scaled_logits = action_logits / max(temperature, 1e-8)
            else:
                scaled_logits = action_logits  # Deterministic handled below
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
        """Evaluates actions during the PPO update step."""
        action_logits = self.actor(map_obs, hero_obs)
        value = self.critic(map_obs, hero_obs).squeeze(-1)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs=probs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy

    def update(self, memory: PPOMemory) -> Dict[str, float]:
        """Performs the PPO update step using data collected in memory."""
        if memory.ptr == 0:
            return {}  # No data to update from

        # 1. Calculate Advantages and Returns using GAE
        with torch.no_grad():
            # Get the last observation from the memory buffer before it's cleared
            # Need to handle the case where memory isn't full (ptr < n_steps)
            last_map_obs = memory.maps[memory.ptr - 1]
            last_hero_obs = memory.heroes[memory.ptr - 1]
            last_value = self.critic(last_map_obs, last_hero_obs).squeeze(-1)
        memory.compute_gae_returns(
            last_value, self.config.gamma, self.config.gae_lambda
        )

        all_metrics = {"policy_loss": [], "value_loss": [], "entropy": []}

        # 2. Optimize policy and value networks
        for _ in range(self.config.num_epochs_per_update):
            # Pass the actual size of the collected data (memory.ptr * num_envs)
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

                # Normalize advantages at the minibatch level
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                new_log_probs, new_values, entropy = self.evaluate_actions(
                    mb_maps, mb_heroes, mb_actions
                )

                # Policy Loss (Clipped Surrogate Objective)
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

                # Value Loss (Clipped Value Function Objective - Optional but often used)
                # mb_old_values = batch["old_values"] # Retrieve old values if needed for clipping
                # value_loss_unclipped = F.mse_loss(new_values, mb_returns)
                # values_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -self.config.clip_epsilon, self.config.clip_epsilon)
                # value_loss_clipped = F.mse_loss(values_clipped, mb_returns)
                # value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                # --- Using simpler MSE value loss ---
                value_loss = 0.5 * F.mse_loss(new_values, mb_returns)

                # Entropy Loss
                entropy_loss = entropy.mean()

                # Total Loss
                loss = (
                    policy_loss
                    - self.config.entropy_coef * entropy_loss
                    + self.config.vf_coef * value_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                # Record metrics for this minibatch
                all_metrics["policy_loss"].append(policy_loss.item())
                all_metrics["value_loss"].append(value_loss.item())
                all_metrics["entropy"].append(entropy_loss.item())

        self.total_updates_performed += 1
        # Calculate average metrics over all minibatches in this update
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items() if v}
        return avg_metrics

    def save_checkpoint(self, path: str, episode: int):
        """Saves agent state checkpoint, tagged by episode number."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "episode": episode,  # Save the episode index
            "total_steps_interacted": self.total_steps_interacted,
            "total_updates_performed": self.total_updates_performed,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,  # Save config for reference (optional)
        }
        torch.save(checkpoint, path)
        # print(f"Checkpoint saved to {path} at episode {episode+1}") # Optional verbosity

    def load_checkpoint(self, path: str) -> int:
        """Loads agent state from checkpoint, returns episode to resume from."""
        if not os.path.exists(path):
            print(
                f"Warning: Checkpoint file not found at {path}. Starting from episode 0."
            )
            return 0

        try:
            checkpoint = torch.load(path, map_location=self.device)
            # Config compatibility check (optional but recommended)
            chk_config = checkpoint.get("config")
            # Basic check example: compare action_dim and mode
            if chk_config and (
                chk_config.action_dim != self.config.action_dim
                or chk_config.mode != self.config.mode
            ):
                print(
                    f"Warning: Config mismatch (ActionDim/Mode) in checkpoint vs current. Loading weights anyway, but check config compatibility."
                )
            elif chk_config:
                # Optionally update current config with some loaded values if desired
                # Be cautious with this - only restore non-architecture defining params maybe?
                pass

            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.total_steps_interacted = checkpoint.get("total_steps_interacted", 0)
            self.total_updates_performed = checkpoint.get("total_updates_performed", 0)
            start_episode = (
                checkpoint.get("episode", -1) + 1
            )  # Resume from next episode

            print(
                f"Checkpoint loaded from {path}. Resuming from episode {start_episode} "
                f"(Total updates in chk: {self.total_updates_performed}, Steps: {self.agent.total_steps_interacted:,})."
            )  # Use agent.total_steps... here
            return start_episode
        except Exception as e:
            print(f"Error loading checkpoint from {path}: {e}. Starting from scratch.")
            # Reset agent state? Potentially dangerous if partial load happened.
            # Re-initializing might be safer depending on error.
            # For simplicity, we just return 0, assuming the __init__ sets fresh state.
            return 0


# --- Visualization (Identical) ---
def print_map(console: Console, map_tensor: torch.Tensor, title: str = "Generated Map"):
    """Prints a map tensor to the console with colors using Rich."""
    if map_tensor.ndim == 3 and map_tensor.shape[0] == 1:
        map_tensor = map_tensor.squeeze(0)
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
        6: "green",
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
            f"[{colors.get(tile, default_color)}]{tile:>{char_width-1}}[/]"
            for tile in map_np[r]
        ]  # Adjusted padding
        table.add_row(*row_cells)
    console.print(table)


# --- Training Orchestrator (Revised for Episodic Training & Metrics) ---
class PPOTrainer:
    """Orchestrates the PPO training process with episodic resets."""

    def __init__(self, config: PPOConfig, critic_func: Callable):
        self.config = config
        self.critic_func = critic_func
        self.device = get_device(config.device)
        self.console = Console()

        # Seed everything
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(config.seed)
            # Consider these for reproducibility vs performance
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

        # Initialize components
        self.env = BatchedEnvSimulator(config, critic_func)
        self.agent = PPOAgent(config)
        # Memory size based on PPO rollout length
        self.memory = PPOMemory(
            config.num_envs,
            config.n_steps_per_rollout,
            config.map_size,
            config.hero_tensor_size,
            self.device,
        )

        self.start_episode = 0  # Track the episode to start/resume from

    def train(self):
        """Runs the main training loop over episodes."""
        cfg = self.config  # Shorthand

        # --- Setup ---
        self.console.print(
            Panel.fit(
                f"Starting PPO Episodic Training: mode='{cfg.mode}', run='{cfg.run_name}'",
                title="Setup",
                border_style="blue",
            )
        )
        self.console.print(
            f"Device: [cyan]{self.device}[/cyan], Episodes: {cfg.num_episodes}, Steps/Episode: {cfg.episode_length}"
        )
        # Highlight the rollout length which determines update frequency
        self.console.print(
            f"Envs: {cfg.num_envs}, Steps/Rollout (Update Freq): [bold yellow]{cfg.n_steps_per_rollout}[/], PPO Epochs: {cfg.num_epochs_per_update}, Minibatch: {cfg.minibatch_size}"
        )
        self.console.print(f"Total Transitions per Update: {cfg.batch_size}")
        self.console.print(f"Checkpoints Dir: [green]{cfg.checkpoint_dir}[/green]")
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)  # Ensure checkpoint dir exists

        # --- Load Checkpoint ---
        latest_checkpoint_path = os.path.join(
            cfg.checkpoint_dir, "latest_checkpoint.pth"
        )
        if os.path.exists(latest_checkpoint_path):
            self.start_episode = self.agent.load_checkpoint(latest_checkpoint_path)
            if self.start_episode >= cfg.num_episodes:
                self.console.print(
                    f"[yellow]Checkpoint indicates training already completed ({self.start_episode}/{cfg.num_episodes} episodes). Exiting.[/yellow]"
                )
                return

        # --- Setup Progress Bar for Episodes ---
        episode_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            # Add the metrics field placeholder
            TextColumn("[bold]Metrics:[/]{task.fields[metrics]}", justify="left"),
            console=self.console,
            transient=False,  # Keep the bar after completion
        )

        # --- Episodic Training Loop ---
        map_obs, hero_obs = None, None  # Will be initialized in the loop

        with episode_progress:
            episode_task = episode_progress.add_task(
                "[cyan]Training Episodes",
                total=cfg.num_episodes,
                completed=self.start_episode,
                metrics=" Starting...",  # Initial metrics text
            )

            for episode in range(self.start_episode, cfg.num_episodes):
                # --- Start of Episode ---
                map_obs, hero_obs, initial_maps_cpu = (
                    self.env.reset()
                )  # Reset envs, get initial state & maps
                episode_rewards = []
                steps_this_episode = 0
                # Store the metrics from the last PPO update in this episode
                last_update_metrics = {}

                # Print Initial Maps for this episode (Optional)
                if (
                    episode + 1
                ) % cfg.print_maps_freq == 0 or episode == self.start_episode:
                    num_maps_to_print = min(3, cfg.num_envs)
                    # Use the progress bar's console to avoid overlap issues
                    episode_progress.console.print(
                        Panel(
                            f"--- Episode {episode+1}: Initial Maps (First {num_maps_to_print}) ---",
                            expand=False,
                        )
                    )
                    for i in range(num_maps_to_print):
                        print_map(
                            episode_progress.console,
                            initial_maps_cpu[i],
                            title=f"Ep {episode+1} Initial (Env {i})",
                        )
                    episode_progress.console.print(
                        "-" * (cfg.map_size[1] * 3 + 4)
                    )  # Adjust width based on map size/char width

                # --- Inner Loop: Steps within the episode ---
                # Use tqdm for steps within the episode (tracks progress towards episode_length)
                # `leave=False` removes the bar once the inner loop finishes for the episode
                # `position=0` tries to keep it at the top level
                with tqdm(
                    total=cfg.episode_length,
                    desc=f"Ep {episode+1}/{cfg.num_episodes} Steps",
                    leave=False,
                    unit="step",
                    position=0,
                ) as step_pbar:
                    while steps_this_episode < cfg.episode_length:
                        # A) Collect one PPO rollout (cfg.n_steps_per_rollout steps)
                        #    or fewer if the episode ends soon.
                        self.memory.clear()  # Clear memory for the new rollout
                        steps_to_run_this_rollout = min(
                            cfg.n_steps_per_rollout,
                            cfg.episode_length - steps_this_episode,
                        )

                        # Check if there are any steps left to run in this potential rollout
                        if steps_to_run_this_rollout <= 0:
                            break  # Exit inner loop if episode length reached

                        for _ in range(steps_to_run_this_rollout):
                            action, log_prob, value = self.agent.select_action(
                                map_obs, hero_obs, cfg.temperature
                            )
                            next_obs_tuple, reward, done, info = self.env.step(action)
                            next_map_obs, next_hero_obs = next_obs_tuple

                            # Add experience to PPO memory
                            self.memory.add(
                                map_obs, hero_obs, action, log_prob, reward, done, value
                            )
                            map_obs, hero_obs = (
                                next_map_obs,
                                next_hero_obs,
                            )  # IMPORTANT: Update state

                            # Track rewards and steps
                            episode_rewards.append(
                                reward.mean().item()
                            )  # Track avg reward per step
                            self.agent.total_steps_interacted += cfg.num_envs
                            steps_this_episode += 1
                            step_pbar.update(1)  # Update step progress bar

                        # B) Perform PPO Update using the collected rollout data (if any data was collected)
                        #    This happens every `cfg.n_steps_per_rollout` steps (or fewer at episode end)
                        if (
                            self.memory.ptr > 0
                        ):  # Ensure memory isn't empty before update
                            last_update_metrics = self.agent.update(
                                self.memory
                            )  # Store metrics from this update

                # --- End of Episode ---
                avg_ep_reward = np.mean(episode_rewards) if episode_rewards else 0

                # *** Update Episode Progress Bar with Metrics ***
                # Ensure metrics dict has values before formatting
                p_loss = last_update_metrics.get("policy_loss", float("nan"))
                v_loss = last_update_metrics.get("value_loss", float("nan"))
                ent = last_update_metrics.get("entropy", float("nan"))

                metrics_str = (
                    f"AvgRew:[yellow]{avg_ep_reward:>7.3f}[/]| "
                    f"P_Loss:[red]{p_loss:>7.3f}[/]| "
                    f"V_Loss:[magenta]{v_loss:>7.3f}[/]| "
                    f"Ent:[blue]{ent:>6.3f}[/]"
                )
                episode_progress.update(episode_task, advance=1, metrics=metrics_str)

                # Print Final Maps for this episode (Optional)
                if (
                    episode + 1
                ) % cfg.print_maps_freq == 0 or episode == cfg.num_episodes - 1:
                    num_maps_to_print = min(3, cfg.num_envs)
                    final_maps_cpu = self.env.maps.detach().cpu()  # Get current maps
                    # Use the progress bar's console
                    episode_progress.console.print(
                        Panel(
                            f"--- Episode {episode+1}: Final Maps (First {num_maps_to_print}) ---",
                            expand=False,
                        )
                    )
                    for i in range(num_maps_to_print):
                        print_map(
                            episode_progress.console,
                            final_maps_cpu[i],
                            title=f"Ep {episode+1} Final (Env {i})",
                        )
                    episode_progress.console.print(
                        "-" * (cfg.map_size[1] * 3 + 4)
                    )  # Adjust width

                # Save Checkpoint periodically (based on episode)
                if (episode + 1) % cfg.save_checkpoint_freq == 0:
                    chk_path = os.path.join(
                        cfg.checkpoint_dir, f"checkpoint_ep_{episode+1}.pth"
                    )
                    self.agent.save_checkpoint(chk_path, episode)
                    latest_path = os.path.join(
                        cfg.checkpoint_dir, "latest_checkpoint.pth"
                    )
                    # Save latest checkpoint atomically (rename)
                    temp_latest_path = latest_path + ".tmp"
                    self.agent.save_checkpoint(temp_latest_path, episode)
                    os.replace(temp_latest_path, latest_path)  # Atomic rename

        # --- End of Training ---
        self.console.print(
            Panel(
                f"Training finished after {cfg.num_episodes} episodes ({self.agent.total_steps_interacted:,} total steps, {self.agent.total_updates_performed:,} updates).",
                title="Complete",
                border_style="green",
            )
        )
        final_chk_path = os.path.join(cfg.checkpoint_dir, "final_checkpoint.pth")
        self.agent.save_checkpoint(
            final_chk_path, cfg.num_episodes - 1
        )  # Save index of last completed episode
        self.console.print(
            f"Final checkpoint saved to: [green]{final_chk_path}[/green]"
        )


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration Loading ---
    config_path = "ppo_episodic_config.yaml"  # Name of your YAML config file
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        config = configure_from_yaml(config_path)
    else:
        print(
            f"Configuration file '{config_path}' not found. Using default settings defined in PPOConfig."
        )
        config = PPOConfig()
        # Optionally override specific defaults if no YAML is found
        # config.n_steps_per_rollout = 64 # Ensure update frequency is set if needed
        # config.num_episodes = 500       # Shorten for quick test

    # --- Initialize Critic ---
    # >>> REPLACE THIS with your actual critic function <<<
    actual_critic = level_critic  # Ensure this points to your actual critic

    # --- Create and Run Trainer ---
    trainer = PPOTrainer(config, actual_critic)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print(
            "\n[yellow]Training interrupted by user. Saving final checkpoint...[/yellow]"
        )
        # Determine the last fully completed episode for saving
        # Find latest checkpoint episode, or use 0 if none exist
        latest_episode = -1
        latest_checkpoint_path = os.path.join(
            trainer.config.checkpoint_dir, "latest_checkpoint.pth"
        )
        if os.path.exists(latest_checkpoint_path):
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
                latest_episode = checkpoint.get("episode", -1)
            except Exception as e:
                print(f"Could not read latest checkpoint episode: {e}")

        interrupted_chk_path = os.path.join(
            trainer.config.checkpoint_dir, "final_checkpoint_interrupted.pth"
        )
        trainer.agent.save_checkpoint(
            interrupted_chk_path, latest_episode
        )  # Save with last known good episode index
        print(
            f"Interrupted state checkpoint saved to {interrupted_chk_path} (based on episode {latest_episode+1})"
        )
    except Exception as e:
        trainer.console.print_exception(show_locals=True)  # Print rich traceback
        print(
            "\n[red]An error occurred during training. Attempting to save final checkpoint...[/red]"
        )
        # Determine the last fully completed episode similarly
        latest_episode = -1
        latest_checkpoint_path = os.path.join(
            trainer.config.checkpoint_dir, "latest_checkpoint.pth"
        )
        if os.path.exists(latest_checkpoint_path):
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
                latest_episode = checkpoint.get("episode", -1)
            except Exception as load_e:
                print(
                    f"Could not read latest checkpoint episode during error handling: {load_e}"
                )

        error_chk_path = os.path.join(
            trainer.config.checkpoint_dir, "final_checkpoint_error.pth"
        )
        trainer.agent.save_checkpoint(error_chk_path, latest_episode)
        print(
            f"Error state checkpoint saved to {error_chk_path} (based on episode {latest_episode+1})"
        )

    # Optional Post-Training Test (remains the same)
    console = Console()
    print("\n--- Optional: Testing Network Forward Pass (Final Config/State) ---")
    try:
        # Use the trainer's agent state directly after training
        test_agent = trainer.agent
        # Or explicitly load the final checkpoint:
        # test_agent = PPOAgent(config)
        # final_checkpoint_path = os.path.join(config.checkpoint_dir, "final_checkpoint.pth")
        # if os.path.exists(final_checkpoint_path):
        #      test_agent.load_checkpoint(final_checkpoint_path)
        # else:
        #      raise FileNotFoundError("Final checkpoint not found for testing.")

        # Create dummy input data
        test_map = torch.rand((2, 1, config.map_size[0], config.map_size[1])).to(
            config.device
        )
        test_hero = torch.rand((2, config.hero_tensor_size)).to(config.device)
        test_agent.actor.eval()  # Set to eval mode
        with torch.no_grad():
            logits = test_agent.actor(test_map, test_hero)
            probs = F.softmax(logits, dim=-1)

        console.print(f"Mode: {config.mode}, Action Dim: {config.action_dim}")
        console.print(f"Input Map: {test_map.shape}, Input Hero: {test_hero.shape}")
        console.print(f"Output Logits: {logits.shape}, Output Probs: {probs.shape}")
        assert logits.shape == (2, config.action_dim), "Output shape mismatch!"
        console.print("[green]Network forward pass test successful.[/green]")
    except FileNotFoundError:
        console.print(
            "[yellow]Agent state/Final checkpoint not available for network test.[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Network forward pass test failed: {e}[/red]")
        console.print_exception()  # Print traceback for the test failure
