# --- START OF FILE ppo_agentsv2.py ---

# --- Imports ---
# ... (keep existing imports: torch, nn, optim, etc.) ...
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import random
import yaml
import os
import sys  # <-- Added for path adjustment if needed
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

# --- Import Critic Components ---
# Assume running from base_directory, adjust path if necessary
# Option 1: Direct relative import (if structure allows)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'critics')))
# Option 2: Assume 'daedalus' is a package installable or in PYTHONPATH
try:
    # Import the original symbolic critic
    from daedalus.critics import level_critic as actual_critic

    # Import the neural critic approximation components
    from daedalus.critics.critic_approximator import CriticApproximatorMLP, CriticConfig
except ImportError as e:
    print(f"Error importing Daedalus components: {e}")
    print("Please ensure 'daedalus' is installed or PYTHONPATH is set correctly.")
    print("Attempting relative import based on assumed structure...")
    try:
        # Adjust path assuming agents/ and critics/ are siblings
        critics_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "critics")
        )
        sys.path.insert(0, critics_path)
        # Need path to daedalus/critics for original level_critic too
        daedalus_critics_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "daedalus", "critics")
        )
        sys.path.insert(0, daedalus_critics_path)

        from critic_approximator import CriticApproximatorMLP, CriticConfig
        import level_critic as actual_critic

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
# ... (Paste the existing NN definitions here) ...
class PolicyNetworkEncoder(nn.Module):
    """Encodes the map and hero state into a latent representation using a NN (MLP)."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (12, 12),
        output_size: int = 1024,
        hero_tensor_size: int = 5,
        hidden_dims: List[int] = [512, 512],  # <<< MLP hidden layers configuration
    ):
        super().__init__()
        self.input_size = self.input_size_x, self.input_size_y = input_size
        self.map_flat_dim = self.input_size_x * self.input_size_y
        self.hero_tensor_size = hero_tensor_size
        self.input_dim = self.map_flat_dim + self.hero_tensor_size
        self.output_size = output_size
        self.hidden_dims = hidden_dims

        # --- NN/MLP Layers ---
        layers = []
        current_dim = self.input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        # Final layer to produce the desired output size
        layers.append(nn.Linear(current_dim, self.output_size))
        layers.append(
            nn.ReLU()
        )  # Activation after final linear layer (common practice)

        self.net = nn.Sequential(*layers)
        # --------------------

    def forward(self, x: torch.Tensor, hero_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass for the NN encoder."""
        # x has shape (batch, channels=1, height, width)
        # Flatten the map input
        x_flat = torch.flatten(x, start_dim=1)  # Shape: (batch, height * width)

        hero_tensor = hero_tensor.float()  # Ensure float type

        # Concatenate flattened map and hero tensor
        combined = torch.cat(
            [x_flat, hero_tensor], dim=1
        )  # Shape: (batch, map_flat_dim + hero_tensor_size)

        # Pass through the MLP
        latent = self.net(combined)
        return latent


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
        input_size: Tuple[int, int] = (12, 12),  # <<< Pass map size to encoder
        encoder_output_size: int = 1024,
        decoder_hidden_sizes: List[int] = [512, 256],
        action_dim: int = 7,
        hero_tensor_size: int = 5,
        encoder_hidden_dims: List[int] = [
            512,
            512,
        ],  # <<< Config for NN encoder hidden layers
    ):
        super().__init__()
        self.encoder = PolicyNetworkEncoder(
            input_size=input_size,
            output_size=encoder_output_size,
            hero_tensor_size=hero_tensor_size,
            hidden_dims=encoder_hidden_dims,  # <<< Pass encoder hidden dims
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
        input_size: Tuple[int, int] = (12, 12),  # <<< Pass map size to encoder
        encoder_output_size: int = 1024,
        decoder_hidden_sizes: List[int] = [512, 256],
        hero_tensor_size: int = 5,
        encoder_hidden_dims: List[int] = [
            512,
            512,
        ],  # <<< Config for NN encoder hidden layers
    ):
        super().__init__()
        self.encoder = PolicyNetworkEncoder(
            input_size=input_size,
            output_size=encoder_output_size,
            hero_tensor_size=hero_tensor_size,
            hidden_dims=encoder_hidden_dims,  # <<< Pass encoder hidden dims
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

    # --- Neural Critic Integration ---
    use_neural_critic: bool = True  # <<< Enable/disable neural critic
    # <<< Path relative to the base directory (where you run the script)
    neural_critic_checkpoint_path: str = (
        "critics/neural_critic_checkpoints/critic_MLP_20250422_113333/latest_checkpoint.pth"
    )

    # Calculated properties
    batch_size: int = field(init=False)
    action_dim: int = field(init=False)

    def __post_init__(self):
        """Calculate derived properties after initialization."""
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
            critic_type = "NeuralCritic" if self.use_neural_critic else "ActualCritic"
            self.run_name = f"ppo_{self.mode}_{critic_type}_ep_{time.strftime('%Y%m%d_%H%M%S')}"  # <<< Include critic type in run name
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.run_name)
        self.log_file_name = f"ppo_training_{self.run_name}.log"


def configure_from_yaml(yaml_path: str) -> PPOConfig:
    """Loads configuration from a YAML file, filtering unknown keys."""
    try:
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        valid_keys = {f.name for f in fields(PPOConfig) if f.init}
        filtered_config = {k: v for k, v in yaml_config.items() if k in valid_keys}

        # Ensure boolean conversion if loading from YAML
        if "use_neural_critic" in filtered_config:
            filtered_config["use_neural_critic"] = bool(
                filtered_config["use_neural_critic"]
            )

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
    return torch.device(device_str)


# ... (keep initialize_map_hero function as is) ...
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
            else random.randint(
                2, 5
            )  # Note: Original Critic uses 2-5 for enemies, 6 for door. Neural critic trained on 1, 6, 3, 4, 5
        )  # Consider aligning initial map generation with critic training data if needed.
        map_tensor[current_pos] = tile_value
        dx, dy = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        next_x = max(0, min(size_x - 1, current_pos[0] + dx))
        next_y = max(0, min(size_y - 1, current_pos[1] + dy))
        current_pos = (next_x, next_y)
    if map_tensor[start_pos] == 0:
        map_tensor[start_pos] = 1  # Ensure start pos is walkable (empty)
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


# --- Environment Simulation (Modified __init__) ---
class BatchedEnvSimulator:
    # <<< Takes the critic function to use directly >>>
    def __init__(self, config: PPOConfig, critic_function_to_use: Callable):
        self.config = config
        self.num_envs = config.num_envs
        self.map_size_x, self.map_size_y = config.map_size
        self.critic_func = critic_function_to_use  # <<< Use the provided function
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

    # ... (keep reset, _get_observation methods as is) ...
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
        map_obs = self.maps.unsqueeze(1).float()
        hero_obs = self.heroes.float()
        return map_obs, hero_obs

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        actions_np = actions.cpu().numpy()
        for i in range(self.num_envs):
            action = actions_np[i]
            pos_x, pos_y = self.agent_positions[i]
            current_map = self.maps[i]
            mode = self.config.mode
            if mode == "narrow":
                # Action map: 0:Wall, 1:Empty, 2:Enemy1, 3:Enemy2, 4:Enemy3, 5:Enemy4, 6:Door
                # Neural critic was trained on: 1:Empty, 6:Door, [3, 4, 5]:Enemies. 0 assumed wall/unused?
                # We might need to map PPO actions to the critic's expected tile values if they differ.
                # Assuming the PPO actions 0-6 directly correspond to tile values expected by BOTH critics for now.
                # Check critic_approximator.py tile values if issues arise.
                if 0 <= action <= 6:
                    current_map[pos_x, pos_y] = action
                # Move agent randomly
                dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                next_x = max(0, min(self.map_size_x - 1, pos_x + dx))
                next_y = max(0, min(self.map_size_y - 1, pos_y + dy))
                self.agent_positions[i] = (next_x, next_y)
            elif mode == "turtle":
                # Actions 0-5: Place tile type 0-5
                # Actions 6-9: Move Turtle (Up, Left, Down, Right)
                # Check critic_approximator.py tile values if issues arise.
                if 0 <= action <= 5:
                    current_map[pos_x, pos_y] = action
                elif action == 6:  # Up
                    pos_x = max(0, pos_x - 1)  # Changed from wrap-around to bounded
                elif action == 7:  # Left
                    pos_y = max(0, pos_y - 1)  # Changed from wrap-around to bounded
                elif action == 8:  # Down
                    pos_x = min(
                        self.map_size_x - 1, pos_x + 1
                    )  # Changed from wrap-around to bounded
                elif action == 9:  # Right
                    pos_y = min(
                        self.map_size_y - 1, pos_y + 1
                    )  # Changed from wrap-around to bounded
                self.agent_positions[i] = (pos_x, pos_y)
            elif mode == "wide":
                num_tile_types = 7  # 0-6
                n_tiles_total = self.map_size_x * self.map_size_y
                if self.config.action_dim != num_tile_types * n_tiles_total:
                    # This check might be overly strict if action_dim was manually set, but good default
                    print(
                        f"Warning: Action dim mismatch for wide mode. Check config. Expected {num_tile_types * n_tiles_total}, got {self.config.action_dim}"
                    )
                tile_type = action // n_tiles_total
                flat_index = action % n_tiles_total
                target_x = flat_index // self.map_size_y
                target_y = flat_index % self.map_size_y
                if 0 <= tile_type < num_tile_types:
                    # Again, assuming tile_type 0-6 matches critic expectations
                    current_map[target_x, target_y] = tile_type
                # Agent position doesn't change in 'wide' mode per action
            else:
                raise ValueError(f"Unknown mode in step: {mode}")

        # --- Calculate Rewards using the provided critic function ---
        # Clone maps to avoid potential modification by critic (good practice)
        # The critic_func here is either the neural wrapper or actual_critic
        rewards = self.critic_func(self.maps.clone(), self.heroes.clone())
        # -------------------------------------------------------------

        # Episode termination is not handled here (fixed length episodes)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        next_obs = self._get_observation()
        infos = {}  # Placeholder for additional info if needed
        return next_obs, rewards, dones, infos


# --- PPO Memory (Identical) ---
# ... (keep PPOMemory class as is) ...
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
        # Dones are always False in this setup, so next_non_terminal is always 1.0
        # Simplified calculation possible, but keeping general form for clarity.
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = (
                    1.0  # (1.0 - self.dones[t].float()) # Effectively 1.0
                )
                next_values = last_value
            else:
                next_non_terminal = (
                    1.0  # (1.0 - self.dones[t].float()) # Effectively 1.0
                )
                next_values = self.values[t + 1]
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            # GAE calculation
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        # Calculate returns
        self.returns = self.advantages + self.values
        # Reset pointer for next rollout or batch generation
        self.ptr = 0

    def get_minibatches(
        self, batch_size: int, minibatch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        if self.advantages is None or self.returns is None:
            raise ValueError("Advantages/returns not computed.")

        # Reshape data from (n_steps, num_envs, ...) to (batch_size, ...)
        # Note: self.ptr should be equal to self.n_steps if buffer is full after compute_gae_returns
        num_transitions = (
            self.n_steps * self.num_envs
        )  # Total transitions in the buffer
        if num_transitions == 0:
            return  # No data

        # Ensure data exists up to n_steps before reshaping
        if self.maps.shape[0] < self.n_steps:
            raise ValueError(
                f"Expected maps buffer size {self.n_steps}, got {self.maps.shape[0]}"
            )
            # Similar checks for other buffers...

        # Flatten rollouts
        flat_maps = self.maps.reshape(num_transitions, *self.maps.shape[2:])
        flat_heroes = self.heroes.reshape(num_transitions, -1)
        flat_actions = self.actions.reshape(-1)
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values.reshape(-1)  # Keep old values for potential clipping

        indices = torch.randperm(num_transitions).to(self.device)

        if minibatch_size > num_transitions:
            print(
                f"Warning: Minibatch size ({minibatch_size}) > available transitions ({num_transitions}). Using {num_transitions} as minibatch size."
            )
            minibatch_size = num_transitions

        for start_idx in range(0, num_transitions, minibatch_size):
            end_idx = start_idx + minibatch_size
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
                "old_values": flat_values[
                    mb_indices
                ],  # Pass old values if needed for value clipping
            }

    def clear(self):
        # Reset pointer, does not erase data but marks buffer as ready for new samples
        self.ptr = 0
        # Optionally zero out tensors if memory safety is a concern, but usually not necessary
        # self.maps.zero_()
        # ... etc ...


# --- PPO Agent (Identical) ---
# ... (keep PPOAgent class as is) ...
class PPOAgent:
    """The PPO Agent class containing policy/value networks and update logic."""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = get_device(config.device)
        # <<< Pass necessary config values to networks >>>
        self.actor = PPOPolicyNetwork(
            input_size=config.map_size,
            encoder_output_size=config.encoder_output_size,
            decoder_hidden_sizes=config.decoder_hidden_sizes,
            action_dim=config.action_dim,
            hero_tensor_size=config.hero_tensor_size,
            encoder_hidden_dims=config.encoder_hidden_dims,  # <<< Pass encoder dims
        ).to(self.device)
        self.critic = ValueNetwork(
            input_size=config.map_size,
            encoder_output_size=config.encoder_output_size,
            decoder_hidden_sizes=config.decoder_hidden_sizes,
            hero_tensor_size=config.hero_tensor_size,
            encoder_hidden_dims=config.encoder_hidden_dims,  # <<< Pass encoder dims
        ).to(self.device)
        # Combine parameters for the optimizer
        all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(
            all_params,
            lr=config.learning_rate,
            eps=1e-5,  # Add epsilon for numerical stability (common in Adam)
        )
        self.total_steps_interacted = 0
        self.total_updates_performed = 0

    def select_action(
        self, map_obs: torch.Tensor, hero_obs: torch.Tensor, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects actions based on policy, returns action, log_prob, and value estimate."""
        self.actor.eval()  # Set to evaluation mode for sampling
        self.critic.eval()
        with torch.no_grad():
            action_logits = self.actor(map_obs, hero_obs)
            value = self.critic(map_obs, hero_obs).squeeze(-1)  # Shape (num_envs,)

            # Apply temperature scaling
            if temperature > 0:
                scaled_logits = action_logits / max(
                    temperature, 1e-8
                )  # Avoid division by zero
            else:
                # If temperature is 0 or less, use argmax (deterministic)
                scaled_logits = action_logits  # Keep original logits for argmax

            probs = F.softmax(scaled_logits, dim=-1)
            dist = Categorical(probs=probs)

            # Sample action if temperature > 0, otherwise take the best action
            if temperature > 0:
                action = dist.sample()
            else:
                action = torch.argmax(probs, dim=-1)

            log_prob = dist.log_prob(action)  # Log probability of the chosen action

        self.actor.train()  # Set back to training mode
        self.critic.train()
        return action, log_prob, value

    def evaluate_actions(
        self, map_obs: torch.Tensor, hero_obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates actions given state, returns log_probs, values, and entropy."""
        action_logits = self.actor(map_obs, hero_obs)
        value = self.critic(map_obs, hero_obs).squeeze(-1)  # Shape (batch_size,)

        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs=probs)

        log_prob = dist.log_prob(actions)  # Log probability of the input actions
        entropy = dist.entropy()  # Entropy of the action distribution

        return log_prob, value, entropy

    def update(self, memory: PPOMemory) -> Dict[str, float]:
        """Performs PPO update using data in memory."""
        if memory.ptr != memory.n_steps:
            # This shouldn't happen if compute_gae_returns resets ptr correctly
            # Or handle case where buffer isn't full? Usually update happens on full buffer.
            print(
                f"Warning: Memory buffer not full (ptr={memory.ptr}, n_steps={memory.n_steps}). Update might use partial data."
            )
            # Decide whether to proceed or return empty metrics if data is insufficient.
            # For now, proceed, assuming compute_gae was called and used available data.
            if memory.ptr == 0:
                return {}

        # 1. Compute GAE and returns (should be done *before* calling update)
        # Assuming memory.compute_gae_returns was called externally

        # 2. PPO Optimization Loop
        all_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        for _ in range(self.config.num_epochs_per_update):
            minibatch_generator = memory.get_minibatches(
                memory.n_steps * self.config.num_envs,  # Pass expected full batch size
                self.config.minibatch_size,
            )
            for batch in minibatch_generator:
                mb_maps, mb_heroes = batch["maps"], batch["heroes"]
                mb_actions, mb_old_log_probs = batch["actions"], batch["old_log_probs"]
                mb_advantages, mb_returns = batch["advantages"], batch["returns"]
                # mb_old_values = batch["old_values"] # Use if value clipping is implemented

                # Normalize advantages per minibatch
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Evaluate current policy on minibatch data
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
                # Optional: Value function clipping (like in SB3)
                # value_loss_unclipped = F.mse_loss(new_values, mb_returns)
                # clipped_values = mb_old_values + torch.clamp(new_values - mb_old_values, -self.config.clip_epsilon, self.config.clip_epsilon)
                # value_loss_clipped = F.mse_loss(clipped_values, mb_returns)
                # value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped)
                # Simpler MSE loss without clipping:
                value_loss = 0.5 * F.mse_loss(new_values, mb_returns)

                # --- Entropy Loss ---
                # Maximize entropy encourages exploration
                entropy_loss = entropy.mean()

                # --- Total Loss ---
                loss = (
                    policy_loss
                    - self.config.entropy_coef * entropy_loss
                    + self.config.vf_coef * value_loss
                )

                # --- Optimization Step ---
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                # --- Logging Metrics (Inside Minibatch Loop) ---
                all_metrics["policy_loss"].append(policy_loss.item())
                all_metrics["value_loss"].append(value_loss.item())
                all_metrics["entropy"].append(entropy_loss.item())

                # Optional: Calculate approximate KL divergence and clip fraction for monitoring
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1.0) > self.config.clip_epsilon).float()
                    ).item()
                    all_metrics["approx_kl"].append(approx_kl)
                    all_metrics["clip_fraction"].append(clip_fraction)

        self.total_updates_performed += 1
        # --- Aggregate Metrics (After Epochs) ---
        avg_metrics = {
            k: np.mean(v) for k, v in all_metrics.items() if v
        }  # Average over all minibatches in this update
        return avg_metrics

    def save_checkpoint(self, path: str, episode: int):
        """Saves model and optimizer state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + ".tmp"  # Save to temp file first
        try:
            checkpoint = {
                "episode": episode,
                "total_steps_interacted": self.total_steps_interacted,
                "total_updates_performed": self.total_updates_performed,
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,  # Save config for reference and potential compatibility checks
            }
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, path)  # Atomic replace
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint to {path}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)  # Clean up temp file

    def load_checkpoint(self, path: str) -> int:
        """Loads model and optimizer state, returns the next episode to start from."""
        if not os.path.exists(path):
            print(
                f"[Warning] Checkpoint file not found at {path}. Starting from episode 0."
            )
            return 0
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # --- Compatibility Check (Example) ---
            chk_config = checkpoint.get("config")
            if chk_config:
                # Compare critical parameters
                if chk_config.action_dim != self.config.action_dim:
                    print(
                        f"[Warning] Action dimension mismatch! Checkpoint: {chk_config.action_dim}, Current: {self.config.action_dim}. Loading weights anyway."
                    )
                if chk_config.mode != self.config.mode:
                    print(
                        f"[Warning] Mode mismatch! Checkpoint: {chk_config.mode}, Current: {self.config.mode}. Loading weights anyway."
                    )
                # Add more checks if needed (e.g., map size, network structure if changed)
            else:
                print(
                    "[Warning] Checkpoint does not contain configuration. Cannot perform compatibility checks."
                )
            # --- End Compatibility Check ---

            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load training progress
            self.total_steps_interacted = checkpoint.get("total_steps_interacted", 0)
            self.total_updates_performed = checkpoint.get("total_updates_performed", 0)
            start_episode = (
                checkpoint.get("episode", -1) + 1
            )  # Resume from the next episode

            print(
                f"Checkpoint loaded from {path}. Resuming from episode {start_episode} "
                f"(Updates: {self.total_updates_performed}, Steps: {self.agent.total_steps_interacted:,})."
            )
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
# ... (keep print_map function as is) ...
def print_map(console: Console, map_tensor: torch.Tensor, title: str = "Generated Map"):
    if map_tensor.ndim == 3 and map_tensor.shape[0] == 1:  # Handle (1, H, W) case
        map_tensor = map_tensor.squeeze(0)
    elif (
        map_tensor.ndim == 4 and map_tensor.shape[0] == 1 and map_tensor.shape[1] == 1
    ):  # Handle (1, 1, H, W) case
        map_tensor = map_tensor.squeeze(0).squeeze(0)

    if map_tensor.device != torch.device("cpu"):
        map_tensor = map_tensor.cpu()

    # Ensure it's 2D before proceeding
    if map_tensor.ndim != 2:
        console.print(
            f"[red]Error: Cannot print map with shape {map_tensor.shape}. Expected 2D.[/red]"
        )
        return

    map_np = map_tensor.numpy().astype(int)
    map_size_x, map_size_y = map_np.shape

    # Define colors (adjust if tile values differ significantly between critics/modes)
    colors = {
        0: "dim grey50",  # Wall/Unused?
        1: "white",  # Empty (Common)
        6: "bright_green",  # Door (Common in critic training)
        # Enemy colors (Check critic_approximator.py and level_critic.py conventions)
        2: "bright_red",  # Enemy type 1?
        3: "red",  # Enemy type 2? (Used in critic training)
        4: "dark_red",  # Enemy type 3? (Used in critic training)
        5: "red3",  # Enemy type 4? (Used in critic training)
        # Add more if needed
    }
    default_color = "magenta"  # Color for unexpected tile values
    char_width = 2  # Width for each cell in the table

    table = Table(
        title=title,
        show_header=False,
        show_edge=True,
        box=None,  # No outer box for cleaner look
        padding=0,
        expand=False,  # Don't expand table to console width
    )
    # Add columns with fixed width
    for _ in range(map_size_y):
        table.add_column(justify="center", width=char_width)

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
        # base_critic_func is the original symbolic critic (actual_critic)
        # We will decide which function to *use* based on config
        self.base_critic_func = base_critic_func
        self.device = get_device(config.device)
        self.console = Console()
        self.logger = logging.getLogger(__name__)

        # Seed everything
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)  # For multi-GPU
            # Potentially add deterministic behavior settings
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

        # --- Setup Logging ---
        self._setup_logging()

        # --- Initialize Neural Critic (if configured) ---
        self.neural_critic: Optional[CriticApproximatorMLP] = None
        self.critic_func_to_use: Callable = self.base_critic_func  # Default to original

        if config.use_neural_critic:
            self.logger.info(
                f"Attempting to load Neural Critic from: {config.neural_critic_checkpoint_path}"
            )
            try:
                critic_checkpoint_path = config.neural_critic_checkpoint_path
                if not os.path.exists(critic_checkpoint_path):
                    raise FileNotFoundError(
                        f"Neural critic checkpoint not found at {critic_checkpoint_path}"
                    )

                # Load checkpoint with config (weights_only=False is crucial)
                critic_checkpoint = torch.load(
                    critic_checkpoint_path, map_location=self.device, weights_only=False
                )

                if "config" not in critic_checkpoint:
                    raise KeyError("Critic checkpoint missing 'config' field.")
                if "model_state_dict" not in critic_checkpoint:
                    raise KeyError(
                        "Critic checkpoint missing 'model_state_dict' field."
                    )

                critic_config: CriticConfig = critic_checkpoint["config"]
                self.logger.info(
                    f"Loaded critic configuration: MapSize={critic_config.map_size}, Hidden={critic_config.mlp_hidden_sizes}"
                )

                # --- Sanity Check: Map Size Compatibility ---
                if critic_config.map_size != self.config.map_size:
                    self.logger.warning(
                        f"Map size mismatch! PPO Config: {self.config.map_size}, Loaded Critic Config: {critic_config.map_size}"
                    )
                    # Decide how to handle: warn, error, or adapt? For now, warn.
                    # raise ValueError("Map size mismatch between PPO config and loaded neural critic config.")

                # Instantiate the MLP critic model using loaded config
                self.neural_critic = CriticApproximatorMLP(
                    input_size=critic_config.map_size[0] * critic_config.map_size[1],
                    hidden_sizes=critic_config.mlp_hidden_sizes,
                    output_size=1,  # Critic outputs a single score
                    dropout_prob=critic_config.mlp_dropout_prob,  # Use dropout from its training config
                ).to(self.device)

                # Load the trained weights
                self.neural_critic.load_state_dict(
                    critic_checkpoint["model_state_dict"]
                )
                self.neural_critic.eval()  # Set to evaluation mode

                self.logger.info(
                    f"[bold green]Successfully loaded and initialized Neural Critic.[/bold green]"
                )

                # --- Create the wrapper function ---
                def neural_critic_wrapper(
                    map_tensor: torch.Tensor, hero_tensor: torch.Tensor
                ) -> torch.Tensor:
                    """
                    Wrapper to format input and call the neural critic MLP.
                    Input map_tensor expected shape: (N, H, W) or (N, 1, H, W), dtype int64 or float32
                    Input hero_tensor is ignored by the MLP critic.
                    Output shape: (N,) float32
                    """
                    # Ensure map_tensor is on the correct device
                    map_tensor = map_tensor.to(self.device)

                    # Ensure correct shape (N, 1, H, W)
                    if map_tensor.dim() == 3:  # Input is (N, H, W)
                        map_tensor = map_tensor.unsqueeze(1)
                    elif map_tensor.dim() != 4 or map_tensor.shape[1] != 1:
                        # Log error and potentially raise or return default scores
                        self.logger.error(
                            f"Invalid map tensor shape for neural critic: {map_tensor.shape}. Expected (N, 1, H, W)."
                        )
                        # Returning zeros as a fallback, consider raising an error instead
                        return torch.zeros(
                            map_tensor.shape[0], dtype=torch.float32, device=self.device
                        )
                        # raise ValueError(f"Invalid map tensor shape for neural critic: {map_tensor.shape}")

                    # Ensure correct dtype (float)
                    map_tensor = map_tensor.float()

                    # Call the neural critic model (no gradients needed)
                    with torch.no_grad():
                        scores = self.neural_critic(map_tensor)  # Output shape (N, 1)

                    # Return scores with shape (N,)
                    return scores.squeeze(-1)

                # Set the function to use for reward calculation
                self.critic_func_to_use = neural_critic_wrapper

            except Exception as e:
                self.logger.error(
                    f"[bold red]Failed to load Neural Critic: {e}[/bold red]"
                )
                self.logger.error(
                    "Falling back to the original symbolic critic function."
                )
                self.console.print(
                    f"[bold red]Error loading Neural Critic:[/bold red] {e}. Using symbolic critic."
                )
                self.critic_func_to_use = self.base_critic_func  # Fallback
                self.neural_critic = None  # Ensure it's None
        else:
            self.logger.info(
                "Using the original symbolic critic function (neural critic disabled)."
            )
            self.console.print(
                "Using original symbolic critic ([green]neural critic disabled[/green])."
            )
            self.critic_func_to_use = self.base_critic_func

        # Initialize components using the selected critic function
        # <<< Pass the determined critic function here >>>
        self.env = BatchedEnvSimulator(config, self.critic_func_to_use)
        self.agent = PPOAgent(config)  # Agent doesn't directly need the critic func
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

    def _setup_logging(self):
        """Configures the logging module."""
        log_dir = self.config.checkpoint_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, self.config.log_file_name)

        # Prevent duplicate handlers if called multiple times
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File Handler (Append mode)
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Optional Console Handler (using Rich console instead)
        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(formatter)
        # self.logger.addHandler(console_handler)

        self.logger.info(f"--- Logging started for run: {self.config.run_name} ---")
        # Log key configuration parameters
        self.logger.info(
            f"PPO Mode: {self.config.mode}, Map Size: {self.config.map_size}"
        )
        self.logger.info(
            f"Device: {self.config.device}, Num Envs: {self.config.num_envs}"
        )
        self.logger.info(f"Using Neural Critic: {self.config.use_neural_critic}")
        if self.config.use_neural_critic:
            self.logger.info(
                f"Neural Critic Path: {self.config.neural_critic_checkpoint_path}"
            )

    def train(self):
        """Runs the main training loop over episodes."""
        cfg = self.config

        # --- Setup ---
        critic_desc = (
            "[bold green]Neural Approximator[/]"
            if self.config.use_neural_critic and self.neural_critic
            else "[bold blue]Symbolic Critic[/]"
        )
        self.console.print(
            Panel.fit(
                f"Starting PPO Episodic Training: mode='{cfg.mode}', run='{cfg.run_name}'\nReward Critic: {critic_desc}",
                title="Setup",
                border_style="blue",
            )
        )
        # ... (rest of the setup prints)
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
        if self.config.use_neural_critic and self.neural_critic is None:
            self.console.print(
                "[bold red]Warning: Neural critic enabled but failed to load. Using symbolic critic instead.[/bold red]"
            )

        self.logger.info("Training setup complete.")

        # --- Load Checkpoint ---
        latest_checkpoint_path = os.path.join(
            cfg.checkpoint_dir, "latest_checkpoint.pth"
        )
        if os.path.exists(latest_checkpoint_path):
            self.logger.info(
                f"Loading PPO agent checkpoint from {latest_checkpoint_path}"
            )
            # Pass self.agent to load_checkpoint
            self.start_episode = self.agent.load_checkpoint(latest_checkpoint_path)
            # Note: Reward baseline state isn't saved/loaded in this version
            self.is_first_episode_for_baseline = self.start_episode == 0
            self.logger.info(f"Resuming from episode {self.start_episode}")
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

        # --- Setup Progress Bar ---
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
            transient=False,  # Keep the bar after completion
        )

        # --- Episodic Training Loop ---
        map_obs, hero_obs = None, None  # Will be initialized in env.reset()

        with episode_progress:
            episode_task = episode_progress.add_task(
                "[cyan]Training Episodes",
                total=cfg.num_episodes,
                completed=self.start_episode,
                metrics=" Starting...",
            )

            for episode in range(self.start_episode, cfg.num_episodes):
                # --- Start of Episode ---
                map_obs, hero_obs, initial_maps_cpu = self.env.reset()
                episode_rewards = []
                steps_this_episode = 0
                last_update_metrics = (
                    {}
                )  # Store metrics from the last update in the episode

                # --- Print Initial Maps ---
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
                        # Pass the console object to print_map
                        print_map(
                            episode_progress.console,
                            initial_maps_cpu[i],
                            title=f"Ep {episode + 1} Initial (Env {i})",
                        )
                    # episode_progress.console.print("-" * (cfg.map_size[1] * 3 + 4)) # Optional separator

                # --- Inner Loop: Steps within an Episode ---
                # Use Rich progress bar for steps *instead* of tqdm for better integration
                step_task_desc = f"Ep {episode + 1}/{cfg.num_episodes} Steps"
                step_task = episode_progress.add_task(
                    step_task_desc, total=cfg.episode_length, visible=True
                )

                rollout_step = 0
                while steps_this_episode < cfg.episode_length:
                    # --- Collect Rollout ---
                    if rollout_step == 0:  # Clear memory at the start of a new rollout
                        self.memory.clear()

                    # Determine how many steps to run in this interaction cycle
                    steps_to_run = min(
                        cfg.n_steps_per_rollout
                        - rollout_step,  # Remaining steps in rollout buffer
                        cfg.episode_length - steps_this_episode,
                    )  # Remaining steps in episode

                    if steps_to_run <= 0:
                        break  # Should not happen if logic is correct, but safe check

                    for _ in range(steps_to_run):
                        if steps_this_episode >= cfg.episode_length:
                            break  # Ensure we don't exceed episode length

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
                    if rollout_step == cfg.n_steps_per_rollout:
                        # Compute GAE and returns *before* update
                        with torch.no_grad():
                            # Get value estimate for the *last* observation in the rollout
                            last_value = self.agent.critic(map_obs, hero_obs).squeeze(
                                -1
                            )
                        self.memory.compute_gae_returns(
                            last_value, cfg.gamma, cfg.gae_lambda
                        )

                        # Perform PPO update epochs
                        update_metrics = self.agent.update(self.memory)
                        last_update_metrics = (
                            update_metrics  # Store latest metrics for logging
                        )

                        # Log PPO update metrics
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

                        # Reset rollout step counter and clear memory (implicitly done by next loop iteration's clear)
                        rollout_step = 0
                        # memory.clear() # Already called at the start of the next rollout

                # --- End of Episode ---
                avg_ep_reward = np.mean(episode_rewards) if episode_rewards else 0.0

                # Update Reward Baseline (EMA)
                if self.is_first_episode_for_baseline:
                    self.reward_baseline = avg_ep_reward
                    self.is_first_episode_for_baseline = False
                else:
                    alpha = cfg.reward_baseline_alpha
                    self.reward_baseline = (
                        alpha * avg_ep_reward + (1 - alpha) * self.reward_baseline
                    )

                # Log Episode Summary
                self.logger.info(
                    f"Ep: {episode + 1}/{cfg.num_episodes} finished. Steps: {steps_this_episode}. "
                    f"AvgReward: {avg_ep_reward:.4f}, RewardBaseline: {self.reward_baseline:.4f}, "
                    f"TotalSteps: {self.agent.total_steps_interacted}"
                )

                # Update Overall Episode Progress Bar
                p_loss = last_update_metrics.get("policy_loss", float("nan"))
                v_loss = last_update_metrics.get("value_loss", float("nan"))
                ent = last_update_metrics.get("entropy", float("nan"))
                metrics_str = (
                    f"AvgRew:[yellow]{avg_ep_reward:>7.3f}[/]| "
                    f"Baseline:[cyan]{self.reward_baseline:>7.3f}[/]| "
                    f"P:[red]{p_loss:>7.3f}[/]| "  # Shorter labels
                    f"V:[magenta]{v_loss:>7.3f}[/]| "
                    f"E:[blue]{ent:>6.3f}[/]"
                )
                episode_progress.update(episode_task, advance=1, metrics=metrics_str)
                episode_progress.remove_task(
                    step_task
                )  # Remove the completed step task

                # --- Print Final Maps ---
                if (
                    episode + 1
                ) % cfg.print_maps_freq == 0 or episode == cfg.num_episodes - 1:
                    num_maps_to_print = min(3, cfg.num_envs)
                    final_maps_cpu = self.env.maps.detach().cpu()  # Get current maps
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
                    # episode_progress.console.print("-" * (cfg.map_size[1] * 3 + 4))

                # --- Save Checkpoint ---
                if (episode + 1) % cfg.save_checkpoint_freq == 0:
                    # Save periodic checkpoint
                    chk_path = os.path.join(
                        cfg.checkpoint_dir, f"checkpoint_ep_{episode + 1}.pth"
                    )
                    self.agent.save_checkpoint(chk_path, episode)  # Save agent state
                    self.logger.info(
                        f"Periodic checkpoint saved to {chk_path} at episode {episode + 1}"
                    )

                    # Save latest checkpoint (overwrite)
                    latest_path = os.path.join(
                        cfg.checkpoint_dir, "latest_checkpoint.pth"
                    )
                    self.agent.save_checkpoint(latest_path, episode)  # Save agent state
                    self.logger.info(f"Latest checkpoint updated to {latest_path}")

        # --- End of Training ---
        msg = (
            f"Training finished after {cfg.num_episodes} episodes "
            f"({self.agent.total_steps_interacted:,} total steps, "
            f"{self.agent.total_updates_performed:,} updates)."
        )
        self.console.print(Panel(msg, title="Complete", border_style="green"))
        self.logger.info(msg)

        # Save final checkpoint
        final_chk_path = os.path.join(cfg.checkpoint_dir, "final_checkpoint.pth")
        self.agent.save_checkpoint(
            final_chk_path, cfg.num_episodes - 1
        )  # Save final agent state
        self.console.print(
            f"Final checkpoint saved to: [green]{final_chk_path}[/green]"
        )
        self.logger.info(f"Final checkpoint saved to {final_chk_path}")

        # Also update latest_checkpoint to be the same as final
        latest_path = os.path.join(cfg.checkpoint_dir, "latest_checkpoint.pth")
        self.agent.save_checkpoint(latest_path, cfg.num_episodes - 1)
        self.logger.info(f"Latest checkpoint updated to final state: {latest_path}")


# --- Main Execution ---
if __name__ == "__main__":
    DEFAULT_CONFIG_PATH = (
        "ppo_episodic_config.yaml"  # Config file in the same directory as the script
    )
    config_path = DEFAULT_CONFIG_PATH

    # Basic argument parsing for config file override
    import argparse

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

    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
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
    # Pass the *base* symbolic critic; trainer handles switching logic
    trainer = PPOTrainer(config, symbolic_critic)
    try:
        trainer.train()
    except KeyboardInterrupt:
        msg = "\nTraining interrupted by user. Saving final checkpoint..."
        trainer.console.print(f"[yellow]{msg}[/yellow]")
        trainer.logger.warning(msg)
        latest_episode = -1
        # Try to determine the last completed episode from agent's state if possible
        # This depends on how load_checkpoint returns/sets the episode
        # A simple approach is to read the latest checkpoint if it exists
        latest_checkpoint_path = os.path.join(
            trainer.config.checkpoint_dir, "latest_checkpoint.pth"
        )
        if os.path.exists(latest_checkpoint_path):
            try:
                # Load minimal info to get episode number
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
        trainer.agent.save_checkpoint(
            interrupted_chk_path, latest_episode
        )  # Save with last known completed episode
        final_msg = f"Interrupted state checkpoint saved to {interrupted_chk_path} (based on episode {latest_episode + 1} starting or interrupted during)"
        print(final_msg)
        trainer.logger.info(final_msg)
    except Exception as e:
        # Log the full exception traceback to the log file
        trainer.logger.error(
            "An critical error occurred during training.", exc_info=True
        )
        # Print exception to console using Rich
        trainer.console.print(
            "\n[bold red]An critical error occurred during training:[/bold red]"
        )
        trainer.console.print_exception(
            show_locals=False
        )  # show_locals=True can be verbose

        print("[bold red]Attempting to save final checkpoint...[/bold red]")
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
                    f"Could not read latest checkpoint episode during error handling: {load_e}"
                )

        error_chk_path = os.path.join(
            trainer.config.checkpoint_dir, "final_checkpoint_error.pth"
        )
        trainer.agent.save_checkpoint(error_chk_path, latest_episode)
        error_msg = f"Error state checkpoint saved to {error_chk_path} (based on episode {latest_episode + 1} starting or interrupted during)"
        print(error_msg)
        trainer.logger.info(error_msg)

    # --- Optional Post-Training Test ---
    console = Console()
    print("\n--- Optional: Testing Network Forward Pass (Final Config/State) ---")
    try:
        # Ensure trainer and agent were initialized
        if "trainer" in locals() and hasattr(trainer, "agent"):
            test_agent = trainer.agent
            current_config = trainer.config  # Use the config from the trainer instance

            # Create dummy input data matching the config
            batch_size = 2  # Test with a small batch
            test_map = torch.randint(
                0,
                7,
                size=(
                    batch_size,
                    1,
                    current_config.map_size[0],
                    current_config.map_size[1],
                ),
                dtype=torch.float32,
            ).to(current_config.device)
            test_hero = torch.rand((batch_size, current_config.hero_tensor_size)).to(
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
            )  # Should be (batch_size, 1)

            # Basic shape assertions
            assert action_logits.shape == (
                batch_size,
                current_config.action_dim,
            ), f"Actor output shape mismatch: {action_logits.shape}"
            assert state_values.shape == (
                batch_size,
                1,
            ), f"Critic output shape mismatch: {state_values.shape}"

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
