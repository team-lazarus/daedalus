#!/usr/bin/env python
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F # Keep for potential future use
import numpy as np
import random
import math
import os
import json # Keep for potential future use
import time # For basic timing if needed
from enum import Enum
from typing import Tuple, List, Optional, Dict, Any, Callable
from collections import deque, defaultdict

# Rich imports for map rendering
from rich.console import Console
from rich.logging import RichHandler # Keep if you want Spinup's logger to use Rich
from rich.text import Text

# Spinup imports
import spinup
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

# --- Setup Logging and Console ---
# Spinup handles its own logging via EpochLogger
console = Console()

# --- Enums and Constants ---
class Entry(Enum):
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"

# Tile Values (State Representation)
EMPTY_TILE = 0
FLOOR_TILE = 1 # Walkable path/area
ENEMY_1 = 2
ENEMY_2 = 3
ENEMY_3 = 4
ENEMY_4 = 5
DOOR = 6
# Derived constants
PLACEABLE_TILES = [FLOOR_TILE, ENEMY_1, ENEMY_2, ENEMY_3, ENEMY_4, DOOR]
NUM_PLACEABLE_TILES = len(PLACEABLE_TILES)
TILE_VALUES = [EMPTY_TILE] + PLACEABLE_TILES
NUM_TILE_VALUES = len(TILE_VALUES)
ENEMY_TILES = [ENEMY_1, ENEMY_2, ENEMY_3, ENEMY_4]
WALKABLE_CONTIGUOUS_TILES = [FLOOR_TILE, ENEMY_1, ENEMY_2, ENEMY_3, ENEMY_4] # Tiles considered for contiguity check

# Map Rendering Colors (using Rich styles)
TILE_COLORS = {
    EMPTY_TILE: "grey50",
    FLOOR_TILE: "white",
    ENEMY_1: "bright_red",
    ENEMY_2: "red",
    ENEMY_3: "dark_red",
    ENEMY_4: "magenta",
    DOOR: "green",
}

# Configuration (Moved to main block for clarity)
RENDER_EVERY_N_EPISODES = 50 # How often to print a map sample during training

# --- Helper Functions ---

def _render_map_to_console(map_tensor: torch.Tensor, title: str):
    """Helper function to print a single map tensor (H, W) with colors."""
    if map_tensor is None:
        console.print(f"--- {title}: Map is None ---")
        return
    console.print(f"--- {title} ---")
    # Ensure map is on CPU and is 2D
    if map_tensor.dim() != 2:
        console.print(f"[red]Error: Map tensor must be 2D, but got shape {map_tensor.shape}[/red]")
        return
    map_tensor_cpu = map_tensor.cpu().long() # Ensure it's on CPU and integer type
    height, width = map_tensor_cpu.shape

    for r in range(height):
        row_text = Text()
        for c in range(width):
            tile_val = int(map_tensor_cpu[r, c].item())
            color = TILE_COLORS.get(tile_val, "default")
            display_char = str(tile_val) # Use number representation
            row_text.append(display_char, style=color)
            row_text.append(" ") # Add space between characters
        console.print(row_text)

def _check_contiguity(map_tensor: torch.Tensor) -> Tuple[bool, int, int]:
    """
    Checks contiguity of WALKABLE_CONTIGUOUS_TILES (1-5).
    Uses Breadth-First Search (BFS).

    Returns:
        Tuple[bool, int, int]:
            - is_fully_contiguous (bool): True if all walkable tiles form one component.
            - num_major_components (int): Number of distinct contiguous components of walkable tiles.
            - largest_component_size (int): Size of the largest component.
    """
    height, width = map_tensor.shape
    visited = torch.zeros_like(map_tensor, dtype=torch.bool)
    component_sizes = []
    total_walkable_tiles = 0

    q = deque()

    for r in range(height):
        for c in range(width):
            tile = map_tensor[r, c].item()
            is_walkable = tile in WALKABLE_CONTIGUOUS_TILES
            if is_walkable:
                total_walkable_tiles += 1
                if not visited[r, c]:
                    # Start BFS for a new component
                    current_component_size = 0
                    q.append((r, c))
                    visited[r, c] = True
                    current_component_size += 1

                    while q:
                        row, col = q.popleft()

                        # Check neighbors (up, down, left, right)
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = row + dr, col + dc
                            # Check bounds
                            if 0 <= nr < height and 0 <= nc < width:
                                neighbor_tile = map_tensor[nr, nc].item()
                                # Check if walkable and not visited
                                if neighbor_tile in WALKABLE_CONTIGUOUS_TILES and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    q.append((nr, nc))
                                    current_component_size += 1
                    component_sizes.append(current_component_size)

    if not component_sizes: # No walkable tiles at all
        return True, 0, 0 # Considered contiguous (vacuously true)

    num_major_components = len(component_sizes)
    largest_component_size = max(component_sizes) if component_sizes else 0
    is_fully_contiguous = (num_major_components == 1) and (largest_component_size == total_walkable_tiles)

    return is_fully_contiguous, num_major_components, largest_component_size


# --- Gym Environment Definition ---
class MapGenEnv(gym.Env):
    """
    Gym environment for procedural map generation using PPO.

    Observation: Flattened map + hero vector.
    Actions: Discrete action representing placing tile V at coordinate (Y, X).
    Reward: Based on the provided level critic criteria.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    # Class variable to track episode count for periodic rendering
    episode_counter = 0

    def __init__(self, map_size: Tuple[int, int], hero_param_ranges: Dict[str, Any],
                 max_steps: int, render_mode: Optional[str] = None):
        super().__init__()

        self.map_width, self.map_height = map_size
        self.hero_param_ranges = hero_param_ranges
        self._parse_hero_ranges() # Pre-process hero ranges
        self.hero_tensor_size = len(self.hero_feature_order)

        self.max_steps = max_steps
        self.render_mode = render_mode # For Gym compatibility, though we use Rich

        # Internal state
        self.current_map: Optional[torch.Tensor] = None
        self.current_hero_vector: Optional[torch.Tensor] = None
        self.step_count = 0
        self.total_tiles = self.map_width * self.map_height
        self.last_map_rendered_episode = -1 # Track when map was last rendered

        # Observation Space: Flattened map (H*W) + hero vector (Size H) -> Box
        map_flat_size = self.map_height * self.map_width
        observation_size = map_flat_size + self.hero_tensor_size
        # Assuming tile values 0-6, normalize map features to [0, 1] for MLP
        # Hero features might need normalization depending on ranges, do basic scaling here
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(observation_size,), dtype=np.float32)

        # Action Space: Discrete action for placing tile V at (Y, X)
        # Action index `a` needs decoding:
        # tile_index = a % NUM_PLACEABLE_TILES -> gives index in PLACEABLE_TILES list
        # coord_index = a // NUM_PLACEABLE_TILES
        # y = coord_index // self.map_width
        # x = coord_index % self.map_width
        self.action_space = gym.spaces.Discrete(self.map_height * self.map_width * NUM_PLACEABLE_TILES)

    def _parse_hero_ranges(self):
        """ Pre-processes hero parameter ranges for easier sampling. """
        self.health_range = list(self.hero_param_ranges["health"])
        self.item1_range = [float(b) for b in self.hero_param_ranges["item1"]]
        self.item2_range = [float(b) for b in self.hero_param_ranges["item2"]]
        self.entry_map = {entry.value: i for i, entry in enumerate(Entry)} # "top"->0, etc.
        self.allowed_entry_enums = self.hero_param_ranges["entry"]
        self.entry_range_indices = [float(self.entry_map[e.value]) for e in self.allowed_entry_enums]
        self.rooms_left_range = list(self.hero_param_ranges["rooms_left"])

        # Define order for consistency
        self.hero_feature_order = ["health", "item1", "item2", "entry", "rooms_left"]
        self.hero_raw_ranges = { # For potential normalization later
            "health": (min(self.health_range), max(self.health_range)),
            "item1": (min(self.item1_range), max(self.item1_range)),
            "item2": (min(self.item2_range), max(self.item2_range)),
            "entry": (min(self.entry_range_indices), max(self.entry_range_indices)),
            "rooms_left": (min(self.rooms_left_range), max(self.rooms_left_range)),
        }


    def _generate_initial_map_random_walk(self, steps=25):
        """Generates a single initial map using random walk."""
        map_tensor = torch.zeros((self.map_height, self.map_width), dtype=torch.long)
        pos_x = random.randint(0, self.map_width - 1)
        pos_y = random.randint(0, self.map_height - 1)
        map_tensor[pos_y, pos_x] = FLOOR_TILE # Start with a floor tile

        for _ in range(steps):
            move = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)]) # N, S, E, W
            pos_y = max(0, min(self.map_height - 1, pos_y + move[0]))
            pos_x = max(0, min(self.map_width - 1, pos_x + move[1]))
            if map_tensor[pos_y, pos_x] == EMPTY_TILE: # Only place floor on empty tiles
                 map_tensor[pos_y, pos_x] = FLOOR_TILE
        return map_tensor

    def _generate_hero_vector(self) -> torch.Tensor:
        """ Generates a single random hero vector based on ranges. """
        hero_values = {
            "health": float(random.choice(self.health_range)),
            "item1": float(random.choice(self.item1_range)),
            "item2": float(random.choice(self.item2_range)),
            "entry": float(random.choice(self.entry_range_indices)),
            "rooms_left": float(random.choice(self.rooms_left_range)),
        }
        # Assemble in defined order
        hero_list = [hero_values[feat] for feat in self.hero_feature_order]
        return torch.tensor(hero_list, dtype=torch.float32)

    def _normalize_observation(self, map_tensor: torch.Tensor, hero_vector: torch.Tensor) -> np.ndarray:
        """ Normalizes map and hero vector for the observation space. """
        # Normalize map: Divide by max possible tile value (e.g., 6)
        map_flat = map_tensor.flatten().float() / (NUM_TILE_VALUES - 1)

        # Normalize hero vector: Simple min-max scaling to [0, 1]
        hero_normalized = hero_vector.clone()
        for i, feat in enumerate(self.hero_feature_order):
            min_val, max_val = self.hero_raw_ranges[feat]
            if max_val > min_val:
                hero_normalized[i] = (hero_vector[i] - min_val) / (max_val - min_val)
            else: # Handle cases where min == max (e.g., boolean flags)
                hero_normalized[i] = 0.0 if min_val == 0.0 else 0.5 # Or map to 0/1 directly

        # Concatenate and convert to numpy
        obs = torch.cat((map_flat, hero_normalized), dim=0)
        return obs.numpy()

    def _get_observation(self):
        """ Gets the current observation state. """
        if self.current_map is None or self.current_hero_vector is None:
            # Should not happen in normal flow after reset
            raise ValueError("Environment not properly reset.")
        return self._normalize_observation(self.current_map, self.current_hero_vector)

    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """Decodes a flat action index into (y, x, tile_value)."""
        tile_index = action % NUM_PLACEABLE_TILES
        tile_value = PLACEABLE_TILES[tile_index] # Get actual tile value (1-6)

        coord_index = action // NUM_PLACEABLE_TILES
        y = coord_index // self.map_width
        x = coord_index % self.map_width

        return y, x, tile_value

    def _count_tiles(self, map_tensor: torch.Tensor, tile_list: List[int]) -> int:
        """ Counts the number of tiles from the list present in the map. """
        count = 0
        for tile_val in tile_list:
            count += torch.sum(map_tensor == tile_val).item()
        return count

    def _calculate_reward(self) -> float:
        """ Calculates the reward based on the current map and hero vector. """
        if self.current_map is None or self.current_hero_vector is None:
            return 0.0 # Should not happen

        map_tensor = self.current_map # Use current map state
        hero_vector = self.current_hero_vector

        total_reward = 0.0

        # --- Reward Criteria ---
        height, width = self.map_height, self.map_width
        walkable_tiles_count = self._count_tiles(map_tensor, WALKABLE_CONTIGUOUS_TILES) # Count 1-5

        # 1. +2 reward if the whole map is contiguous (walkable tiles 1-5)
        is_contiguous, num_components, largest_comp_size = _check_contiguity(map_tensor)
        tiles_outside_main_area = 0
        if walkable_tiles_count > 0:
            if is_contiguous:
                total_reward += 2.0
            else:
                # 2. -0.5 reward for each tile [1-5] placed away from the primary contiguous area
                tiles_outside_main_area = walkable_tiles_count - largest_comp_size
                total_reward -= 0.5 * tiles_outside_main_area

                # 3. -3 reward for more than one primary contiguous area
                # Interpretation: Penalize if there's more than one distinct island of walkable tiles.
                if num_components > 1:
                    total_reward -= 3.0
        # If no walkable tiles, is_contiguous is true, num_components 0. No penalty/reward here.

        # Hero features (extract for readability, assuming order)
        hero_health = hero_vector[0].item()
        hero_entry_idx = int(hero_vector[3].item()) # 0:Top, 1:Right, 2:Bottom, 3:Left
        try:
             # Find the Entry enum member corresponding to the index
             hero_entry_direction = next(e for e, idx in self.entry_map.items() if idx == hero_entry_idx)
        except StopIteration:
             hero_entry_direction = None # Should not happen if generated correctly

        # 4. +1 reward for placing a door in the direction the hero is going to enter from
        door_locations = (map_tensor == DOOR).nonzero(as_tuple=False) # Get (y, x) pairs
        entry_door_reward_applied = False
        if hero_entry_direction:
            for loc in door_locations:
                y, x = loc[0].item(), loc[1].item()
                if hero_entry_direction == Entry.TOP.value and y == 0:
                    total_reward += 1.0; entry_door_reward_applied = True; break
                elif hero_entry_direction == Entry.BOTTOM.value and y == height - 1:
                    total_reward += 1.0; entry_door_reward_applied = True; break
                elif hero_entry_direction == Entry.LEFT.value and x == 0:
                    total_reward += 1.0; entry_door_reward_applied = True; break
                elif hero_entry_direction == Entry.RIGHT.value and x == width - 1:
                    total_reward += 1.0; entry_door_reward_applied = True; break
                # Note: This only rewards the *first* correctly placed door found.

        # 5. & 6. Enemy Penalties
        num_enemies = self._count_tiles(map_tensor, ENEMY_TILES)
        num_floor_tiles = self._count_tiles(map_tensor, [FLOOR_TILE]) # Ratio based on tile 1 (floor)

        if num_floor_tiles > 0:
            enemy_ratio = num_enemies / num_floor_tiles

            # 5. -0.5 reward for each extra enemy than desired (ratio < 0.25)
            desired_max_enemies_normal = math.floor(0.25 * num_floor_tiles)
            if num_enemies > desired_max_enemies_normal:
                extra_enemies = num_enemies - desired_max_enemies_normal
                total_reward -= 0.5 * extra_enemies

            # 6. -0.35 reward for each extra enemy given low health (ratio < 0.2)
            if hero_health < 3.0: # Using the exact value from the description
                desired_max_enemies_low_health = math.floor(0.2 * num_floor_tiles)
                # Calculate extra enemies *relative to the low health threshold*
                extra_enemies_low_health = max(0, num_enemies - desired_max_enemies_low_health)
                # Apply penalty only if the low health threshold is exceeded
                if extra_enemies_low_health > 0:
                     # This penalty seems cumulative with the previous one in the description.
                     # Let's apply it *per extra enemy* beyond the low health limit.
                     total_reward -= 0.35 * extra_enemies_low_health
                     # Alternative interpretation: apply -0.35 *only* if num_enemies > desired_max_low_health,
                     # regardless of how many extra? Let's stick to "per extra enemy".

        # 7. -2 reward for placing more than 4 doors, or inaccessible doors, or doors not on edge
        num_doors = self._count_tiles(map_tensor, [DOOR])
        door_penalty = 0
        if num_doors > 4:
            door_penalty = -2.0 # Penalty applies once if > 4 doors
        else:
            # Check each door's placement
            for loc in door_locations:
                y, x = loc[0].item(), loc[1].item()
                is_on_edge = (y == 0 or y == height - 1 or x == 0 or x == width - 1)
                if not is_on_edge:
                    door_penalty = -2.0 # Penalty applies once if any door is not on edge
                    break
                # Accessibility check is complex (requires pathfinding from door to walkable area)
                # Simple check: Is there a walkable neighbor?
                has_walkable_neighbor = False
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = y + dr, x + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if map_tensor[nr, nc].item() in WALKABLE_CONTIGUOUS_TILES:
                            has_walkable_neighbor = True
                            break
                if not has_walkable_neighbor:
                     # Commenting out inaccessible penalty for now - complex & potentially noisy
                     # door_penalty = -2.0
                     # break
                     pass # Ignoring inaccessible doors for simplicity based on reward text

        total_reward += door_penalty

        # 8. +0.2 reward for each traversable tile placed
        # Interpretation: Reward for floor and enemy tiles (1-5).
        total_reward += 0.2 * walkable_tiles_count

        # --- Small penalty for empty maps to encourage generation ---
        if walkable_tiles_count == 0 and num_doors == 0:
             total_reward -= 1.0 # Discourage completely empty maps

        return total_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Gym API recommendation

        self.current_map = self._generate_initial_map_random_walk()
        self.current_hero_vector = self._generate_hero_vector()
        self.step_count = 0

        # --- Visualization Trigger ---
        MapGenEnv.episode_counter += 1
        if self.render_mode == 'human' and MapGenEnv.episode_counter % RENDER_EVERY_N_EPISODES == 0:
            if MapGenEnv.episode_counter != self.last_map_rendered_episode:
                 self.render()
                 self.last_map_rendered_episode = MapGenEnv.episode_counter # Avoid double render if reset called mid-episode

        observation = self._get_observation()
        info = {} # No extra info needed for standard PPO

        return observation, info

    def step(self, action):
        if self.current_map is None or self.current_hero_vector is None:
            raise ValueError("Environment not properly reset before step.")

        # 1. Decode and Apply Action
        y, x, tile_value = self._decode_action(action)
        self.current_map[y, x] = tile_value # Apply the change

        # 2. Calculate Reward
        reward = self._calculate_reward()

        # 3. Update State
        self.step_count += 1

        # 4. Check Done Condition
        done = self.step_count >= self.max_steps

        # 5. Get Next Observation
        observation = self._get_observation()

        # 6. Info dictionary (optional)
        info = {}

        # Gym API v26 returns 5 values: obs, reward, terminated, truncated, info
        terminated = done # Episode ended due to reaching goal or condition (here, only max steps)
        truncated = False # Episode ended due to external limit (e.g. time limit, not used here directly)
        if self.step_count >= self.max_steps:
             truncated = True # If done is *only* due to max_steps, it's truncation
             terminated = False # If truncated, cannot be terminated

        # Ensure reward is float
        reward = float(reward)

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
             title = f"Episode {MapGenEnv.episode_counter}, Step {self.step_count}"
             _render_map_to_console(self.current_map, title)
        else:
             # For other modes like 'rgb_array', return np array (not implemented here)
             pass

    def close(self):
        # Clean up any resources if needed
        pass

# --- Main Execution Block ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=10, help='Map width')
    parser.add_argument('--height', type=int, default=8, help='Map height')
    parser.add_argument('--max_steps', type=int, default=50, help='Max steps per episode (map modifications)')
    parser.add_argument('--hid', type=int, default=64, help='Number of hidden units in policy/value networks')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers in policy/value networks')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--cpu', type=int, default=1, help='Number of parallel CPUs to use (requires MPI)')
    parser.add_argument('--steps', type=int, default=4000, help='Steps per epoch')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--exp_name', type=str, default='ppo_mapgen', help='Experiment name for logging')
    parser.add_argument('--render_freq', type=int, default=10, help='Render map every N episodes')
    args = parser.parse_args()

    # --- Configure MPI if using multiple CPUs ---
    mpi_fork(args.cpu)
    setup_pytorch_for_mpi()

    # --- Set Random Seeds ---
    seed = args.seed + 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) # Seed python's random module too

    # --- Experiment Setup / Logging ---
    logger_kwargs = dict(output_dir=f'data/{args.exp_name}_s{args.seed}', exp_name=args.exp_name)

    # --- Environment Setup ---
    MAP_SIZE = (args.width, args.height)
    # Define Hero Parameter Ranges (as in the original MDPTrainer example)
    HERO_PARAM_RANGES = {
        "health": range(50, 151, 10), # Example: 50, 60, ..., 150
        "item1": [0.0, 1.0],          # Boolean flags as floats
        "item2": [0.0, 1.0],
        "entry": [Entry.TOP, Entry.BOTTOM, Entry.LEFT, Entry.RIGHT], # Use Enum members
        "rooms_left": range(1, 6),    # Example: 1, 2, 3, 4, 5
    }

    # Update global render frequency
    RENDER_EVERY_N_EPISODES = args.render_freq

    # Environment Factory for Spinup
    env_fn = lambda: MapGenEnv(map_size=MAP_SIZE,
                               hero_param_ranges=HERO_PARAM_RANGES,
                               max_steps=args.max_steps,
                               render_mode='human' if proc_id() == 0 else None) # Only render on proc 0

    console.print(f"[bold cyan]Starting PPO Training for Map Generation[/bold cyan]")
    console.print(f"Map Size: {args.width}x{args.height}")
    console.print(f"Max Steps per Episode: {args.max_steps}")
    console.print(f"Epochs: {args.epochs}")
    console.print(f"Steps per Epoch: {args.steps}")
    console.print(f"CPUs: {args.cpu}")
    console.print(f"Seed: {args.seed} (Proc 0 base)")
    console.print(f"Output Dir: {logger_kwargs['output_dir']}")
    console.print(f"Rendering every {RENDER_EVERY_N_EPISODES} episodes on Proc 0.")

    # --- Run PPO Training ---
    ppo_kwargs = dict(
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.layers),
        gamma=args.gamma,
        seed=seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        # --- PPO Specific Hyperparameters (Defaults from Spinup, tune as needed) ---
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=args.max_steps, # Ensure consistent with env
        target_kl=0.01,
        save_freq=10 # How often to save model checkpoints (epochs)
    )

    spinup.ppo(env_fn=env_fn, **ppo_kwargs)

    console.print(f"[bold green]Training finished.[/bold green]")
    console.print(f"Results saved in: {logger_kwargs['output_dir']}")
