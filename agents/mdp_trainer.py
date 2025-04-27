import os
import json
import math
import random
import logging
import itertools  # Keep just in case, though maybe not strictly needed now
from enum import Enum
from typing import Tuple, Callable, List, Optional, Dict, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm  # Used in MDPAgent step loop

# Rich imports for logging and progress bars
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from daedalus.models.mdp import MDPAgent

# --- Setup Logging ---
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")
console = Console()


# --- Enums and Constants ---
class Entry(Enum):
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"


# Tile Modification Values/Actions
NO_ACTION = 0
SET_EMPTY = 1
PLACE_ENEMY_1 = 2
PLACE_ENEMY_2 = 3
PLACE_ENEMY_3 = 4
PLACE_ENEMY_4 = 5
PLACE_DOOR = 6
NUM_TILE_VALUES = 7  # 0 through 6

# Turtle Movement Actions
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
NUM_TURTLE_MOVES = 4

# --- MDPAgent Class ---


# --- MDPTrainer Class ---
class MDPTrainer:
    """
    Manages the training process for the MDPAgent over multiple episodes,
    tracking and saving detailed metrics including policy entropy.
    Generates initial maps using random walks.
    Visualizes average reward per map template alongside an example modified map.
    Uses Rich for progress bars.
    """

    def __init__(
        self,
        agent: MDPAgent,
        map_size: Tuple[int, int],  # (width, height)
        hero_param_ranges: Dict[str, Any],
        num_episodes: int,
        max_steps_per_episode: int,
        metrics_csv_path: str = "training_metrics.csv",
        metric_update_steps: int = 10,
        agent_save_path: str = "mdpO_agent_state.json",
        # --- Random Walk Parameters ---
        num_initial_maps: int = 16,  # Default: Number of base map templates to generate
        random_walk_steps: int = 25,  # Default: Steps for random walk generation
        # --- Hero Parameters ---
        heroes_per_map: int = 20,  # Number of hero variations per map template
    ):
        self.agent = agent
        self.map_size = map_size  # (width, height)
        self.map_width, self.map_height = map_size
        self.hero_param_ranges = hero_param_ranges
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.metrics_csv_path = metrics_csv_path
        self.metric_update_steps = metric_update_steps
        self.agent_save_path = agent_save_path
        # Store map generation parameters
        self.num_initial_maps = num_initial_maps
        self.random_walk_steps = random_walk_steps
        self.heroes_per_map = heroes_per_map

        # Stores dict for each episode: e.g., {'episode': ..., 'avg_cumulative_reward': ..., 'policy_entropy': ...}
        self.metrics: List[Dict[str, Any]] = []
        self.random_hero_tensors: Optional[torch.Tensor] = (
            None  # Shape (N_batch, hero_size), CPU
        )
        self.hero_combinations: List[Tuple] = []  # List of tuples, CPU
        self.batch_size_N: int = 0  # Will be num_initial_maps * heroes_per_map
        self.original_maps: Optional[torch.Tensor] = (
            None  # Shape (num_initial_maps, H, W), CPU
        )
        self.map_indices_for_batch: Optional[List[int]] = (
            None  # Len N_batch, maps item i to original map index, CPU
        )

        self.tile_colors = {
            0: "grey50",
            1: "white",
            2: "bright_red",
            3: "red",
            4: "dark_red",
            5: "magenta",
            6: "green",
        }
        self.device = agent.device  # Use the same device as the agent

    def _generate_initial_maps_random_walk(self):
        """
        Generates initial maps using a batch random walk process.
        Each walker starts at a random location, moves in one of the 4 cardinal
        directions at each step, and sets the tile it lands on to 1.
        Stores results in self.original_maps on CPU.
        """
        width, height = self.map_width, self.map_height
        log.info(
            f"Generating {self.num_initial_maps} initial maps ({width}x{height}) "
            f"using {self.random_walk_steps}-step random walks (placing tile 1)..."
        )

        # Use agent's device for generation, then move to CPU
        gen_device = self.device

        # Action space: 4 cardinal moves
        MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT = 0, 1, 2, 3
        NUM_ACTIONS = 4  # Only 4 move actions needed

        # Initialize maps (batch) - use height, width order for tensor dims (N, H, W)
        maps = torch.zeros(
            (self.num_initial_maps, height, width), dtype=torch.long, device=gen_device
        )

        # Initialize walker positions (batch) - (x, y) coords
        pos_x = torch.randint(0, width, (self.num_initial_maps,), device=gen_device)
        pos_y = torch.randint(0, height, (self.num_initial_maps,), device=gen_device)

        batch_indices = torch.arange(self.num_initial_maps, device=gen_device)

        # --- Mark the starting position as 1 ---
        # Use advanced indexing (use H, W order -> [index, y, x])
        maps[batch_indices, pos_y, pos_x] = 1

        # --- Perform the random walk steps ---
        for step in range(self.random_walk_steps):
            # Choose random move actions for the batch (0, 1, 2, or 3)
            actions = torch.randint(
                0, NUM_ACTIONS, (self.num_initial_maps,), device=gen_device
            )

            # --- Calculate potential new positions based on actions ---
            # Use temporary variables or direct update with torch.where
            next_pos_y = pos_y.clone()
            next_pos_x = pos_x.clone()

            next_pos_y = torch.where(actions == MOVE_UP, next_pos_y - 1, next_pos_y)
            next_pos_y = torch.where(actions == MOVE_DOWN, next_pos_y + 1, next_pos_y)
            next_pos_x = torch.where(actions == MOVE_LEFT, next_pos_x - 1, next_pos_x)
            next_pos_x = torch.where(actions == MOVE_RIGHT, next_pos_x + 1, next_pos_x)

            # --- Clamp positions to stay within map boundaries ---
            pos_x = torch.clamp(next_pos_x, 0, width - 1)
            pos_y = torch.clamp(next_pos_y, 0, height - 1)

            # --- Place tile 1 at the new position ---
            # The walker has moved, now mark the tile it landed on
            maps[batch_indices, pos_y, pos_x] = 1

        log.info(f"Generated {maps.shape[0]} maps via random walk.")
        self.original_maps = maps.cpu()  # Store final maps on CPU
        return self.original_maps

    def _generate_random_hero_tensors(self):
        """
        Generate random hero tensors for each generated map template.
        Stores results in self.random_hero_tensors (Tensor, CPU) and self.map_indices_for_batch (List, CPU).
        Sets self.batch_size_N.
        """
        log.info(
            f"Generating {self.heroes_per_map} random hero tensors for each of the {self.num_initial_maps} map templates..."
        )

        if self.original_maps is None:
            raise ValueError("Maps must be generated before creating hero tensors.")

        num_maps = self.original_maps.shape[0]
        if num_maps == 0:
            log.warning("No original maps generated, cannot create hero tensors.")
            self.batch_size_N = 0
            self.random_hero_tensors = torch.empty(
                (0, self.agent.hero_tensor_size), dtype=torch.float
            )
            self.map_indices_for_batch = []
            self.hero_combinations = []
            return self.random_hero_tensors, self.hero_combinations

        total_heroes_needed = num_maps * self.heroes_per_map

        try:
            # Prepare ranges (Ensure Entry enum is available)
            health_range = list(self.hero_param_ranges["health"])
            item1_range = [float(b) for b in self.hero_param_ranges["item1"]]
            item2_range = [float(b) for b in self.hero_param_ranges["item2"]]
            try:
                # Map Entry enum members ("top", "right", etc.) to numeric indices (0, 1, etc.)
                entry_map = {
                    entry.value: i for i, entry in enumerate(Entry)
                }  # "top"->0, "right"->1...
                # Get the list of allowed entry values (e.g., [Entry.TOP, Entry.LEFT]) from params
                allowed_entry_enums = self.hero_param_ranges["entry"]
                # Convert these allowed enums to their corresponding numeric indices
                entry_range = [float(entry_map[e.value]) for e in allowed_entry_enums]
            except NameError:
                log.warning(
                    "Entry Enum not found or not used in hero_param_ranges. Assuming 'entry' provides numeric values directly."
                )
                entry_range = [
                    float(e) for e in self.hero_param_ranges["entry"]
                ]  # Assume numeric if Enum fails
            except KeyError as e:
                log.error(
                    f"Invalid Entry value name provided in hero_param_ranges['entry']: {e}. Expected values like 'top', 'bottom', etc."
                )
                raise
            except AttributeError:
                log.error(
                    "hero_param_ranges['entry'] should contain Enum members (like Entry.TOP), not strings, if using Enum."
                )
                raise

            rooms_left_range = list(self.hero_param_ranges["rooms_left"])

            # Generate random hero combinations and track map indices (all on CPU)
            self.hero_combinations = []
            self.map_indices_for_batch = []  # Track which map each hero corresponds to
            for map_idx in range(num_maps):
                for _ in range(self.heroes_per_map):
                    hero_combo = (
                        random.choice(health_range),
                        random.choice(item1_range),
                        random.choice(item2_range),
                        random.choice(entry_range),
                        random.choice(rooms_left_range),
                    )
                    self.hero_combinations.append(hero_combo)
                    self.map_indices_for_batch.append(
                        map_idx
                    )  # Store corresponding map index

            self.random_hero_tensors = torch.tensor(
                self.hero_combinations, dtype=torch.float
            )  # Stays on CPU for now
            self.batch_size_N = len(self.hero_combinations)  # Set actual batch size

            log.info(
                f"Generated {self.batch_size_N} random hero tensors (Total training batch size)"
            )
            log.info(f"Hero batch tensor shape: {self.random_hero_tensors.shape}")
            log.info(f"Map indices length: {len(self.map_indices_for_batch)}")

            # Validate tensor size against agent expectation
            if self.random_hero_tensors.shape[1] != self.agent.hero_tensor_size:
                log.error(
                    f"Generated hero tensor size ({self.random_hero_tensors.shape[1]}) "
                    f"does not match agent's expected size ({self.agent.hero_tensor_size}). Check hero_param_ranges."
                )
                raise ValueError("Hero tensor size mismatch.")

            return self.random_hero_tensors, self.hero_combinations

        except Exception as e:
            log.error(f"Error generating random hero tensors: {e}")
            # console.print_exception(show_locals=True) # Uncomment for debugging
            raise

    def _save_metrics(self):
        """Saves collected metrics (incl. policy entropy) to CSV."""
        if not self.metrics:
            log.warning("No metrics collected yet, skipping CSV save.")
            return
        # Define the desired order of columns
        column_order = [
            "episode",
            "final_epsilon",
            "avg_cumulative_reward",
            "policy_entropy",
            "avg_steps_to_threshold",
            "n_finished",
            "n_total",
        ]
        df = pd.DataFrame(self.metrics)
        # Ensure all columns exist, adding NaN for missing ones, and reorder
        df = df.reindex(columns=column_order)
        try:
            output_dir = os.path.dirname(self.metrics_csv_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            df.to_csv(self.metrics_csv_path, index=False, float_format="%.4f")
            log.info(f"Metrics saved to {self.metrics_csv_path}")
        except IOError as e:
            log.error(f"Error saving metrics to {self.metrics_csv_path}: {e}")
        except Exception as e:
            log.error(f"An unexpected error occurred while saving metrics: {e}")

    def _render_map_to_console(self, map_tensor: torch.Tensor, title: str):
        """Helper function to print a single map tensor (H, W) with colors."""
        console.print(f"--- {title} ---")
        # Ensure map is on CPU before iterating
        map_tensor_cpu = map_tensor.cpu()
        height, width = map_tensor_cpu.shape  # Expects (H, W)
        for r in range(height):
            row_text = Text()
            for c in range(width):
                tile_val = int(map_tensor_cpu[r, c].item())
                color = self.tile_colors.get(tile_val, "default")
                display_char = str(tile_val)
                row_text.append(display_char, style=color)
                row_text.append(" ")  # Add space between characters
            console.print(row_text)

    def visualize_maps(
        self,
        original_maps: Optional[
            torch.Tensor
        ],  # The initial set of generated maps (CPU, N_orig, H, W)
        final_maps: Optional[
            torch.Tensor
        ],  # The maps after the last training episode (CPU, N_batch, H, W)
        map_indices_for_batch: Optional[
            List[int]
        ],  # Maps batch item index -> original map index (CPU)
        map_avg_rewards: Dict[
            int, float
        ],  # Original Map Index -> Avg Reward from last episode (CPU data)
        num_to_show: int = 5,
    ):
        """
        Visualizes a random sample of original map templates, their avg reward,
        and one corresponding example of a modified map from the final batch.
        Assumes input tensors (original_maps, final_maps) are on CPU.
        """
        if original_maps is None or original_maps.shape[0] == 0:
            log.warning("No original maps available for visualization.")
            return

        if map_indices_for_batch is None:
            log.warning(
                "Map indices for batch not available, cannot link modified maps."
            )
            final_maps = None  # Cannot link, so don't show modified examples

        num_maps_total = original_maps.shape[0]
        num_to_show = min(num_to_show, num_maps_total)
        if num_to_show <= 0:
            log.warning("Number of maps to show is zero or less.")
            return

        log.info(
            f"Visualizing {num_to_show} random map templates (out of {num_maps_total}) with average rewards and final example..."
        )

        # Select random *indices* of the original maps
        indices_to_show = random.sample(range(num_maps_total), k=num_to_show)

        for i, original_map_idx in enumerate(indices_to_show):
            original_map_tensor = original_maps[
                original_map_idx
            ]  # (H, W) tensor on CPU
            avg_reward = map_avg_rewards.get(
                original_map_idx, float("nan")
            )  # Get reward, default to NaN

            console.print(
                f"\n=== Map Template {i+1}/{num_to_show} (Original Index: {original_map_idx}) | Avg Reward: {avg_reward:.3f} ==="
            )

            # --- Print Original Map ---
            self._render_map_to_console(
                original_map_tensor, f"Original Map (Index {original_map_idx})"
            )

            # --- Find and Print Corresponding Modified Map ---
            modified_map_tensor = None
            chosen_batch_idx_str = "N/A"
            if final_maps is not None and map_indices_for_batch is not None:
                if final_maps.shape[0] != len(map_indices_for_batch):
                    log.warning(
                        f"Mismatch between final_maps count ({final_maps.shape[0]}) and map_indices length ({len(map_indices_for_batch)}). Cannot reliably link modified maps."
                    )
                else:
                    # Find batch indices that correspond to this original map index
                    corresponding_batch_indices = [
                        b_idx
                        for b_idx, m_idx in enumerate(map_indices_for_batch)
                        if m_idx == original_map_idx
                    ]

                    if corresponding_batch_indices:
                        # Randomly select one batch item to show the final state for
                        chosen_batch_idx = random.choice(corresponding_batch_indices)
                        chosen_batch_idx_str = str(chosen_batch_idx)
                        if 0 <= chosen_batch_idx < final_maps.shape[0]:
                            modified_map_tensor = final_maps[
                                chosen_batch_idx
                            ]  # (H, W) tensor on CPU
                        else:
                            log.warning(
                                f"Chosen batch index {chosen_batch_idx} out of bounds for final maps."
                            )
                    # else: # No corresponding item found (less likely if generation is correct)
                    #    log.debug(f"No batch items found corresponding to original map index {original_map_idx}.")

            if modified_map_tensor is not None:
                self._render_map_to_console(
                    modified_map_tensor,
                    f"Example Modified Map (from Batch Item {chosen_batch_idx_str})",
                )
            else:
                console.print(
                    f"--- Example Modified Map: Not Available (Batch Item: {chosen_batch_idx_str}) ---"
                )

    def train(self):
        """Runs the main training loop using Rich progress."""
        log.info(f"--- Starting Training ---")
        log.info(f"Agent Strategy: {self.agent.strategy}")
        log.info(f"Map Size: {self.map_width}x{self.map_height}")
        log.info(f"Number of Episodes: {self.num_episodes}")
        log.info(f"Max Steps per Episode: {self.max_steps_per_episode}")
        log.info(
            f"Modification Threshold: {self.agent.percentage_change_threshold:.1%}"
        )
        log.info(f"Target Modifications: {self.agent.max_modifications} tiles")
        log.info(f"Initial Epsilon: {self.agent.epsilon:.3f}")
        log.info(f"Epsilon Decay: {self.agent.epsilon_decay}")
        log.info(f"Min Epsilon: {self.agent.min_epsilon}")
        log.info(f"Device: {self.device}")

        # 1. Generate initial maps using random walk
        self._generate_initial_maps_random_walk()
        if self.original_maps is None or self.original_maps.shape[0] == 0:
            log.error("Failed to generate initial maps. Aborting training.")
            return

        # 2. Generate random hero tensors based on generated maps
        # This sets self.batch_size_N, self.random_hero_tensors, self.map_indices_for_batch
        try:
            self._generate_random_hero_tensors()
        except (
            ValueError
        ) as e:  # Catch potential size mismatch or other generation errors
            log.error(f"Error during hero tensor generation: {e}. Aborting training.")
            return

        if self.batch_size_N == 0:
            log.error("Generated hero batch size is 0. Cannot train.")
            return

        num_map_templates = self.original_maps.shape[0]
        log.info(f"Using {num_map_templates} generated map templates.")
        log.info(
            f"Total training batch size (Map Templates * Heroes per Map): {self.batch_size_N}"
        )

        # 3. Prepare the initial map batch for the first episode
        # Create a large batch where each item corresponds to a hero tensor,
        # populated with the correct original map template. Stays on CPU for now.
        initial_batch_maps_cpu = torch.zeros(
            (self.batch_size_N, self.map_height, self.map_width), dtype=torch.long
        )
        try:
            for batch_idx, map_idx in enumerate(self.map_indices_for_batch):
                if 0 <= map_idx < num_map_templates:
                    # Clone from the stored original maps (which are on CPU)
                    initial_batch_maps_cpu[batch_idx] = self.original_maps[
                        map_idx
                    ].clone()
                else:
                    log.error(
                        f"Invalid map index {map_idx} encountered for batch item {batch_idx}. Check generation logic."
                    )
                    return  # Stop if indices are wrong
        except IndexError:
            log.error(
                f"IndexError during initial batch map creation. Batch size={self.batch_size_N}, map_indices len={len(self.map_indices_for_batch)}, num_templates={num_map_templates}"
            )
            return

        # --- Training Loop Setup ---
        last_episode_map_rewards: Dict[int, float] = (
            {}
        )  # Store last rewards for visualization
        last_episode_final_maps: Optional[torch.Tensor] = (
            None  # Store last modified maps (on CPU)
        )

        # Setup Rich Progress Bar for Episodes
        progress = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[metrics]}"),  # Field for custom metrics text
            console=console,
            transient=False,  # Keep the bar visible after completion
        )

        # --- Start Training Loop ---
        try:
            with progress:
                task_id = progress.add_task(
                    "[bold green]Training Episodes",
                    total=self.num_episodes,
                    metrics="Starting...",
                )

                for episode_idx in range(self.num_episodes):
                    # --- Run one episode ---
                    # Prepare inputs for the agent (move to agent's device)
                    # Clone maps/heroes to ensure each episode starts fresh from the originals
                    episode_initial_maps_device = initial_batch_maps_cpu.clone().to(
                        self.device
                    )
                    episode_initial_heroes_device = self.random_hero_tensors.clone().to(
                        self.device
                    )  # Heroes are float

                    episode_result = self.agent.train_episode(
                        episode_initial_maps_device,
                        episode_initial_heroes_device,
                        self.max_steps_per_episode,
                    )

                    # Store the final maps from this episode (agent returns them on CPU)
                    last_episode_final_maps = episode_result.get(
                        "final_maps"
                    )  # Shape (N_batch, H, W), CPU
                    episode_metrics = episode_result.get("metrics", {})
                    final_rewards_tensor = episode_result.get(
                        "final_cumulative_rewards"
                    )  # Shape (N_batch,), CPU

                    # --- Calculate average reward per original map template for this episode ---
                    map_rewards_agg = defaultdict(list)
                    current_map_avg_rewards = {}
                    if (
                        final_rewards_tensor is not None
                        and self.map_indices_for_batch is not None
                        and len(final_rewards_tensor) == self.batch_size_N
                    ):
                        for i in range(self.batch_size_N):
                            map_idx = self.map_indices_for_batch[i]
                            reward = final_rewards_tensor[
                                i
                            ].item()  # .item() gets Python number from 0-dim tensor
                            map_rewards_agg[map_idx].append(reward)

                        for map_idx, rewards_list in map_rewards_agg.items():
                            current_map_avg_rewards[map_idx] = (
                                sum(rewards_list) / len(rewards_list)
                                if rewards_list
                                else float("nan")
                            )
                        last_episode_map_rewards = current_map_avg_rewards  # Store for visualization after loop
                    else:
                        log.warning(
                            f"Episode {episode_idx+1}: Could not calculate per-map rewards (final rewards tensor missing, mismatched size, or map indices missing)."
                        )
                        last_episode_map_rewards = {}

                    # --- Collect Metrics for Saving ---
                    full_metrics = {
                        "episode": episode_idx + 1,
                        "final_epsilon": round(
                            episode_metrics.get("final_epsilon", self.agent.epsilon), 5
                        ),
                        "avg_cumulative_reward": episode_metrics.get(
                            "avg_cumulative_reward", float("nan")
                        ),
                        "policy_entropy": episode_metrics.get(
                            "policy_entropy", float("nan")
                        ),
                        "avg_steps_to_threshold": episode_metrics.get(
                            "avg_steps_to_threshold", float("nan")
                        ),
                        "n_finished": episode_metrics.get("n_finished", 0),
                        "n_total": episode_metrics.get("n_total", self.batch_size_N),
                    }
                    self.metrics.append(full_metrics)

                    # --- Prepare metrics string for Rich progress bar ---
                    metrics_str_parts = [
                        f"Ep: {episode_idx+1}/{self.num_episodes}",
                        f"ε: {full_metrics['final_epsilon']:.3f}",
                        f"AvgEpRwd: {full_metrics['avg_cumulative_reward']:.2f}",
                        f"Entropy: {full_metrics['policy_entropy']:.3f}",
                    ]
                    if not np.isnan(full_metrics["avg_steps_to_threshold"]):
                        metrics_str_parts.append(
                            f"Steps: {full_metrics['avg_steps_to_threshold']:.1f}"
                        )
                    else:
                        metrics_str_parts.append("Steps: N/A")
                    metrics_str_parts.append(
                        f"Done: {full_metrics['n_finished']}/{full_metrics['n_total']}"
                    )
                    metrics_display = " | ".join(metrics_str_parts)

                    # Update Rich progress bar
                    progress.update(task_id, advance=1, metrics=metrics_display)

                    # Save metrics CSV periodically or at the end
                    if (episode_idx + 1) % self.metric_update_steps == 0 or (
                        episode_idx + 1
                    ) == self.num_episodes:
                        self._save_metrics()
                    try:
                        self.visualize_maps(
                            self.original_maps,  # Generated original maps (CPU)
                            last_episode_final_maps,  # Modified maps from the last episode (CPU)
                            self.map_indices_for_batch,  # Linking list (CPU)
                            last_episode_map_rewards,  # Dict of avg rewards (CPU data)
                            num_to_show=5,  # Or make this configurable
                        )
                    except Exception as e:
                        log.error(f"Error visualizing maps after training:")
                        console.print_exception(show_locals=True)

            # --- End of Training Loop ---
            log.info("Training loop completed.")

        except Exception as e:
            log.error("An error occurred during the training loop:")
            console.print_exception(show_locals=True)  # Show traceback
            # Attempt to save whatever metrics were collected
            self._save_metrics()
            # Optionally save agent state even on error?
            # self.agent.save_state(self.agent_save_path + "_error")

        finally:
            # --- Final Visualization (runs even if loop errors out, if data is available) ---
            log.info("--- Post-Training Visualization ---")
            try:
                self.visualize_maps(
                    self.original_maps,  # Generated original maps (CPU)
                    last_episode_final_maps,  # Modified maps from the last episode (CPU)
                    self.map_indices_for_batch,  # Linking list (CPU)
                    last_episode_map_rewards,  # Dict of avg rewards (CPU data)
                    num_to_show=5,  # Or make this configurable
                )
            except Exception as e:
                log.error(f"Error visualizing maps after training:")
                console.print_exception(show_locals=True)

            # --- Save Agent State ---
            self.agent.save_state(self.agent_save_path)
            log.info("--- Training finished ---")


# --- Example Usage ---
if __name__ == "__main__":
    log.info("--- Starting MDP Training Script ---")

    # --- Configuration ---
    MAP_WIDTH = 10
    MAP_HEIGHT = 8
    MAP_SIZE = (MAP_WIDTH, MAP_HEIGHT)
    HERO_TENSOR_SIZE = 5  # health, item1, item2, entry, rooms_left
    PERCENTAGE_CHANGE = 0.15  # Target % of tiles to modify
    STRATEGY = "wide"  # Choose "narrow", "turtle", or "wide"
    INITIAL_EPSILON = 1.0
    EPSILON_DECAY = 0.997
    MIN_EPSILON = 0.05
    WIDE_K_SAMPLES = 128  # Samples for wide strategy greedy action

    NUM_EPISODES = 100
    MAX_STEPS_PER_EPISODE = 200
    METRIC_UPDATE_STEPS = 20  # Save CSV every N episodes

    # Random Walk Map Generation Params (using defaults: 64 maps, 25 steps)
    NUM_INITIAL_MAP_TEMPLATES = 64  # Trainer default = 64
    RANDOM_WALK_GENERATION_STEPS = 25  # Trainer default = 25

    # Hero Params
    HEROES_PER_MAP_TEMPLATE = 10  # Trainer default = 20
    HERO_PARAM_RANGES = {
        "health": range(50, 151, 10),  # Example: 50, 60, ..., 150
        "item1": [0.0, 1.0],  # Boolean flags as floats
        "item2": [0.0, 1.0],
        "entry": [Entry.TOP, Entry.BOTTOM, Entry.LEFT, Entry.RIGHT],  # Use Enum members
        "rooms_left": range(1, 6),  # Example: 1, 2, 3, 4, 5
    }

    # --- Dummy Critic Function ---
    # Replace this with your actual trained critic model/function
    def dummy_critic(map_batch: torch.Tensor, hero_batch: torch.Tensor) -> torch.Tensor:
        """
        A placeholder critic. Returns a random reward for each map in the batch.
        Input tensors are expected on the agent's device.
        Output tensor should be on the same device.
        """
        n_batch = map_batch.size(0)
        # Example: reward slightly higher for more non-empty tiles
        # Ensure calculation stays on the same device as input
        non_empty_tiles = (map_batch > 0).view(n_batch, -1).sum(dim=1).float()
        base_reward = torch.rand(n_batch, device=map_batch.device) * 0.5
        bonus = (non_empty_tiles / (map_batch.shape[1] * map_batch.shape[2])) * 0.5
        return base_reward + bonus

    # --- Setup Agent ---
    log.info("Initializing MDPAgent...")
    mdp_agent = MDPAgent(
        size=MAP_SIZE,
        hero_tensor_size=HERO_TENSOR_SIZE,
        critic=dummy_critic,  # Use the dummy critic
        strategy=STRATEGY,
        percentage_change=PERCENTAGE_CHANGE,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=MIN_EPSILON,
        wide_greedy_sample_k=WIDE_K_SAMPLES,
    )
    # Optional: Load previous agent state (epsilon)
    # mdp_agent.load_state("mdpO_agent_state.json")

    # --- Setup Trainer ---
    log.info("Initializing MDPTrainer...")
    trainer = MDPTrainer(
        agent=mdp_agent,
        map_size=MAP_SIZE,
        hero_param_ranges=HERO_PARAM_RANGES,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        metrics_csv_path="training_metrics_output.csv",
        metric_update_steps=METRIC_UPDATE_STEPS,
        agent_save_path="mdpO_agent_state_output.json",
        # Pass map generation parameters explicitly if not using defaults:
        num_initial_maps=NUM_INITIAL_MAP_TEMPLATES,
        random_walk_steps=RANDOM_WALK_GENERATION_STEPS,
        # Pass hero parameter:
        heroes_per_map=HEROES_PER_MAP_TEMPLATE,
    )

    # --- Run Training ---
    trainer.train()

    log.info("--- MDP Training Script Finished ---")
