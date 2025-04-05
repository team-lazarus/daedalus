import os
import json
import math
import random
import logging
import itertools
from enum import Enum
from typing import Tuple, Callable, List, Optional, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
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

from daedalus.models.mdp import MDPAgent, Entry

# --- Setup Logging ---
# Use RichHandler for pretty console logging
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
# Get the logger instance
log = logging.getLogger("rich")
# Create a console object for direct printing if needed
console = Console()


# --- Updated Trainer Class ---
class MDPTrainer:
    """
    Manages the training process for the MDPAgent over multiple episodes,
    tracking and saving detailed metrics. Uses Rich for progress bars.
    """

    def __init__(
        self,
        agent: MDPAgent,
        map_size: Tuple[int, int],
        hero_param_ranges: Dict[str, Any],
        num_episodes: int,
        max_steps_per_episode: int,
        metrics_csv_path: str = "training_metrics.csv",
        metric_update_steps: int = 10,  # How often to save CSV
        agent_save_path: str = "mdpO_agent_state.json",
    ):
        self.agent = agent
        self.map_size = map_size
        self.hero_param_ranges = hero_param_ranges
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.metrics_csv_path = metrics_csv_path
        self.metric_update_steps = metric_update_steps
        self.agent_save_path = agent_save_path

        # Stores dict for each episode: {'episode': ..., 'final_epsilon': ..., 'avg_reward': ..., ...}
        self.metrics: List[Dict[str, Any]] = []
        self.all_hero_combinations: List[Tuple] = []
        self.initial_hero_batch: Optional[torch.Tensor] = None
        self.batch_size_N: int = 0

        # Define tile color mapping for visualization
        self.tile_colors = {
            0: "grey50",  # Explicit gray for 0 (assuming it means 'unmodified' or 'empty space')
            1: "white",  # SET_EMPTY
            2: "bright_red",  # PLACE_ENEMY_1
            3: "red",  # PLACE_ENEMY_2
            4: "dark_red",  # PLACE_ENEMY_3
            5: "magenta",  # PLACE_ENEMY_4
            6: "green",  # PLACE_DOOR
        }
        # Add more colors if needed

    def _generate_initial_hero_batch(self):
        """Generates all combinations of hero states based on ranges."""
        if self.initial_hero_batch is not None:
            return self.initial_hero_batch, self.all_hero_combinations

        log.info("Generating all hero state combinations...")
        param_names = ["health", "item1", "item2", "entry", "rooms_left"]
        value_lists = []

        # Ensure correct order and types
        value_lists.append(list(self.hero_param_ranges["health"]))  # e.g., range(1, 11)
        value_lists.append(
            [float(b) for b in self.hero_param_ranges["item1"]]
        )  # e.g., [False, True] -> [0.0, 1.0]
        value_lists.append(
            [float(b) for b in self.hero_param_ranges["item2"]]
        )  # e.g., [False, True] -> [0.0, 1.0]
        # Map Enum entries to integer indices 0-3
        entry_map = {entry: i for i, entry in enumerate(Entry)}
        value_lists.append(
            [float(entry_map[e]) for e in self.hero_param_ranges["entry"]]
        )  # e.g., [Entry.TOP,...] -> [0.0, 1.0, 2.0, 3.0]
        value_lists.append(
            list(self.hero_param_ranges["rooms_left"])
        )  # e.g., range(0, 7)

        self.all_hero_combinations = list(itertools.product(*value_lists))
        self.batch_size_N = len(self.all_hero_combinations)
        log.info(f"Generated {self.batch_size_N} unique hero state combinations.")

        # Convert combinations to a tensor
        self.initial_hero_batch = torch.tensor(
            self.all_hero_combinations, dtype=torch.float
        )
        log.info(f"Hero batch tensor shape: {self.initial_hero_batch.shape}")
        # Ensure hero tensor size matches agent configuration
        if self.initial_hero_batch.shape[1] != self.agent.hero_tensor_size:
            log.error(
                f"Generated hero tensor size ({self.initial_hero_batch.shape[1]}) "
                f"does not match agent's expected size ({self.agent.hero_tensor_size}). Check hero_param_ranges."
            )
            raise ValueError("Hero tensor size mismatch.")

        return self.initial_hero_batch, self.all_hero_combinations

    def _save_metrics(self):
        """Saves collected metrics to the specified CSV file."""
        if not self.metrics:
            log.warning("No metrics collected yet, skipping CSV save.")
            return

        # Define the desired order of columns
        column_order = [
            "episode",
            "final_epsilon",
            "avg_reward",
            "cumulative_reward",
            "avg_tile_changes",
            "avg_steps_to_threshold",
            "n_finished",
            "n_total",
        ]

        # Create DataFrame, handling potential missing columns gracefully if needed
        df = pd.DataFrame(self.metrics)

        # Reorder columns, adding missing ones as NaN if necessary
        df = df.reindex(columns=column_order)

        try:
            # Ensure directory exists
            output_dir = os.path.dirname(self.metrics_csv_path)
            if output_dir:  # Check if dirname is not empty (i.e., not just filename)
                os.makedirs(output_dir, exist_ok=True)

            df.to_csv(
                self.metrics_csv_path, index=False, float_format="%.4f"
            )  # Format floats for readability
            log.info(f"Metrics saved to {self.metrics_csv_path}")
        except IOError as e:
            log.error(f"Error saving metrics to {self.metrics_csv_path}: {e}")
        except Exception as e:
            log.error(f"An unexpected error occurred while saving metrics: {e}")

    def visualize_maps(
        self,
        final_maps: torch.Tensor,
        hero_combinations: List[Tuple],  # Use combinations for display
        num_to_show: int = 5,
    ):
        """Visualizes a sample of generated maps with colors on the terminal."""
        if final_maps is None or not hero_combinations:
            log.warning("No maps or hero combinations available for visualization.")
            return

        num_maps = final_maps.shape[0]
        num_to_show = min(num_to_show, num_maps)
        if num_to_show <= 0:
            log.warning("Number of maps to show is zero or less.")
            return

        log.info(f"Visualizing {num_to_show} generated maps (out of {num_maps})...")

        # Get indices for the maps to show (e.g., first N)
        indices_to_show = list(range(num_to_show))
        # Or use: indices_to_show = random.sample(range(num_maps), num_to_show)

        param_names = ["H", "I1", "I2", "Entry", "RLeft"]  # Short names
        entry_reverse_map = {
            i: entry.name for i, entry in enumerate(Entry)
        }  # Map index back to name

        for i in indices_to_show:
            map_tensor = final_maps[i]
            # Retrieve the original hero combination tuple using the index
            hero_combo_float = hero_combinations[
                i
            ]  # This is the tuple of floats used for tensor

            # Format hero state nicely from the float tuple
            hero_str_parts = []
            hero_str_parts.append(
                f"{param_names[0]}={int(hero_combo_float[0])}"
            )  # Health
            hero_str_parts.append(
                f"{param_names[1]}={'T' if int(hero_combo_float[1])==1 else 'F'}"
            )  # Item1
            hero_str_parts.append(
                f"{param_names[2]}={'T' if int(hero_combo_float[2])==1 else 'F'}"
            )  # Item2
            hero_str_parts.append(
                f"{param_names[3]}={entry_reverse_map.get(int(hero_combo_float[3]), '?')}"
            )  # Entry
            hero_str_parts.append(
                f"{param_names[4]}={int(hero_combo_float[4])}"
            )  # Rooms Left
            hero_info = ", ".join(hero_str_parts)

            console.print(
                f"\n--- Map {i+1}/{num_to_show} (Hero State: {hero_info}) ---"
            )

            height, width = map_tensor.shape
            for r in range(height):
                row_text = Text()
                for c in range(width):
                    tile_val = int(
                        map_tensor[r, c].item()
                    )  # Ensure integer for dict lookup
                    color = self.tile_colors.get(
                        tile_val, "default"
                    )  # Default color if not mapped
                    # Add space for better alignment if numbers are single digit
                    display_char = str(
                        tile_val
                    )  # Don't add extra space, let rich handle width
                    row_text.append(display_char, style=color)
                    row_text.append(" ")  # Add space between numbers
                console.print(row_text)

    # Fix for MDPTrainer.train() method
    # Replace the problematic section in the train method

    def train(self):
        """Runs the main training loop using Rich progress."""
        log.info(f"Starting training for {self.num_episodes} episodes.")
        log.info(f"Max steps per episode: {self.max_steps_per_episode}")
        log.info(
            f"Modification threshold: {self.agent.percentage_change_threshold:.1%}"
        )
        log.info(f"Target modifications: {self.agent.max_modifications} tiles")

        # Generate hero batch once
        initial_hero_batch, hero_combinations = self._generate_initial_hero_batch()
        if self.batch_size_N == 0:
            log.error("Generated hero batch size is 0. Cannot train.")
            return

        # Create initial map batch (all zeros or customize if needed)
        initial_maps = torch.zeros(
            (self.batch_size_N, self.map_size[0], self.map_size[1]), dtype=torch.long
        )

        last_maps_result = None  # To store maps from the last episode for visualization

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
            transient=False,  # Keep progress bar after finishing
        )

        with progress:
            task_id = progress.add_task(
                "[bold green]Training Episodes",
                total=self.num_episodes,
                metrics="Starting...",  # Initial metrics text
            )

            for episode_idx in range(self.num_episodes):
                # Run one episode - agent handles internal step progress (tqdm)
                episode_result = self.agent.train_episode(
                    initial_maps.clone(),  # Start fresh map each episode
                    initial_hero_batch.clone(),  # Pass the full hero batch
                    self.max_steps_per_episode,
                )

                # FIX: Check if 'final_maps' exists in episode_result and handle properly
                last_maps_result = episode_result[
                    "final_maps"
                ]  # Keep track of the last map results

                episode_metrics = episode_result.get(
                    "metrics", {}
                )  # Get metrics dict, default to empty if not present

                # Add episode-specific info to metrics
                full_metrics = {
                    "episode": episode_idx + 1,
                    "final_epsilon": round(self.agent.epsilon, 5),
                    **episode_metrics,  # Unpack metrics from agent
                }
                self.metrics.append(full_metrics)

                # --- Prepare metrics string for Rich progress bar ---
                metrics_str_parts = [
                    f"Ep: {episode_idx+1}/{self.num_episodes}",
                    f"ε: {full_metrics['final_epsilon']:.3f}",
                ]

                # Add optional metrics if they exist
                if "avg_reward" in full_metrics:
                    metrics_str_parts.append(f"AvgR: {full_metrics['avg_reward']:.2f}")
                if "cumulative_reward" in full_metrics:
                    metrics_str_parts.append(
                        f"CumR: {full_metrics['cumulative_reward']:.1f}"
                    )

                # Use np.isnan to check for NaN safely if the metric exists
                if "avg_steps_to_threshold" in full_metrics and not np.isnan(
                    full_metrics["avg_steps_to_threshold"]
                ):
                    metrics_str_parts.append(
                        f"Steps: {full_metrics['avg_steps_to_threshold']:.1f}"
                    )
                else:
                    metrics_str_parts.append("Steps: N/A")

                if "n_finished" in full_metrics and "n_total" in full_metrics:
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

                # Visualize results from the last episode
                if last_maps_result is not None:
                    try:
                        self.visualize_maps(last_maps_result, hero_combinations)
                    except Exception as e:
                        console.print_exception(show_locals=True)
                        log.error(f"Error visualizing maps: {e}")
                else:
                    log.warning("No maps generated in the last episode to visualize.")

        self.agent.save_state(self.agent_save_path)
        log.info("Training finished.")
