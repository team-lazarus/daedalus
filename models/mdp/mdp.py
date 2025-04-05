# Previous imports...
import torch
import torch.nn.functional as F
import random
import math
from enum import Enum
from typing import Tuple, Callable, List, Optional, Dict, Any
import logging

# Use standard tqdm for step progress, rich.progress for episode progress
from tqdm import tqdm, trange
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)  # For better episode progress
import json  # For saving/loading state
import pandas as pd  # For metrics CSV
import itertools  # For hero state combinations
from rich.text import Text  # For colored map display
import os  # To ensure directory exists for saving
import numpy as np  # For NaN

# --- Setup Logging & Console (if not already done) ---
log = logging.getLogger("rich")
if not log.hasHandlers():
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
console = Console()


# --- Enums and Constants (Keep as before) ---
class Entry(Enum):
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"


NO_ACTION = 0
SET_EMPTY = 1
PLACE_ENEMY_1 = 2
PLACE_ENEMY_2 = 3
PLACE_ENEMY_3 = 4
PLACE_ENEMY_4 = 5
PLACE_DOOR = 6
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3


# --- Updated MDPAgent Class ---
class MDPAgent:
    """
    An Epsilon-Greedy MDP agent with Epsilon Decay for map modification tasks.
    Tracks and returns episode metrics. Supports saving/loading state.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        hero_tensor_size: int,
        critic: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        strategy: str,
        percentage_change: float,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ):
        # --- Store config for saving ---
        self.config = {
            "size": size,
            "hero_tensor_size": hero_tensor_size,
            # 'critic': critic, # Critic function cannot be easily serialized to JSON
            "strategy": strategy,
            "percentage_change": percentage_change,
            "initial_epsilon": initial_epsilon,
            "epsilon_decay": epsilon_decay,
            "min_epsilon": min_epsilon,
        }

        # --- Basic Init (as before) ---
        if strategy not in ["narrow", "turtle", "wide"]:
            raise ValueError("Strategy must be 'narrow', 'turtle', or 'wide'")
        if not (0.0 <= percentage_change <= 1.0):
            raise ValueError("percentage_change must be between 0.0 and 1.0")

        self.map_width, self.map_height = size
        self.hero_tensor_size = hero_tensor_size
        self.critic = critic  # Store the function itself (won't be saved)
        self.strategy = strategy
        self.percentage_change_threshold = percentage_change
        self.max_modifications = math.ceil(
            self.map_width * self.map_height * percentage_change
        )

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(
            f"MDPAgent initialized on device: {self.device}. Strategy: {self.strategy}"
        )

        # Action space sizes (as before)
        self.num_tile_change_actions = 7
        if self.strategy == "turtle":
            self.num_actions_turtle_move = 4
            self.num_actions_turtle_change = self.num_tile_change_actions
            self.total_turtle_actions = (
                self.num_actions_turtle_move + self.num_actions_turtle_change
            )
        elif self.strategy == "wide":
            self.num_wide_change_actions = (
                self.num_tile_change_actions - 1
            )  # Exclude NO_ACTION equivalent
            self.total_wide_actions = (
                self.map_width * self.map_height * self.num_wide_change_actions
            )
        else:  # narrow
            self.num_actions_narrow = self.num_tile_change_actions

    # --- _decay_epsilon, _get_greedy_action, _select_action, _apply_action (unchanged) ---
    def _decay_epsilon(self):
        """Decays epsilon."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _get_greedy_action(
        self,
        current_map_batch: torch.Tensor,
        current_hero_batch: torch.Tensor,  # Hero state is input
        agent_pos_batch: Optional[torch.Tensor] = None,
        current_coords: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Determines the greedy action by simulating potential next states and
        evaluating them with the critic, which considers both map and hero state.
        """
        n_batch = current_map_batch.size(0)
        best_actions = torch.zeros(n_batch, dtype=torch.long, device=self.device)

        if n_batch == 0:  # Handle empty batch case
            return best_actions

        with torch.no_grad():
            # Narrow Strategy
            if self.strategy == "narrow":
                assert current_coords is not None
                x, y = current_coords
                possible_next_maps = []
                baseline_reward = self.critic(current_map_batch, current_hero_batch)

                # Simulate change actions (action_idx 1 to num_actions_narrow-1)
                for action_idx in range(
                    1, self.num_actions_narrow
                ):  # Skip NO_ACTION (0)
                    next_map_batch = current_map_batch.clone()
                    next_map_batch[:, x, y] = action_idx
                    possible_next_maps.append(next_map_batch)

                if (
                    not possible_next_maps
                ):  # If only NO_ACTION is possible (shouldn't happen here)
                    return torch.zeros(
                        n_batch, dtype=torch.long, device=self.device
                    )  # Return NO_ACTION

                # Evaluate simulated maps
                simulated_maps = torch.stack(possible_next_maps, dim=1).view(
                    -1, self.map_width, self.map_height
                )
                # Repeat hero batch to match map simulations
                simulated_heroes = current_hero_batch.repeat_interleave(
                    len(possible_next_maps), dim=0
                )

                rewards = self.critic(simulated_maps, simulated_heroes).view(
                    n_batch, -1
                )

                # Find best simulated action index (relative to the simulated actions, offset by 1)
                best_sim_action_indices = torch.argmax(rewards, dim=1)
                # Get the reward corresponding to the best simulated action
                best_sim_rewards = torch.gather(
                    rewards, 1, best_sim_action_indices.unsqueeze(1)
                ).squeeze(1)

                # Decide if the best simulated action is better than doing nothing (NO_ACTION)
                take_best_sim_action = best_sim_rewards > baseline_reward

                # Assign the action: if better, use best sim index + 1; otherwise, use NO_ACTION (0)
                best_actions = torch.where(
                    take_best_sim_action, best_sim_action_indices + 1, NO_ACTION
                )

            # Turtle Strategy
            elif self.strategy == "turtle":
                assert agent_pos_batch is not None
                possible_rewards = []
                current_state_reward = self.critic(
                    current_map_batch, current_hero_batch
                )

                # Rewards for move actions (same as current state reward)
                for _ in range(self.num_actions_turtle_move):
                    possible_rewards.append(current_state_reward)

                # Reward for NO_ACTION change (index num_actions_turtle_move)
                possible_rewards.append(current_state_reward)

                # Rewards for actual tile change actions
                x_coords, y_coords = agent_pos_batch[:, 0], agent_pos_batch[:, 1]
                batch_indices = torch.arange(n_batch, device=self.device)
                for change_action_idx in range(
                    1, self.num_actions_turtle_change
                ):  # Start from 1 (SET_EMPTY)
                    next_map_batch = current_map_batch.clone()
                    # Apply change action to current agent positions
                    next_map_batch[batch_indices, x_coords, y_coords] = (
                        change_action_idx
                    )
                    possible_rewards.append(
                        self.critic(next_map_batch, current_hero_batch)
                    )

                # Stack all rewards (moves + no_change + changes)
                all_rewards = torch.stack(
                    possible_rewards, dim=1
                )  # Shape: (n_batch, total_turtle_actions)
                # Find the action index with the maximum reward
                best_actions = torch.argmax(all_rewards, dim=1)

            # Wide Strategy
            elif self.strategy == "wide":
                # log.warning("Wide strategy greedy search simulating all actions. This can be slow.")
                best_overall_actions = torch.zeros(
                    n_batch, dtype=torch.long, device=self.device
                )
                # Start with the reward of doing nothing (implicitly action 0)
                max_rewards = self.critic(current_map_batch, current_hero_batch)

                action_offset = 0
                # Iterate through possible change actions (1 to 6)
                for change_action_idx in range(1, self.num_tile_change_actions):
                    # Iterate through all map coordinates
                    for x in range(self.map_width):
                        for y in range(self.map_height):
                            # Calculate the flat action index corresponding to this change and location
                            # flat_action = (y * self.map_width + x) * self.num_wide_change_actions + (change_action_idx - 1)
                            # simpler: just increment offset
                            current_flat_action = action_offset

                            # Simulate applying this action
                            next_map_batch = current_map_batch.clone()
                            next_map_batch[:, x, y] = change_action_idx
                            rewards = self.critic(next_map_batch, current_hero_batch)

                            # Check if this action yields a better reward
                            is_better = rewards > max_rewards
                            # Update best actions and max rewards where improvement is found
                            best_overall_actions = torch.where(
                                is_better, current_flat_action, best_overall_actions
                            )
                            max_rewards = torch.where(is_better, rewards, max_rewards)

                            action_offset += 1  # Move to the next flat action index

                best_actions = best_overall_actions  # The best flat action index found

        return best_actions

    def _select_action(
        self,
        current_map_batch: torch.Tensor,
        current_hero_batch: torch.Tensor,
        agent_pos_batch: Optional[torch.Tensor] = None,
        current_coords: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Selects an action for each item in the batch using epsilon-greedy.
        """
        n_batch = current_map_batch.size(0)
        if n_batch == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        sample = torch.rand(n_batch, device=self.device)
        explore_mask = sample < self.epsilon

        greedy_actions = self._get_greedy_action(
            current_map_batch, current_hero_batch, agent_pos_batch, current_coords
        )

        # Determine total number of possible actions based on strategy
        if self.strategy == "narrow":
            total_actions = self.num_actions_narrow
        elif self.strategy == "turtle":
            total_actions = self.total_turtle_actions
        elif self.strategy == "wide":
            total_actions = self.total_wide_actions
        else:
            raise NotImplementedError

        # Generate random actions within the valid range for the strategy
        random_actions = torch.randint(0, total_actions, (n_batch,), device=self.device)

        chosen_actions = torch.where(explore_mask, random_actions, greedy_actions)
        return chosen_actions

    def _apply_action(
        self,
        map_batch: torch.Tensor,
        action_batch: torch.Tensor,
        modified_mask_batch: torch.Tensor,
        agent_pos_batch: Optional[torch.Tensor] = None,  # For Turtle
        current_coords: Optional[Tuple[int, int]] = None,  # For Narrow
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Applies the chosen actions to the map batch, updates modification masks,
        and returns a mask indicating which actions resulted in a tile change.
        """
        next_map_batch = map_batch.clone()
        next_modified_mask_batch = modified_mask_batch.clone()
        next_agent_pos_batch = (
            agent_pos_batch.clone() if agent_pos_batch is not None else None
        )
        n_batch = map_batch.size(0)
        batch_indices = torch.arange(n_batch, device=self.device)
        # Track which actions actually changed a tile's value
        tile_changed_mask = torch.zeros(n_batch, dtype=torch.bool, device=self.device)

        if self.strategy == "narrow":
            assert current_coords is not None
            x, y = current_coords
            # Change occurs only if action is > NO_ACTION
            change_mask = action_batch > NO_ACTION
            if torch.any(change_mask):
                action_values = action_batch[change_mask]  # Actual tile values
                batch_idx_change = batch_indices[change_mask]
                next_map_batch[batch_idx_change, x, y] = action_values
                next_modified_mask_batch[batch_idx_change, x, y] = True
                tile_changed_mask[change_mask] = True  # Mark these as tile changes

        elif self.strategy == "turtle":
            assert next_agent_pos_batch is not None
            current_x, current_y = (
                next_agent_pos_batch[:, 0],
                next_agent_pos_batch[:, 1],
            )

            # --- Handle Moves ---
            move_mask = action_batch < self.num_actions_turtle_move
            if torch.any(move_mask):
                move_actions = action_batch[move_mask]
                batch_idx_move = batch_indices[move_mask]
                pos_to_update = next_agent_pos_batch[move_mask]

                up_mask = move_actions == MOVE_UP
                down_mask = move_actions == MOVE_DOWN
                left_mask = move_actions == MOVE_LEFT
                right_mask = move_actions == MOVE_RIGHT

                pos_to_update[up_mask, 1] = torch.clamp(
                    pos_to_update[up_mask, 1] - 1, 0, self.map_height - 1
                )
                pos_to_update[down_mask, 1] = torch.clamp(
                    pos_to_update[down_mask, 1] + 1, 0, self.map_height - 1
                )
                pos_to_update[left_mask, 0] = torch.clamp(
                    pos_to_update[left_mask, 0] - 1, 0, self.map_width - 1
                )
                pos_to_update[right_mask, 0] = torch.clamp(
                    pos_to_update[right_mask, 0] + 1, 0, self.map_width - 1
                )

                next_agent_pos_batch[batch_idx_move] = pos_to_update
            # Move actions do not change tiles

            # --- Handle Changes ---
            change_mask = action_batch >= self.num_actions_turtle_move
            if torch.any(change_mask):
                # Calculate the intended tile value (action index - offset)
                tile_values = action_batch[change_mask] - self.num_actions_turtle_move
                # An *actual* change happens only if the intended tile value is not NO_ACTION (0)
                is_actual_change = tile_values > NO_ACTION

                if torch.any(is_actual_change):
                    # Get indices for batch items where an actual change occurred
                    batch_idx_actual_change = batch_indices[change_mask][
                        is_actual_change
                    ]
                    # Get coordinates and action values for these changes
                    x_change = current_x[change_mask][is_actual_change]
                    y_change = current_y[change_mask][is_actual_change]
                    action_val_change = tile_values[
                        is_actual_change
                    ]  # Use the non-zero tile values

                    # Apply the changes to the map and modification mask
                    next_map_batch[batch_idx_actual_change, x_change, y_change] = (
                        action_val_change
                    )
                    next_modified_mask_batch[
                        batch_idx_actual_change, x_change, y_change
                    ] = True
                    # Mark these actions as having caused a tile change
                    # Need to map back `is_actual_change` to the original `change_mask` dimensions
                    tile_changed_mask[change_mask] = is_actual_change

        elif self.strategy == "wide":
            # Determine target tile value (action_val_indices 0-5 map to tiles 1-6)
            action_val_indices = action_batch % self.num_wide_change_actions
            tile_values = (
                action_val_indices + 1
            )  # Map to SET_EMPTY, PLACE_ENEMY_1, ... PLACE_DOOR

            # Determine target coordinates
            flat_coords = action_batch // self.num_wide_change_actions
            target_x = flat_coords % self.map_width
            target_y = flat_coords // self.map_width  # Corrected calculation

            # Apply the change
            next_map_batch[batch_indices, target_x, target_y] = tile_values
            next_modified_mask_batch[batch_indices, target_x, target_y] = True
            # All wide actions result in a tile change
            tile_changed_mask[:] = True  # Set all to True for wide strategy

        else:
            raise NotImplementedError

        return (
            next_map_batch,
            next_modified_mask_batch,
            next_agent_pos_batch,
            tile_changed_mask,
        )

    def train_episode(
        self,
        initial_map_state: torch.Tensor,
        initial_hero_state: torch.Tensor,
        max_steps: int = 1000,
    ) -> Dict[str, Any]:
        """
        Runs a single training "episode" for a batch of maps with progress bar
        and returns final maps along with calculated metrics.
        """
        n_batch = initial_map_state.size(0)
        log.debug(
            f"Starting episode for {n_batch} instances. Max steps: {max_steps}. Strategy: {self.strategy}"
        )

        # --- Initialization for Episode ---
        current_map_batch = initial_map_state.clone().to(self.device)
        current_hero_batch = initial_hero_state.clone().to(
            self.device
        )  # Keep hero state constant for the episode
        modified_mask_batch = torch.zeros_like(
            current_map_batch, dtype=torch.bool, device=self.device
        )
        num_modified_tiles = torch.zeros(n_batch, dtype=torch.long, device=self.device)
        active_mask = torch.ones(
            n_batch, dtype=torch.bool, device=self.device
        )  # Tracks which batch items are still running

        agent_pos_batch = None
        narrow_x, narrow_y = 0, 0
        if self.strategy == "turtle":
            # Start turtle at (0,0) for all batch items
            agent_pos_batch = torch.zeros(
                (n_batch, 2), dtype=torch.long, device=self.device
            )
        elif self.strategy == "narrow":
            narrow_x, narrow_y = 0, 0  # Start narrow scan at (0,0)

        # --- Metrics Tracking ---
        episode_cumulative_reward = torch.zeros(n_batch, device=self.device)
        # Count of actions that actually changed a tile (proxy for exploration/entropy)
        episode_tile_change_count = torch.zeros(
            n_batch, dtype=torch.long, device=self.device
        )
        episode_steps_taken = torch.zeros(n_batch, dtype=torch.long, device=self.device)
        # Record the step number when each item finishes (-1 if not finished)
        steps_to_threshold = torch.full(
            (n_batch,), -1, dtype=torch.long, device=self.device
        )
        final_step_count = max_steps  # Assume max_steps unless all terminate earlier

        # --- Step Loop with tqdm ---
        # Use standard tqdm here for simpler step progress display
        pbar = tqdm(
            range(max_steps),
            desc=f"Ep Step (ε={self.epsilon:.3f})",
            leave=False,
            dynamic_ncols=True,
            ascii=True,
        )
        for step in pbar:
            # Check if any agents are still active
            if not torch.any(active_mask):
                final_step_count = step  # Record the step where all finished
                log.debug(f"All agents terminated at step {step}.")
                break

            # --- Process only active agents ---
            active_indices = torch.where(active_mask)[0]
            if (
                len(active_indices) == 0
            ):  # Should be caught by the check above, but safety first
                break

            active_maps = current_map_batch[active_mask]
            active_heroes = current_hero_batch[
                active_mask
            ]  # Use the corresponding hero states
            active_agent_pos = (
                agent_pos_batch[active_mask] if agent_pos_batch is not None else None
            )
            active_mod_masks = modified_mask_batch[active_mask]

            # Determine current coordinates for narrow strategy (same for all active agents in narrow)
            current_narrow_coords = (
                (narrow_x, narrow_y) if self.strategy == "narrow" else None
            )

            # --- Action Selection ---
            actions = self._select_action(
                active_maps,
                active_heroes,  # Pass active hero states
                active_agent_pos,
                current_narrow_coords,
            )

            # --- Action Application ---
            next_maps, next_mod_masks, next_agent_pos, tile_changed = (
                self._apply_action(
                    active_maps,
                    actions,
                    active_mod_masks,
                    active_agent_pos,
                    current_narrow_coords,
                )
            )

            # --- Reward Calculation (based on the state *after* the action) ---
            # Use the critic to evaluate the resulting maps for the active agents
            current_rewards = self.critic(next_maps, active_heroes)
            # Add rewards to the cumulative sum for active agents
            episode_cumulative_reward[active_mask] += current_rewards

            # --- Update State for Active Agents ---
            current_map_batch[active_mask] = next_maps
            modified_mask_batch[active_mask] = next_mod_masks
            if agent_pos_batch is not None and next_agent_pos is not None:
                agent_pos_batch[active_mask] = next_agent_pos

            # Increment step count for active agents
            episode_steps_taken[active_mask] += 1
            # Increment tile change count where applicable
            episode_tile_change_count[active_mask] += tile_changed.long()

            # --- Check Termination ---
            # Recalculate modification counts for the *updated* active agents
            current_mod_counts = next_mod_masks.sum(dim=(1, 2))
            # Update the global modification count tracker
            num_modified_tiles[active_mask] = current_mod_counts

            # Determine which active agents have *just* met the threshold in this step
            terminated_now_mask_active = current_mod_counts >= self.max_modifications
            # Map this back to the original batch indices
            newly_terminated_global_mask = torch.zeros_like(active_mask)
            newly_terminated_global_mask[active_mask] = terminated_now_mask_active

            # Record step number for newly terminated agents if not already recorded
            not_recorded_mask = steps_to_threshold == -1
            record_step_mask = newly_terminated_global_mask & not_recorded_mask
            steps_to_threshold[record_step_mask] = (
                step + 1
            )  # Record the step number (1-based)

            # Update the active mask: remove agents that terminated now
            active_mask = active_mask & ~newly_terminated_global_mask

            # --- Epsilon Decay ---
            self._decay_epsilon()

            # --- Update Strategy-Specific State (e.g., Narrow Scan Position) ---
            if self.strategy == "narrow":
                narrow_y += 1
                if narrow_y >= self.map_height:
                    narrow_y = 0
                    narrow_x += 1
                    if narrow_x >= self.map_width:
                        log.debug("Narrow strategy completed map scan.")
                        # Force termination if scan completes, regardless of modification %
                        steps_to_threshold[(steps_to_threshold == -1) & active_mask] = (
                            step + 1
                        )
                        active_mask[:] = False  # Stop all remaining narrow agents

            # --- Update tqdm Description ---
            # Show avg cumulative reward *so far* for currently active agents
            if torch.any(active_mask):
                avg_cum_reward_active = torch.mean(
                    episode_cumulative_reward[active_mask]
                ).item()
                pbar.set_description(
                    f"Ep Step (Act:{torch.sum(active_mask).item()}/{n_batch}, ε={self.epsilon:.3f})"
                )
                pbar.set_postfix(avg_rew=f"{avg_cum_reward_active:.2f}")
            else:
                pbar.set_description(f"Ep Step (Finished, ε={self.epsilon:.3f})")

        pbar.close()
        log.debug(f"Episode finished processing after {final_step_count} steps.")

        # --- Calculate Final Metrics for the Episode (Averaged over Batch) ---
        valid_steps_mask = episode_steps_taken > 0
        # Average reward per step = total cumulative reward / total steps taken
        avg_reward_per_step = torch.zeros_like(episode_cumulative_reward)
        avg_reward_per_step[valid_steps_mask] = (
            episode_cumulative_reward[valid_steps_mask]
            / episode_steps_taken[valid_steps_mask].float()
        )

        # Calculate batch averages
        batch_avg_reward = torch.mean(avg_reward_per_step).item()
        batch_cumulative_reward = torch.mean(episode_cumulative_reward).item()
        batch_avg_tile_changes = torch.mean(
            episode_tile_change_count.float()
        ).item()  # Entropy proxy

        # Calculate average steps to threshold for those that finished
        finished_mask = steps_to_threshold != -1
        if torch.any(finished_mask):
            batch_avg_steps_to_threshold = torch.mean(
                steps_to_threshold[finished_mask].float()
            ).item()
        else:
            # Handle case where no items finished (e.g., max_steps too low)
            batch_avg_steps_to_threshold = float(
                "nan"
            )  # Or max_steps, depending on desired representation

        metrics = {
            "avg_reward": batch_avg_reward,
            "cumulative_reward": batch_cumulative_reward,
            "avg_tile_changes": batch_avg_tile_changes,  # Proxy for policy entropy
            "avg_steps_to_threshold": batch_avg_steps_to_threshold,
            "n_finished": torch.sum(finished_mask).item(),
            "n_total": n_batch,
        }

        return {"final_maps": current_map_batch.cpu(), "metrics": metrics}

    # --- save_state / load_state (unchanged) ---
    def save_state(self, filepath: str):
        """Saves the agent's configuration and current epsilon to a JSON file."""
        state = {"config": self.config, "epsilon": self.epsilon}
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            with open(filepath, "w") as f:
                json.dump(state, f, indent=4)
            log.info(f"Agent state saved successfully to {filepath}")
        except IOError as e:
            log.error(f"Error saving agent state to {filepath}: {e}")
        except Exception as e:
            log.error(f"An unexpected error occurred while saving agent state: {e}")

    def load_state(self, filepath: str):
        """Loads the agent's state (epsilon) from a JSON file."""
        try:
            with open(filepath, "r") as f:
                state = json.load(f)

            # --- Optional: Config Verification ---
            loaded_config = state.get("config", {})
            mismatched_keys = []
            # Compare loaded config (except critic) with current agent config
            for key, value in loaded_config.items():
                if key != "critic":  # Critic cannot be saved/loaded
                    current_value = self.config.get(key)
                    # Handle potential type differences (e.g., list vs tuple for size)
                    if isinstance(current_value, list) and isinstance(value, list):
                        current_value = tuple(current_value)
                        value = tuple(value)
                    if current_value != value:
                        mismatched_keys.append(key)

            if mismatched_keys:
                log.warning(
                    f"Loaded state config mismatch for keys: {mismatched_keys}. "
                    f"Current: {[self.config.get(k) for k in mismatched_keys]}, "
                    f"Loaded: {[loaded_config.get(k) for k in mismatched_keys]}. "
                    f"Only epsilon will be loaded."
                )
            # --- End Optional Verification ---

            # Load epsilon, default to initial if missing in file
            self.epsilon = state.get("epsilon", self.config["initial_epsilon"])
            log.info(
                f"Agent state loaded successfully from {filepath}. Current epsilon: {self.epsilon:.4f}"
            )

        except FileNotFoundError:
            log.error(f"Error loading agent state: File not found at {filepath}")
        except (IOError, json.JSONDecodeError) as e:
            log.error(f"Error reading or parsing agent state file {filepath}: {e}")
        except Exception as e:
            log.error(f"An unexpected error occurred while loading agent state: {e}")
