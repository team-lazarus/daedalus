import torch
import torch.nn.functional as F
import random
import math
from enum import Enum
from typing import Tuple, Callable, List, Optional, Dict, Any
import logging
import os
import json
import numpy as np
# Use standard tqdm here, Rich progress is handled by the Trainer
from tqdm import tqdm # Changed from trange to tqdm for more control
from rich.logging import RichHandler

# --- Setup Logging ---
log = logging.getLogger("rich")
if not log.hasHandlers():
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

# --- Enums and Constants ---
class Entry(Enum):
    TOP = "top"; RIGHT = "right"; BOTTOM = "bottom"; LEFT = "left"

# Tile Modification Values/Actions
NO_ACTION = 0
SET_EMPTY = 1
PLACE_ENEMY_1 = 2
PLACE_ENEMY_2 = 3
PLACE_ENEMY_3 = 4
PLACE_ENEMY_4 = 5
PLACE_DOOR = 6
NUM_TILE_VALUES = 7 # 0 through 6

# Turtle Movement Actions
MOVE_UP = 0; MOVE_DOWN = 1; MOVE_LEFT = 2; MOVE_RIGHT = 3
NUM_TURTLE_MOVES = 4

class MDPAgent:
    """
    Epsilon-Greedy MDP agent with Epsilon Decay for map modification.
    Supports narrow (random start), turtle, and optimized wide (sampling) strategies.
    Includes prerequisite for initial tile placement before termination counting.
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
        wide_greedy_sample_k: int = 64, # Number of samples for Wide greedy action
    ):
        if strategy not in ["narrow", "turtle", "wide"]:
            raise ValueError("Strategy must be 'narrow', 'turtle', or 'wide'")
        if not (0.0 <= percentage_change <= 1.0):
            raise ValueError("percentage_change must be between 0.0 and 1.0")

        self.map_width, self.map_height = size
        self.hero_tensor_size = hero_tensor_size
        self.critic = critic
        self.strategy = strategy
        self.percentage_change_threshold = percentage_change
        self.max_modifications = math.ceil(
            self.map_width * self.map_height * percentage_change
        )
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.wide_greedy_sample_k = wide_greedy_sample_k

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"MDPAgent using {self.strategy} strategy on {self.device}.")

        # Action Space Sizes
        self.num_tile_place_values = NUM_TILE_VALUES - 1 # Values 1 through 6 (SET_EMPTY to PLACE_DOOR)

        if self.strategy == "turtle":
            self.num_actions_turtle_change = NUM_TILE_VALUES # Includes NO_ACTION(0)
            self.total_turtle_actions = NUM_TURTLE_MOVES + self.num_actions_turtle_change # Moves + Change Tile (incl NO_ACTION)
            self.num_total_actions = self.total_turtle_actions
        elif self.strategy == "wide":
            # Total possible modification actions (place value V at coord X,Y)
            self.total_wide_actions = self.map_width * self.map_height * self.num_tile_place_values
            # Add 1 for the explicit "no action" (action index 0) case used in select/apply logic
            self.num_total_actions = self.total_wide_actions + 1
        else: # narrow
            self.num_actions_narrow = NUM_TILE_VALUES # Includes NO_ACTION(0) for changing the current tile
            self.num_total_actions = self.num_actions_narrow

        # Config for saving state (critic cannot be saved)
        # Recalculate num_total_actions in __init__ based on strategy
        self.config = {k: v for k, v in locals().items() if k not in ['self', 'critic', '__class__']}


    def _decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _get_greedy_action(
        self,
        current_map_batch: torch.Tensor,
        current_hero_batch: torch.Tensor,
        agent_pos_batch: Optional[torch.Tensor] = None,
        current_coords_narrow: Optional[Tuple[int, int]] = None, # Used only by Narrow
    ) -> torch.Tensor:
        """
        Determines the greedy action using tensorized simulation or sampling.
        For Wide, uses sampling for efficiency. The primary bottleneck is the critic evaluation.
        """
        n_batch = current_map_batch.size(0)
        if n_batch == 0: return torch.zeros(0, dtype=torch.long, device=self.device)

        with torch.no_grad():
            # Baseline reward for taking NO_ACTION (or making no change)
            baseline_reward = self.critic(current_map_batch, current_hero_batch)

            if self.strategy == "narrow":
                assert current_coords_narrow is not None, "Narrow strategy requires current_coords_narrow"
                x, y = current_coords_narrow
                num_sim_actions = self.num_actions_narrow - 1 # Actions 1 through 6 (tile changes)
                if num_sim_actions == 0: # Only NO_ACTION possible
                    return torch.zeros(n_batch, dtype=torch.long, device=self.device)

                # Simulate changing the current tile (x, y) to each possible value (1-6)
                simulated_maps = current_map_batch.unsqueeze(1).repeat(1, num_sim_actions, 1, 1)
                action_values = torch.arange(1, self.num_actions_narrow, device=self.device) # Values 1 to 6
                # Apply action values to the specific coord (x, y) for each simulation
                simulated_maps[:, torch.arange(num_sim_actions), x, y] = action_values

                # Flatten for batch critic call
                simulated_maps_flat = simulated_maps.view(-1, self.map_width, self.map_height)
                simulated_heroes = current_hero_batch.repeat_interleave(num_sim_actions, dim=0)

                # Get rewards for simulated actions
                rewards = self.critic(simulated_maps_flat, simulated_heroes).view(n_batch, num_sim_actions)

                # Find the best simulation reward and corresponding action index
                best_sim_rewards, best_sim_action_indices = torch.max(rewards, dim=1)

                # Choose the best simulated action only if it's better than doing nothing
                take_best_sim_action = best_sim_rewards > baseline_reward
                # Action index 0 is NO_ACTION, simulation indices map to actions 1 to num_sim_actions+1
                best_actions = torch.where(take_best_sim_action, best_sim_action_indices + 1, torch.tensor(NO_ACTION, device=self.device))


            elif self.strategy == "turtle":
                assert agent_pos_batch is not None, "Turtle strategy requires agent_pos_batch"
                # Evaluate tile change actions (1-6) at the current agent position
                num_sim_change_actions = self.num_actions_turtle_change - 1 # Actions 1 through 6
                if num_sim_change_actions > 0:
                    x_coords, y_coords = agent_pos_batch[:, 0], agent_pos_batch[:, 1]
                    batch_indices = torch.arange(n_batch, device=self.device)

                    simulated_change_maps = current_map_batch.unsqueeze(1).repeat(1, num_sim_change_actions, 1, 1)
                    action_values = torch.arange(1, self.num_actions_turtle_change, device=self.device) # 1 to 6
                    # Apply changes at current agent positions for each simulation
                    simulated_change_maps[batch_indices.unsqueeze(1), torch.arange(num_sim_change_actions).unsqueeze(0), x_coords.unsqueeze(1), y_coords.unsqueeze(1)] = action_values.unsqueeze(0)

                    simulated_change_maps_flat = simulated_change_maps.view(-1, self.map_width, self.map_height)
                    simulated_heroes_change = current_hero_batch.repeat_interleave(num_sim_change_actions, dim=0)
                    change_rewards = self.critic(simulated_change_maps_flat, simulated_heroes_change).view(n_batch, num_sim_change_actions)
                else:
                    change_rewards = torch.empty((n_batch, 0), device=self.device) # No change actions possible

                # Combine rewards: Moves (implicitly baseline), No Tile Change (baseline), Tile Changes
                move_rewards = baseline_reward.unsqueeze(1).repeat(1, NUM_TURTLE_MOVES)
                no_change_reward_val = NUM_TURTLE_MOVES # Action index for NO_ACTION tile change is 4
                no_change_reward = baseline_reward.unsqueeze(1) # Action corresponding to value NO_ACTION(0)

                all_rewards = torch.cat([move_rewards, no_change_reward, change_rewards], dim=1)
                # Action indices: 0-3 (moves), 4 (no tile change), 5-10 (tile changes 1-6)
                best_actions = torch.argmax(all_rewards, dim=1)


            elif self.strategy == "wide":
                # Sampled Greedy for Wide: Evaluate K random actions instead of all W*H*V actions.
                k = min(self.wide_greedy_sample_k, self.total_wide_actions)
                if k == 0: # No possible modification actions
                     return torch.zeros(n_batch, dtype=torch.long, device=self.device) # Return 0 for "no action"

                # Sample k random flat action indices per batch item. Indices range 0 to total_wide_actions - 1.
                sampled_action_indices = torch.randint(0, self.total_wide_actions, (n_batch, k), device=self.device)

                # Decode flat actions: tile value (1-6), x, y
                action_val_indices = sampled_action_indices % self.num_tile_place_values # 0 to num_tile_place_values-1
                tile_values = action_val_indices + 1                                    # 1 to 6 (actual tile value)
                flat_coords_times_vals = sampled_action_indices // self.num_tile_place_values
                target_x = flat_coords_times_vals % self.map_width
                target_y = flat_coords_times_vals // self.map_width

                # --- Simulate the k sampled actions for each batch item ---
                simulated_maps = current_map_batch.unsqueeze(1).expand(-1, k, -1, -1).clone()
                simulated_heroes = current_hero_batch.repeat_interleave(k, dim=0)

                batch_idx_rep = torch.arange(n_batch, device=self.device).unsqueeze(1).expand(-1, k)
                simulated_maps[batch_idx_rep, torch.arange(k, device=self.device).unsqueeze(0), target_x, target_y] = tile_values

                simulated_maps_flat = simulated_maps.view(-1, self.map_width, self.map_height)
                rewards = self.critic(simulated_maps_flat, simulated_heroes).view(n_batch, k) # Reshape rewards back

                best_sampled_rewards, best_indices_in_k = torch.max(rewards, dim=1)
                best_sampled_flat_actions = sampled_action_indices.gather(1, best_indices_in_k.unsqueeze(1)).squeeze(1)

                take_best_sampled_action = best_sampled_rewards > baseline_reward
                # Action index 0 = "no action", indices >= 1 are actual actions (shifted flat index + 1)
                best_actions = torch.where(take_best_sampled_action, best_sampled_flat_actions + 1, torch.tensor(0, device=self.device))


        return best_actions

    def _select_action(
        self,
        current_map_batch: torch.Tensor,
        current_hero_batch: torch.Tensor,
        agent_pos_batch: Optional[torch.Tensor] = None,
        current_coords_narrow: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """ Selects actions using epsilon-greedy policy. """
        n_batch = current_map_batch.size(0)
        if n_batch == 0: return torch.empty((0,), dtype=torch.long, device=self.device)

        explore = torch.rand(1).item() < self.epsilon # Single random draw for exploration decision

        if explore:
             # Select random action from the valid range for the strategy
             total_actions = self.num_total_actions
             if total_actions > 0:
                 # randint samples from [low, high)
                 chosen_actions = torch.randint(0, total_actions, (n_batch,), device=self.device)
             else: # No actions possible at all
                 chosen_actions = torch.zeros(n_batch, dtype=torch.long, device=self.device) # Default to NO_ACTION/0

        else:
            # Select greedy action
            chosen_actions = self._get_greedy_action(
                current_map_batch, current_hero_batch, agent_pos_batch, current_coords_narrow
            )

        return chosen_actions


    def _apply_action(
        self,
        map_batch: torch.Tensor,
        action_batch: torch.Tensor,
        modified_mask_batch: torch.Tensor,
        agent_pos_batch: Optional[torch.Tensor] = None,
        current_coords_narrow: Optional[Tuple[int, int]] = None, # Used only by Narrow
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Applies actions to the map batch, updates the modification mask *only* if the tile value actually changes.
        Returns the next map state, next modification mask, and next agent positions (for turtle).
        """
        next_map_batch = map_batch.clone()
        next_modified_mask_batch = modified_mask_batch.clone()
        next_agent_pos_batch = agent_pos_batch.clone() if agent_pos_batch is not None else None
        n_batch = map_batch.size(0)
        if n_batch == 0: return next_map_batch, next_modified_mask_batch, next_agent_pos_batch

        batch_indices = torch.arange(n_batch, device=self.device)

        if self.strategy == "narrow":
            assert current_coords_narrow is not None, "Narrow strategy requires current_coords_narrow"
            x, y = current_coords_narrow
            # Actions 0-6 represent tile values (0=No Change, 1-6=Set Tile)
            action_values = action_batch
            attempt_mask = action_values > NO_ACTION # Mask for actions that intend to change the tile (value > 0)
            if torch.any(attempt_mask):
                batch_idx_attempt = batch_indices[attempt_mask]
                action_val_attempt = action_values[attempt_mask] # Values 1 to 6

                # Check if the value actually changes
                original_values = map_batch[batch_idx_attempt, x, y]
                value_did_change = (action_val_attempt != original_values)

                # Apply the change
                next_map_batch[batch_idx_attempt, x, y] = action_val_attempt

                # Update modification mask only where the value actually changed
                batch_idx_effective = batch_idx_attempt[value_did_change]
                if batch_idx_effective.numel() > 0:
                     next_modified_mask_batch[batch_idx_effective, x, y] = True


        elif self.strategy == "turtle":
            assert next_agent_pos_batch is not None, "Turtle strategy requires agent_pos_batch"
            current_x, current_y = next_agent_pos_batch[:, 0], next_agent_pos_batch[:, 1]

            # --- Apply Moves (Actions 0-3) ---
            move_mask = action_batch < NUM_TURTLE_MOVES
            if torch.any(move_mask):
                move_actions = action_batch[move_mask]
                batch_idx_move = batch_indices[move_mask]
                pos_to_update = next_agent_pos_batch[move_mask].clone() # Clone slice to avoid modifying original inplace during calculation

                up_mask = move_actions == MOVE_UP
                down_mask = move_actions == MOVE_DOWN
                left_mask = move_actions == MOVE_LEFT
                right_mask = move_actions == MOVE_RIGHT

                pos_to_update[up_mask, 1] = torch.clamp(pos_to_update[up_mask, 1] - 1, 0, self.map_height - 1)
                pos_to_update[down_mask, 1] = torch.clamp(pos_to_update[down_mask, 1] + 1, 0, self.map_height - 1)
                pos_to_update[left_mask, 0] = torch.clamp(pos_to_update[left_mask, 0] - 1, 0, self.map_width - 1)
                pos_to_update[right_mask, 0] = torch.clamp(pos_to_update[right_mask, 0] + 1, 0, self.map_width - 1)

                next_agent_pos_batch[batch_idx_move] = pos_to_update # Update the main tensor

            # --- Apply Tile Changes (Actions >= NUM_TURTLE_MOVES) ---
            # Actions 4-10 correspond to tile values 0-6
            change_action_mask = action_batch >= NUM_TURTLE_MOVES
            if torch.any(change_action_mask):
                tile_values_intended = action_batch[change_action_mask] - NUM_TURTLE_MOVES # Map actions 4.. to values 0..
                batch_idx_change_action = batch_indices[change_action_mask]
                x_change = current_x[change_action_mask]
                y_change = current_y[change_action_mask]

                # Filter for actions that actually intend to place a tile (value > 0)
                attempt_mask = tile_values_intended > NO_ACTION # Values 1 to 6
                if torch.any(attempt_mask):
                    batch_idx_attempt = batch_idx_change_action[attempt_mask]
                    x_attempt = x_change[attempt_mask]
                    y_attempt = y_change[attempt_mask]
                    action_val_attempt = tile_values_intended[attempt_mask] # 1 to 6

                    # Check if the value actually changes
                    original_values = map_batch[batch_idx_attempt, x_attempt, y_attempt]
                    value_did_change = action_val_attempt != original_values

                    # Apply the change
                    next_map_batch[batch_idx_attempt, x_attempt, y_attempt] = action_val_attempt

                    # Update modification mask only where value changed
                    batch_idx_effective = batch_idx_attempt[value_did_change]
                    x_effective = x_attempt[value_did_change]
                    y_effective = y_attempt[value_did_change]
                    if batch_idx_effective.numel() > 0:
                        # Use advanced indexing for mask update
                        next_modified_mask_batch[batch_idx_effective, x_effective, y_effective] = True


        elif self.strategy == "wide":
             # Actions are 0 (no-op) or 1 to total_wide_actions
             valid_action_mask = action_batch > 0 # Ignore if action is 0 (no-op chosen by greedy or random)
             if torch.any(valid_action_mask):
                 action_batch_valid = action_batch[valid_action_mask] - 1 # Adjust back to 0-based flat index
                 batch_indices_valid = batch_indices[valid_action_mask]

                 # Decode flat actions
                 action_val_indices = action_batch_valid % self.num_tile_place_values # 0 to num_tile_place_values-1
                 tile_values_intended = action_val_indices + 1                        # 1 to 6
                 flat_coords_times_vals = action_batch_valid // self.num_tile_place_values
                 target_x = flat_coords_times_vals % self.map_width
                 target_y = flat_coords_times_vals // self.map_width

                 # Check if the value actually changes
                 original_values = map_batch[batch_indices_valid, target_x, target_y]
                 value_did_change = tile_values_intended != original_values

                 # Apply the change
                 next_map_batch[batch_indices_valid, target_x, target_y] = tile_values_intended

                 # Update modification mask only where value changed
                 batch_idx_effective = batch_indices_valid[value_did_change]
                 target_x_effective = target_x[value_did_change]
                 target_y_effective = target_y[value_did_change]
                 if batch_idx_effective.numel() > 0:
                     next_modified_mask_batch[batch_idx_effective, target_x_effective, target_y_effective] = True

        else: raise NotImplementedError("Unknown strategy in _apply_action")

        return next_map_batch, next_modified_mask_batch, next_agent_pos_batch

    def _check_initial_tiles_placed(self, map_batch: torch.Tensor) -> torch.Tensor:
        """ Checks if all required tile types (1-6) are present in each map of the batch. """
        n_batch = map_batch.size(0)
        if n_batch == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        tile_values_to_check = torch.arange(1, NUM_TILE_VALUES, device=self.device) # 1 to 6
        num_tiles_to_check = len(tile_values_to_check)

        map_flat = map_batch.view(n_batch, -1)
        present_mask = (map_flat.unsqueeze(1) == tile_values_to_check.view(1, -1, 1)).any(dim=2)
        all_placed_mask = present_mask.all(dim=1) # (n_batch,) - True if all tiles 1-6 exist

        return all_placed_mask


    def train_episode(
        self,
        initial_map_state: torch.Tensor,
        initial_hero_state: torch.Tensor,
        max_steps: int = 1000,
    ) -> Dict[str, Any]:
        """ Runs a single training episode. """
        n_batch = initial_map_state.size(0)
        total_map_tiles = self.map_width * self.map_height
        if n_batch == 0:
             log.warning("train_episode called with empty batch.")
             return {
                 "final_maps": torch.empty((0, self.map_width, self.map_height)),
                 "final_cumulative_rewards": torch.empty(0), # NEW: Return empty tensor
                 "metrics": {
                     "avg_cumulative_reward": 0.0,
                     "avg_steps_to_threshold": float('nan'),
                     "n_finished": 0, # Renamed for clarity
                     "n_total": 0,
                     "policy_entropy": 0.0, # NEW: Add entropy metric
                     "final_epsilon": self.epsilon
                 }
             }

        # --- Initialization ---
        current_map_batch = initial_map_state.clone().to(self.device)
        current_hero_batch = initial_hero_state.clone().to(self.device)
        modified_mask_batch = torch.zeros_like(current_map_batch, dtype=torch.bool, device=self.device)
        active_mask = torch.ones(n_batch, dtype=torch.bool, device=self.device)
        initial_tiles_placed = self._check_initial_tiles_placed(current_map_batch) # Check initial state

        agent_pos_batch = None
        narrow_coords_sequence = None

        if self.strategy == "turtle":
            rand_x = torch.randint(0, self.map_width, (n_batch,), device=self.device)
            rand_y = torch.randint(0, self.map_height, (n_batch,), device=self.device)
            agent_pos_batch = torch.stack([rand_x, rand_y], dim=1).long()
        elif self.strategy == "narrow":
            start_x = random.randint(0, self.map_width - 1)
            start_y = random.randint(0, self.map_height - 1)
            #log.info(f"Narrow strategy starting episode scan at ({start_x}, {start_y})") # Less verbose
            narrow_coords_sequence = []
            for i in range(total_map_tiles):
                current_y_offset = (start_y + i) % self.map_height
                current_x_offset = (start_x + (start_y + i) // self.map_height) % self.map_width
                narrow_coords_sequence.append((current_x_offset, current_y_offset))

        # --- Metrics Tracking ---
        # Store cumulative reward PER ITEM in the batch
        episode_cumulative_reward = torch.zeros(n_batch, device=self.device)
        episode_steps_taken = torch.zeros(n_batch, dtype=torch.long, device=self.device)
        steps_to_threshold = torch.full((n_batch,), -1, dtype=torch.long, device=self.device)
        final_step_count = max_steps

        # --- Step Loop ---
        actual_max_steps = max_steps
        if self.strategy == "narrow":
            actual_max_steps = max(max_steps, total_map_tiles)

        # Use standard tqdm for step progress
        pbar = tqdm(range(actual_max_steps), desc=f"Ep Step (ε={self.epsilon:.3f})", leave=False, dynamic_ncols=True, ascii=True)
        for step in pbar:
            if not torch.any(active_mask):
                final_step_count = step
                break

            active_indices = torch.where(active_mask)[0]
            if len(active_indices) == 0: break

            active_maps = current_map_batch[active_mask]
            active_heroes = current_hero_batch[active_mask]
            active_agent_pos = agent_pos_batch[active_mask] if agent_pos_batch is not None else None
            active_mod_masks = modified_mask_batch[active_mask]

            current_narrow_coords = None
            if self.strategy == "narrow":
                 current_narrow_coords = narrow_coords_sequence[step % total_map_tiles]

            actions = self._select_action(active_maps, active_heroes, active_agent_pos, current_narrow_coords)
            next_maps, next_mod_masks, next_agent_pos = self._apply_action(
                active_maps, actions, active_mod_masks, active_agent_pos, current_narrow_coords
            )

            # --- Reward & State Update ---
            current_rewards = self.critic(next_maps, active_heroes)
            # Update cumulative reward for the active items
            episode_cumulative_reward[active_mask] += current_rewards
            current_map_batch[active_mask] = next_maps
            modified_mask_batch[active_mask] = next_mod_masks
            if agent_pos_batch is not None and next_agent_pos is not None:
                agent_pos_batch[active_mask] = next_agent_pos
            episode_steps_taken[active_mask] += 1

            # --- Check Prerequisite & Termination ---
            not_met_prereq_active_mask = ~initial_tiles_placed[active_mask]
            if torch.any(not_met_prereq_active_mask):
                 indices_to_check = active_indices[not_met_prereq_active_mask]
                 maps_to_check = current_map_batch[indices_to_check]
                 newly_met_mask = self._check_initial_tiles_placed(maps_to_check)
                 initial_tiles_placed[indices_to_check] = newly_met_mask

            can_terminate_mask = active_mask & initial_tiles_placed
            if torch.any(can_terminate_mask):
                eligible_indices = torch.where(can_terminate_mask)[0]
                current_mod_counts = modified_mask_batch[eligible_indices].sum(dim=(1, 2))
                terminated_now_eligible_mask = current_mod_counts >= self.max_modifications

                terminated_now_global_mask = torch.zeros_like(active_mask)
                terminated_now_global_mask[eligible_indices[terminated_now_eligible_mask]] = True

                not_recorded_mask = steps_to_threshold == -1
                record_step_mask = terminated_now_global_mask & not_recorded_mask
                steps_to_threshold[record_step_mask] = step + 1

                active_mask = active_mask & ~terminated_now_global_mask

            # Epsilon Decay (applied once per step)
            self._decay_epsilon()

            # Progress Bar Update (using standard tqdm)
            num_active = torch.sum(active_mask).item()
            # Calculate average cumulative reward of ACTIVE agents
            avg_cum_rew_active = 0.0
            if num_active > 0:
                 avg_cum_rew_active = episode_cumulative_reward[active_mask].mean().item()

            # Update description string to include avg reward
            desc = f"Ep Step (Act:{num_active}/{n_batch}, ε={self.epsilon:.3f}, AvgR:{avg_cum_rew_active:.2f})"
            pbar.set_description(desc)
            # Optionally add postfix if description gets too long
            # pbar.set_postfix(avg_rwd=f"{avg_cum_rew_active:.2f}")


        pbar.close()
        steps_to_threshold[(steps_to_threshold == -1) & active_mask] = actual_max_steps + 1

        # --- Calculate Policy Entropy (based on final epsilon) ---
        final_epsilon = self.epsilon
        num_actions = self.num_total_actions # Use pre-calculated total actions

        policy_entropy = 0.0
        # Check num_actions > 0 to avoid division by zero if strategy somehow yields 0 actions
        # Check num_actions > 1 because entropy is 0 for a deterministic policy (1 action)
        if num_actions > 1:
            prob_greedy = (1.0 - final_epsilon) + final_epsilon / num_actions
            prob_other = final_epsilon / num_actions

            # Use log2 for entropy in bits, or math.log for nats. Let's use math.log (nats).
            # Handle log(0) case: term is 0 if probability is 0. Add small epsilon for numerical stability.
            log_prob_greedy = math.log(prob_greedy) if prob_greedy > 1e-9 else 0.0
            log_prob_other = math.log(prob_other) if prob_other > 1e-9 else 0.0

            term1 = -prob_greedy * log_prob_greedy
            term2 = -(num_actions - 1) * prob_other * log_prob_other
            policy_entropy = term1 + term2
        # If num_actions <= 1, entropy is 0.0 (already initialized)


        # --- Final Metrics Calculation ---
        # Average cumulative reward across the entire batch for the episode
        batch_avg_cumulative_reward = episode_cumulative_reward.mean().item() if n_batch > 0 else 0.0

        finished_mask = (steps_to_threshold != -1) & (steps_to_threshold <= actual_max_steps)
        if torch.any(finished_mask):
            batch_avg_steps_to_threshold = steps_to_threshold[finished_mask].float().mean().item()
        else:
            batch_avg_steps_to_threshold = float("nan") # No agents finished within steps

        metrics = {
            "avg_cumulative_reward": batch_avg_cumulative_reward, # Renamed
            "avg_steps_to_threshold": batch_avg_steps_to_threshold,
            "n_finished": torch.sum(finished_mask).item(), # Renamed
            "n_total": n_batch,
            "policy_entropy": policy_entropy, # NEW
            "final_epsilon": self.epsilon, # Keep final epsilon
        }

        # Return the per-item cumulative rewards as well
        return {
            "final_maps": current_map_batch.cpu(),
            "final_cumulative_rewards": episode_cumulative_reward.cpu(), # NEW
            "metrics": metrics
        }

    # --- Save/Load State ---
    def save_state(self, filepath: str):
        """ Saves agent config and epsilon. Critic state is not saved. """
        # Ensure config includes num_total_actions if it wasn't saved before
        if 'num_total_actions' not in self.config:
            self.config['num_total_actions'] = self.num_total_actions

        state = {"config": self.config, "epsilon": self.epsilon}
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            with open(filepath, "w") as f: json.dump(state, f, indent=4)
            log.info(f"Agent state saved to {filepath}")
        except IOError as e: log.error(f"Error saving state to {filepath}: {e}")
        except Exception as e: log.error(f"Unexpected error saving state: {e}")

    def load_state(self, filepath: str):
        """ Loads agent epsilon from state file. Config mismatch triggers a warning. """
        try:
            with open(filepath, "r") as f: state = json.load(f)

            loaded_config = state.get("config", {})
            mismatched_keys = []
            critical_keys = ['strategy', 'percentage_change', 'initial_epsilon', 'epsilon_decay', 'min_epsilon', 'wide_greedy_sample_k']
            for k in critical_keys:
                 if self.config.get(k) != loaded_config.get(k):
                     mismatched_keys.append(k)
            if mismatched_keys:
                 log.warning(f"Loaded state config mismatch for keys: {mismatched_keys}. "
                             f"Current: { {k: self.config.get(k) for k in mismatched_keys} }, "
                             f"Loaded: { {k: loaded_config.get(k) for k in mismatched_keys} }. "
                             "Loading epsilon only.")

            self.epsilon = state.get("epsilon", self.config.get("initial_epsilon", 1.0))
            log.info(f"Agent state loaded from {filepath}. Epsilon set to {self.epsilon:.4f}")

        except FileNotFoundError: log.error(f"State file not found: {filepath}")
        except (IOError, json.JSONDecodeError) as e: log.error(f"Error reading state file {filepath}: {e}")
        except Exception as e: log.error(f"Unexpected error loading state: {e}")