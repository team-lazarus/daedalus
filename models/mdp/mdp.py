import torch
import torch.nn.functional as F
import random
import math
from enum import Enum
from typing import Tuple, Callable, List, Optional
import logging  # Import logging
from tqdm import trange  # Import tqdm specific range for loops
from rich.logging import RichHandler  # Import RichHandler
from rich.console import Console  # Import Console

import daedalus.models.mdp.constants as c

# --- Setup Logging ---
# Use RichHandler for pretty console logging
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
# Get the logger instance
log = logging.getLogger("rich")
# Create a console object for direct printing if needed
console = Console()


class MDPAgent:
    """
    An Epsilon-Greedy MDP agent with Epsilon Decay for map modification tasks,
    inspired by PCGRL. Includes tqdm progress and rich logging.

    The agent modifies maps based on a chosen strategy ('narrow', 'turtle', 'wide')
    considering both map and hero state via the critic function. It terminates
    when a certain percentage of unique tiles have been changed.
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
        """
        Initializes the MDPAgent.

        Args:
            size: Tuple (width, height) of the map.
            hero_tensor_size: The size (dimensionality) of the hero state tensor.
            critic: A callable function that takes a batch of map states
                    (N, X, Y) and hero states (N, H) and returns rewards (N,).
                    This function is key to how hero state influences actions.
            strategy: The modification strategy ("narrow", "turtle", "wide").
            percentage_change: The maximum percentage (0.0 to 1.0) of unique
                               map tiles allowed to be modified before termination.
            initial_epsilon: Starting value for epsilon in epsilon-greedy.
            epsilon_decay: Multiplicative factor to decay epsilon each step.
            min_epsilon: The minimum value epsilon can decay to.
        """
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

        # Epsilon greedy parameters are already part of the class
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Determine device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"MDPAgent using device: {self.device}")  # Use logger

        # Define action space sizes based on strategy
        self.num_tile_change_actions = 7  # 0: no-op, 1: empty, 2-5: enemies, 6: door
        if self.strategy == "turtle":
            self.num_actions_turtle_move = 4
            self.num_actions_turtle_change = self.num_tile_change_actions
            self.total_turtle_actions = (
                self.num_actions_turtle_move + self.num_actions_turtle_change
            )
        elif self.strategy == "wide":
            self.num_wide_change_actions = self.num_tile_change_actions - 1  # 1 to 6
            self.total_wide_actions = (
                self.map_width * self.map_height * self.num_wide_change_actions
            )
        else:  # narrow
            self.num_actions_narrow = self.num_tile_change_actions

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

        Args:
            current_map_batch: The current batch of map states (N, X, Y).
            current_hero_batch: The current batch of hero states (N, H). This state
                                influences the critic's evaluation.
            agent_pos_batch: Current agent positions for Turtle strategy (N, 2).
            current_coords: Current coordinates (x, y) for Narrow strategy.

        Returns:
            A tensor (N,) containing the chosen greedy action index for each item.
        """
        n_batch = current_map_batch.size(0)
        best_actions = torch.zeros(n_batch, dtype=torch.long, device=self.device)

        # The critic function implicitly uses the hero_batch to determine rewards,
        # thus influencing the greedy action selection.
        with torch.no_grad():
            # --- Narrow Strategy ---
            if self.strategy == "narrow":
                assert current_coords is not None
                x, y = current_coords
                possible_next_maps = []
                # Pass current_hero_batch to critic for baseline reward
                baseline_reward = self.critic(current_map_batch, current_hero_batch)

                for action_idx in range(1, self.num_actions_narrow):
                    next_map_batch = current_map_batch.clone()
                    next_map_batch[:, x, y] = action_idx
                    possible_next_maps.append(next_map_batch)

                if not possible_next_maps:
                    return torch.zeros(n_batch, dtype=torch.long, device=self.device)

                simulated_maps = torch.stack(possible_next_maps, dim=1).view(
                    -1, self.map_width, self.map_height
                )
                # Repeat hero batch - hero state is assumed constant for these simulations
                simulated_heroes = current_hero_batch.repeat_interleave(
                    len(possible_next_maps), dim=0
                )

                # Critic evaluates based on simulated map AND the constant hero state
                rewards = self.critic(simulated_maps, simulated_heroes)
                rewards = rewards.view(n_batch, -1)

                best_sim_action_indices = torch.argmax(rewards, dim=1)
                best_sim_rewards = torch.gather(
                    rewards, 1, best_sim_action_indices.unsqueeze(1)
                ).squeeze(1)

                take_best_sim_action = best_sim_rewards > baseline_reward
                best_actions = torch.where(
                    take_best_sim_action, best_sim_action_indices + 1, c.NO_ACTION
                )

            # --- Turtle Strategy ---
            elif self.strategy == "turtle":
                assert agent_pos_batch is not None
                possible_rewards = []

                # 1. Simulate Movement Actions (0-3)
                # Assuming move actions don't change the map/hero state relevant to the critic directly
                # The reward comes from the state *after* the move. If critic only sees map/global hero,
                # reward might seem unchanged unless position matters to critic.
                current_state_reward = self.critic(
                    current_map_batch, current_hero_batch
                )
                for move_action in range(self.num_actions_turtle_move):
                    # Simulate move (update position - not shown here as critic takes global map)
                    # Evaluate the state *as if* the move happened.
                    # If critic is position-aware, need to pass next_pos here.
                    possible_rewards.append(
                        current_state_reward
                    )  # Placeholder if critic ignores pos

                # 2. Simulate Tile Change Actions (4-10 mapped from original 0-6)
                # Action 4 (Turtle) == Action 0 (No-Op) -> use current state reward
                possible_rewards.append(current_state_reward)

                for change_action_idx in range(
                    1, self.num_actions_turtle_change
                ):  # Actions 1-6
                    next_map_batch = current_map_batch.clone()
                    x_coords = agent_pos_batch[:, 0]
                    y_coords = agent_pos_batch[:, 1]
                    batch_indices = torch.arange(n_batch, device=self.device)
                    next_map_batch[batch_indices, x_coords, y_coords] = (
                        change_action_idx
                    )
                    # Critic evaluates next map with current hero state
                    possible_rewards.append(
                        self.critic(next_map_batch, current_hero_batch)
                    )

                all_rewards = torch.stack(possible_rewards, dim=1)
                best_actions = torch.argmax(all_rewards, dim=1)

            # --- Wide Strategy ---
            elif self.strategy == "wide":
                # Use logger for warning
                log.warning(
                    "Wide strategy greedy search simulating all actions. This can be slow."
                )

                best_overall_actions = torch.zeros(
                    n_batch, dtype=torch.long, device=self.device
                )
                # Initialize max_rewards with the reward of doing nothing (baseline)
                # Although wide doesn't have an explicit NO_ACTION, this sets a floor
                max_rewards = self.critic(
                    current_map_batch, current_hero_batch
                )  # Baseline reward

                action_offset = 0
                for change_action_idx in range(
                    1, self.num_tile_change_actions
                ):  # Actions 1 to 6
                    for x in range(self.map_width):
                        for y in range(self.map_height):
                            current_flat_action = action_offset  # Track the flat index

                            next_map_batch = current_map_batch.clone()
                            next_map_batch[:, x, y] = change_action_idx
                            # Critic evaluates next map with current hero state
                            rewards = self.critic(next_map_batch, current_hero_batch)

                            is_better = rewards > max_rewards
                            best_overall_actions = torch.where(
                                is_better, current_flat_action, best_overall_actions
                            )
                            max_rewards = torch.where(is_better, rewards, max_rewards)
                            action_offset += 1

                # If no simulated action was better than baseline, should we force an action?
                # The definition implies Wide *always* makes a change. So we return the best *simulated* one found.
                # If max_rewards didn't increase beyond baseline, it means all simulated changes were worse
                # than doing nothing, but Wide strategy forces a change anyway.
                best_actions = best_overall_actions

        return best_actions

    def _select_action(
        self,
        current_map_batch: torch.Tensor,
        current_hero_batch: torch.Tensor,  # Pass hero state here too
        agent_pos_batch: Optional[torch.Tensor] = None,
        current_coords: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Selects an action for each item in the batch using epsilon-greedy.
        """
        n_batch = current_map_batch.size(0)
        sample = torch.rand(n_batch, device=self.device)
        explore_mask = sample < self.epsilon

        # Get greedy actions (which considers hero state via the critic)
        greedy_actions = self._get_greedy_action(
            current_map_batch, current_hero_batch, agent_pos_batch, current_coords
        )

        # Generate random actions for exploration
        if self.strategy == "narrow":
            random_actions = torch.randint(
                0, self.num_actions_narrow, (n_batch,), device=self.device
            )
        elif self.strategy == "turtle":
            random_actions = torch.randint(
                0, self.total_turtle_actions, (n_batch,), device=self.device
            )
        elif self.strategy == "wide":
            random_actions = torch.randint(
                0, self.total_wide_actions, (n_batch,), device=self.device
            )
        else:
            raise NotImplementedError

        # Combine greedy and random actions based on explore_mask
        chosen_actions = torch.where(explore_mask, random_actions, greedy_actions)

        return chosen_actions

    # _apply_action remains the same as before, as it only modifies the map/agent pos
    def _apply_action(
        self,
        map_batch: torch.Tensor,
        action_batch: torch.Tensor,
        modified_mask_batch: torch.Tensor,
        agent_pos_batch: Optional[torch.Tensor] = None,  # For Turtle
        current_coords: Optional[Tuple[int, int]] = None,  # For Narrow
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Applies the chosen actions to the map batch and updates modification masks.
        (Code is identical to previous version)
        """
        next_map_batch = map_batch.clone()
        next_modified_mask_batch = modified_mask_batch.clone()
        next_agent_pos_batch = (
            agent_pos_batch.clone() if agent_pos_batch is not None else None
        )
        n_batch = map_batch.size(0)
        batch_indices = torch.arange(n_batch, device=self.device)

        if self.strategy == "narrow":
            assert current_coords is not None
            x, y = current_coords
            change_mask = action_batch > c.NO_ACTION
            if torch.any(change_mask):
                action_values = action_batch[change_mask]
                batch_idx_change = batch_indices[change_mask]
                next_map_batch[batch_idx_change, x, y] = action_values
                next_modified_mask_batch[batch_idx_change, x, y] = True

        elif self.strategy == "turtle":
            assert next_agent_pos_batch is not None
            current_x = next_agent_pos_batch[:, 0]
            current_y = next_agent_pos_batch[:, 1]

            move_mask = action_batch < self.num_actions_turtle_move
            if torch.any(move_mask):
                move_actions = action_batch[move_mask]
                batch_idx_move = batch_indices[move_mask]
                pos_to_update = next_agent_pos_batch[move_mask]
                up_mask, down_mask = (
                    move_actions == c.MOVE_UP,
                    move_actions == c.MOVE_DOWN,
                )
                left_mask, right_mask = (
                    move_actions == c.MOVE_LEFT,
                    move_actions == c.MOVE_RIGHT,
                )
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

            change_mask = action_batch >= self.num_actions_turtle_move
            if torch.any(change_mask):
                tile_values = action_batch[change_mask] - self.num_actions_turtle_move
                is_actual_change = tile_values > c.NO_ACTION
                if torch.any(is_actual_change):
                    batch_idx_change = batch_indices[change_mask][is_actual_change]
                    x_change = current_x[change_mask][is_actual_change]
                    y_change = current_y[change_mask][is_actual_change]
                    action_val_change = tile_values[is_actual_change]
                    next_map_batch[batch_idx_change, x_change, y_change] = (
                        action_val_change
                    )
                    next_modified_mask_batch[batch_idx_change, x_change, y_change] = (
                        True
                    )

        elif self.strategy == "wide":
            action_val_indices = action_batch % self.num_wide_change_actions
            tile_values = action_val_indices + 1
            flat_coords = action_batch // self.num_wide_change_actions
            target_x = flat_coords % self.map_width
            target_y = flat_coords // self.map_width
            next_map_batch[batch_indices, target_x, target_y] = tile_values
            next_modified_mask_batch[batch_indices, target_x, target_y] = True
        else:
            raise NotImplementedError

        return next_map_batch, next_modified_mask_batch, next_agent_pos_batch

    def train_episode(
        self,
        initial_map_state: torch.Tensor,
        initial_hero_state: torch.Tensor,
        max_steps: int = 1000,
    ) -> torch.Tensor:
        """
        Runs a single training "episode" for a batch of maps with progress bar.

        Args:
            initial_map_state: Tensor of shape [N, X, Y].
            initial_hero_state: Tensor of shape [N, H].
            max_steps: The maximum number of steps to run the episode for.

        Returns:
            A tensor of shape [N, X, Y] representing the final map states.
        """
        n_batch = initial_map_state.size(0)
        log.info(
            f"Starting episode for {n_batch} instances. Max steps: {max_steps}. Strategy: {self.strategy}"
        )

        current_map_batch = initial_map_state.clone().to(self.device)
        current_hero_batch = initial_hero_state.clone().to(
            self.device
        )  # Keep hero state on device

        modified_mask_batch = torch.zeros_like(
            current_map_batch, dtype=torch.bool, device=self.device
        )
        num_modified_tiles = torch.zeros(n_batch, dtype=torch.long, device=self.device)
        active_mask = torch.ones(n_batch, dtype=torch.bool, device=self.device)

        agent_pos_batch = None
        narrow_x, narrow_y = 0, 0
        if self.strategy == "turtle":
            agent_pos_batch = torch.zeros(
                (n_batch, 2), dtype=torch.long, device=self.device
            )
        elif self.strategy == "narrow":
            narrow_x, narrow_y = 0, 0

        # --- Training Loop with tqdm ---
        # Use trange for iteration with progress bar
        pbar = trange(max_steps, desc="Episode Progress", leave=True)
        final_step = max_steps
        for step in pbar:
            if not torch.any(active_mask):
                final_step = step  # Record when all instances finished
                break  # Exit loop if no instances are active

            active_indices = torch.where(active_mask)[0]
            active_maps = current_map_batch[active_mask]
            active_heroes = current_hero_batch[
                active_mask
            ]  # Use the corresponding active hero states
            active_agent_pos = (
                agent_pos_batch[active_mask] if agent_pos_batch is not None else None
            )
            active_mod_masks = modified_mask_batch[active_mask]

            # 1. Select Action (passing map and hero state)
            current_narrow_coords = (
                (narrow_x, narrow_y) if self.strategy == "narrow" else None
            )
            actions = self._select_action(
                active_maps,
                active_heroes,  # Pass hero state here
                active_agent_pos,
                current_narrow_coords,
            )

            # 2. Apply Action & Update State
            next_maps, next_mod_masks, next_agent_pos = self._apply_action(
                active_maps,
                actions,
                active_mod_masks,
                active_agent_pos,
                current_narrow_coords,
            )

            # 3. Update global state tensors
            current_map_batch[active_mask] = next_maps
            modified_mask_batch[active_mask] = next_mod_masks
            if agent_pos_batch is not None and next_agent_pos is not None:
                agent_pos_batch[active_mask] = next_agent_pos

            # 4. Update modification counts and check termination
            current_mod_counts = modified_mask_batch[active_mask].sum(dim=(1, 2))
            num_modified_tiles[active_mask] = current_mod_counts
            terminated_mask = num_modified_tiles >= self.max_modifications
            newly_terminated = (
                active_mask & terminated_mask
            )  # Identify which just terminated
            active_mask = active_mask & ~terminated_mask  # Update active mask

            # 5. Decay Epsilon
            self._decay_epsilon()

            # 6. Advance Narrow Strategy Position
            if self.strategy == "narrow":
                narrow_y += 1
                if narrow_y >= self.map_height:
                    narrow_y = 0
                    narrow_x += 1
                    if narrow_x >= self.map_width:
                        log.info("Narrow strategy completed full map scan.")
                        active_mask[:] = False  # Stop all after one full scan

            # Update tqdm progress bar description
            pbar.set_description(
                f"Step {step+1}/{max_steps} | "
                f"Active: {torch.sum(active_mask).item()}/{n_batch} | "
                f"Epsilon: {self.epsilon:.4f}"
            )
        else:  # Loop finished without break (i.e., reached max_steps)
            final_step = max_steps

        pbar.close()  # Close the progress bar
        log.info(f"Episode finished after {final_step} steps.")
        return current_map_batch.cpu()
