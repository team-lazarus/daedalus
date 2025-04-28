import torch
import torch.nn as nn
import random
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional, List, Union

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskID,
)
from rich import print as rprint
from tqdm import tqdm


# Assuming Daedalus constants and critic model are correctly imported
import daedalus.utils.constants as c
from daedalus.critics.critic_approximator import CriticApproximatorMLP

console = Console()


def visualize_maps(map, x=0, y=0, title: str = "- Map 0 -") -> None:
    """Visualize maps using rich."""

    color_map = {
        0: "[grey27]0[/grey27]",  # Empty/Wall
        1: "[white]1[/white]",  # Path
        2: "[red]2[/red]",  # Enemy
        3: "[red]3[/red]",  # Damaged Enemy?
        4: "[red]4[/red]",  # Dead Enemy?
        5: "[red]5[/red]",  # Player (if present)
        6: "[green]6[/green]",  # Door
    }

    console.print(f"\n{title}")

    table = Table(
        title=title,
        show_header=False,
        show_lines=False,
        box=None,
        padding=0,
    )

    map_height, map_width = map.shape[0], map.shape[1]
    for _ in range(map_width):
        table.add_column()  # No header text needed

    for j in range(map_height):
        row = []
        for k in range(map_width):
            cell_value = int(map[j, k].item())
            if j == x and k == y:
                row.append(f"[yellow]{cell_value}[/yellow]")
            row.append(color_map.get(cell_value, f"[cyan]{cell_value}[/cyan]"))
        table.add_row(*row)

    console.print(table)
    rprint("")  # Use rich print for spacing


class DaedalusEnvironment:
    """
    Daedalus environment compatible with the provided PPO agent.

    Handles multiple parallel environments (batch_size).
    Interacts using standard Python/NumPy types and PyTorch tensors,
    returning observations, rewards, dones, truncated flags, and info.
    """

    def __init__(
        self,
        mode: str,
        batch_size: int,  # Changed type hint
        device: str,
        *,
        map_size: Tuple[int, int] = (12, 12),
        critic_path: Optional[str] = None,
        max_steps: int = 256,
    ):  # Default matches PPO agent steps_per_episode

        if mode.upper() not in c.POSSIBLE_MODES:
            raise ValueError(f"Mode of environment has to be one of {c.POSSIBLE_MODES}")

        self.mode = mode.upper()
        self.device = torch.device(device)  # Use torch.device
        self.batch_size = batch_size  # Store as int
        self.map_size = map_size
        self.max_steps = max_steps

        # Environment state variables (batched)
        self.maps: Optional[torch.Tensor] = None
        self.current_positions: Optional[torch.Tensor] = None
        self.consecutive_moves: Optional[torch.Tensor] = (
            None  # Track consecutive moves for turtle mode
        )
        self.step_count: Optional[torch.Tensor] = None

        # Calculate observation dimension
        self.obs_dim = (map_size[0] * map_size[1]) + 2 + c.HERO_TENSOR_SIZE

        # Calculate action space size based on mode
        if self.mode == "NARROW":
            # Modifications + No Action
            self.action_space = c.MODIFICATION_ACTIONS + len(c.NO_ACTION)
        elif self.mode == "TURTLE":
            # Modifications + Movement Actions
            self.action_space = c.MODIFICATION_ACTIONS + len(c.MOVE_ACTION)
            print(self.action_space)
        elif self.mode == "WIDE":
            # Modification for every cell
            self.action_space = c.MODIFICATION_ACTIONS * map_size[0] * map_size[1]
        else:
            raise ValueError(
                f"Unknown mode: {self.mode}"
            )  # Should be caught earlier, but safety check

        # Initialize critic approximator if path is provided
        self.critic = None
        if critic_path and os.path.exists(critic_path):
            self.critic = CriticApproximatorMLP(
                input_size=map_size[0] * map_size[1], hidden_sizes=[256, 128]
            )
            self.critic.to(device)
            checkpoint = torch.load(critic_path, weights_only=False)
            self.critic.load_state_dict(checkpoint["model_state_dict"])
            self.critic.eval()
        elif critic_path:
            print(
                f"Critic path specified ({critic_path}), but file not found. Critic will not be used."
            )

        # Initialize state variables on the first reset
        self._generate_maps()
        self._initialize_state()

    def _generate_maps(self, seed: Optional[int] = None):
        self.bootstrapped_maps = []
        if seed is not None:
            # Note: This seeds the global random state.
            # For perfectly isolated seeding per reset, more complex handling is needed.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)

        console.print(
            "[bold][cyan]= Initializing a massive batch of maps =[/cyan][/bold]"
        )
        for i in tqdm(range(self.batch_size * 128)):
            # Randomly choose a generation algorithm
            map_data = self._apply_random_walk(i)

            # Set random starting position for this environment
            start_row = random.randint(0, self.map_size[0] - 1)
            start_col = random.randint(0, self.map_size[1] - 1)

            self.bootstrapped_maps.append((start_row, start_col, map_data))

    def _initialize_state(self):
        """Initializes or resets the core state tensors."""
        self.maps = torch.zeros(
            (self.batch_size, self.map_size[0], self.map_size[1]),
            dtype=torch.int64,
            device=self.device,
        )
        self.current_positions = torch.zeros(
            (self.batch_size, 2), dtype=torch.int64, device=self.device
        )
        self.consecutive_moves = torch.zeros(
            self.batch_size, dtype=torch.int64, device=self.device
        )
        self.step_count = torch.zeros(
            self.batch_size, dtype=torch.int64, device=self.device
        )

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Resets all environments in the batch to an initial state.

        Args:
            seed: Optional random seed for reproducibility during reset.

        Returns:
            Initial observation tensor of shape (batch_size, obs_dim).
        """
        # Re-initialize all state tensors
        self._initialize_state()

        # Apply procedural generation and set initial positions for each environment
        samples = random.sample(self.bootstrapped_maps, self.batch_size)

        # Create lists to hold the map and position data
        maps_list = []
        positions_list = []

        for row, col, map_data in samples:
            # Check if map_data is already a tensor and move to CPU if needed
            if isinstance(map_data, torch.Tensor):
                map_data = map_data.cpu().numpy()
            maps_list.append(map_data)
            positions_list.append([row, col])

        # Convert lists to NumPy arrays, then to tensors, and finally move to the device
        self.maps = torch.tensor(
            np.array(maps_list, dtype=np.int64), device=self.device
        )
        self.current_positions = torch.tensor(
            np.array(positions_list, dtype=np.int64), device=self.device
        )

        # Create and return the initial observations
        initial_observations = self._create_observations()
        print("RESET")
        return initial_observations

    def step(
        self, actions: np.ndarray
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Execute one time step in all parallel environments.

        Args:
            actions: NumPy array of actions for each environment, shape (batch_size,).

        Returns:
            A tuple containing:
            - next_observation (torch.Tensor): Observations after taking the action, shape (batch_size, obs_dim).
            - reward (np.ndarray): Amount of reward received after the step, shape (batch_size,).
            - done (np.ndarray): Boolean array indicating if the episode has ended (terminal state), shape (batch_size,).
            - truncated (np.ndarray): Boolean array indicating if the episode was truncated (e.g., max steps), shape (batch_size,).
            - info (List[Dict]): List of dictionaries containing auxiliary diagnostic information (one dict per env).
        """
        if not isinstance(actions, np.ndarray):
            # If actions are tensor, convert to numpy on CPU
            actions = actions.cpu().numpy()

        actions = actions.reshape(-1)  # Ensure actions is 1D array

        if actions.shape[0] != self.batch_size:
            raise ValueError(
                f"Actions array size ({actions.shape[0]}) does not match batch size ({self.batch_size})"
            )

        # Increment step counter for all environments
        self.step_count += 1

        # Process actions for each environment
        rewards = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

        if self.mode == "TURTLE":
            #   self.maps         is a torch.Tensor of shape [B, H, W]
            #   self.current_positions is a torch.Tensor of shape [B, 2]  (row, col)
            #   actions           is a 1D tensor of length B
            #   c.MODIFICATION_ACTIONS == number of “paint” actions
            #
            # device is already defined

            # 1) pull out NumPy buffers
            maps_np = self.maps.cpu().detach().numpy()  # shape (B, H, W)
            pos_np = self.current_positions.cpu().detach().numpy()  # shape (B, 2)
            acts_np = actions  # shape (B,)

            B, H, W = maps_np.shape

            # 2) modification (painting) actions
            mod_mask = acts_np < c.MODIFICATION_ACTIONS  # shape (B,)
            batch_idx = np.nonzero(mod_mask)[0]  # e.g. [0,3,5,12,…]
            pos_mod = pos_np[mod_mask]  # shape (M,2)
            maps_np[batch_idx, pos_mod[:, 0], pos_mod[:, 1]] = acts_np[mod_mask]

            # 3) movement actions
            move_mask = ~mod_mask  # or acts_np >= c.MODIFICATION_ACTIONS
            rows, cols = (
                pos_np[:, 0].copy(),
                pos_np[:, 1].copy(),
            )  # make sure we’re not overwriting pos_np too early

            # action codes → vector masks
            up = acts_np == (c.MODIFICATION_ACTIONS + 0)  # e.g. 7
            left = acts_np == (c.MODIFICATION_ACTIONS + 1)  # e.g. 8
            down = acts_np == (c.MODIFICATION_ACTIONS + 2)  # e.g. 9
            right = acts_np == (c.MODIFICATION_ACTIONS + 3)  # e.g. 10

            # clamp into valid range [0 … H-1] or [0 … W-1]
            rows[up] = np.maximum(0, rows[up] - 1)
            rows[down] = np.minimum(H - 1, rows[down] + 1)
            cols[left] = np.maximum(0, cols[left] - 1)
            cols[right] = np.minimum(W - 1, cols[right] + 1)

            # write back
            pos_np[:, 0], pos_np[:, 1] = rows, cols

            # 4) push back into torch
            self.maps = torch.from_numpy(maps_np).to(device=self.device)
            self.current_positions = torch.from_numpy(pos_np).to(device=self.device)

            # 5) (optional) re-visualize
            visualize_maps(self.maps[0])
        else:
            raise NotImplementedError(f"{self.mode} has currently not been implemented")

        """
        for i in range(self.batch_size):
            act = actions[i]  # Get action for the i-th environment
            current_row, current_col = self.current_positions[i]

            if self.mode == "NARROW":
                if act < c.MODIFICATION_ACTIONS:  # Modification action
                    self.maps[i, current_row, current_col] = act
                    self.consecutive_moves[i] = (
                        0  # Reset consecutive moves if applicable
                    )
                # Action >= c.MODIFICATION_ACTIONS corresponds to NO_ACTION, do nothing

            elif self.mode == "TURTLE":
                if act < c.MODIFICATION_ACTIONS:  # Modification action
                    self.maps[i, current_row, current_col] = act
                    self.consecutive_moves[i] = 0  # Reset consecutive moves counter
                    
                else:  # Movement action (index >= c.MODIFICATION_ACTIONS)
                    if i == 0:
                        pass
                        #print(current_row, current_col)
                        #visualize_maps(self.maps[i], current_row, current_col)
                    move_index = act # Adjust index for MOVE_ACTION dict
                    if move_index in c.MOVE_ACTION:
                        self.consecutive_moves[i] += 1
                        move_func = c.MOVE_ACTION[move_index]
                        # Ensure indices are Python ints for the function
                        new_row, new_col = move_func(
                            current_row.item(),
                            current_col.item(),
                            self.map_size[1],
                        )
                        if i == 0:
                            print("new:",new_row, new_col)
                        self.current_positions[i, 0] = new_row
                        self.current_positions[i, 1] = new_col

                        # Apply punishment for repeated movement
                        if self.consecutive_moves[i] > 5:
                            # Exponential punishment increases quickly
                            punishment = -(
                                1.05 ** (self.consecutive_moves[i].item() - 5)
                            )
                            rewards[i] += punishment
                    else:
                        # Handle invalid action index if necessary
                        # print(f"Warning: Invalid move action index {move_index} for TURTLE mode.")
                        pass

            elif self.mode == "WIDE":
                # WIDE mode action encodes both modification type and cell location
                map_size_prod = self.map_size[0] * self.map_size[1]
                if act < c.MODIFICATION_ACTIONS * map_size_prod:
                    modification_type = act % c.MODIFICATION_ACTIONS
                    cell_index = act // c.MODIFICATION_ACTIONS

                    # Convert flat cell index to 2D coordinates
                    target_row = cell_index // self.map_size[1]
                    target_col = cell_index % self.map_size[1]

                    # Apply modification if coordinates are valid
                    if (
                        0 <= target_row < self.map_size[0]
                        and 0 <= target_col < self.map_size[1]
                    ):
                        self.maps[i, target_row, target_col] = modification_type
                        # Optionally reset consecutive moves if relevant for WIDE mode
                        # self.consecutive_moves[i] = 0
                # else: Handle invalid action index if necessary
        """

        # --- Calculate Rewards ---
        # Add rewards from critic if available
        if self.critic is not None:
            with torch.no_grad():  # Ensure no gradients are calculated for critic
                # Critic expects input shape like (batch, channels, height, width)
                # or (batch, features) depending on its architecture.
                # Assuming MLP critic expects flattened maps:
                x = torch.unsqueeze(self.maps, dim=1).float()
                critic_rewards = self.critic(x).squeeze(
                    -1
                )  # Remove trailing dim if present
                rewards += critic_rewards

        # --- Check for Termination and Truncation ---
        # Termination: Usually based on environment-specific goals (not implemented here)
        # For now, 'done' means the episode ended naturally. We'll set it to False.
        # If there were specific win/loss conditions, they would set 'dones'.
        dones = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        # Truncation: Episode ends due to reaching the step limit
        truncateds = self.step_count >= self.max_steps

        # --- Create Next Observations ---
        next_observations = self._create_observations()

        # --- Prepare Return Values ---
        # Convert rewards, dones, truncateds to NumPy arrays for standard interface
        rewards_np = rewards.cpu().numpy()
        dones_np = dones.cpu().numpy()
        truncateds_np = truncateds.cpu().numpy()

        # Create info list (can be empty dictionaries)
        infos = [{} for _ in range(self.batch_size)]

        # --- Handle Environment Resets on Done/Truncated ---
        # If an environment is done or truncated, reset its state for the next episode
        # This is crucial for continuous training with parallel environments
        reset_indices = torch.where(dones | truncateds)[0]
        if len(reset_indices) > 0:
            # print(f"Resetting environments at indices: {reset_indices.tolist()}") # Debugging
            for idx in reset_indices:
                self.step_count[idx] = 0
                self.consecutive_moves[idx] = 0  # Reset turtle counter

                # Reset map and position
                gen_algorithm = random.choice(["random_walk", "connected_squares"])
                if gen_algorithm == "random_walk":
                    self._apply_random_walk(idx.item())
                else:
                    self._apply_connected_squares(idx.item())

                start_row = random.randint(0, self.map_size[0] - 1)
                start_col = random.randint(0, self.map_size[1] - 1)
                self.current_positions[idx, 0] = start_row
                self.current_positions[idx, 1] = start_col

                # Overwrite the observation for the reset environments with the new initial state
                next_observations[idx] = self._create_observation_for_index(idx.item())

        return next_observations, rewards_np / 100, dones_np, truncateds_np, infos

    def _apply_random_walk(self, batch_idx: int):
        """Apply random walk algorithm to generate a map for a specific batch index."""
        # Clear the existing map for this index first
        pcgrl_map = torch.zeros(
            (self.map_size[0], self.map_size[1]),
            dtype=torch.int64,
            device=self.device,
        )

        steps = random.randint(96, 128)
        row, col = random.randint(0, self.map_size[0] - 1), random.randint(
            0, self.map_size[1] - 1
        )

        for _ in range(steps):
            # Place a tile
            tile_choice = random.random()
            if tile_choice < 0.90:  # 80% empty # 90% removing doors
                tile_type = c.TILE_EMPTY
            # elif tile_choice < 0.84:  # 4% door
            #     tile_type = c.TILE_DOOR
            else:  # 16% enemy # 10% chance
                tile_type = c.ENEMY_TILES[0]
            pcgrl_map[row, col] = tile_type

            # Move randomly
            direction = random.randint(0, 3)  # 0: up, 1: down, 2: left, 3: right
            if direction == 0 and row > 1:
                row -= 1
            elif direction == 1 and row < self.map_size[0] - 2:
                row += 1
            elif direction == 2 and col > 1:
                col -= 1
            elif direction == 3 and col < self.map_size[1] - 2:
                col += 1

        return pcgrl_map

    def _apply_connected_squares(self, batch_idx: int):
        """Generate a map with connected squares for a specific batch index."""
        # Clear the existing map for this index first
        self.maps[batch_idx].zero_()

        num_squares = random.randint(3, 8)
        min_size, max_size = 2, 4
        generated_squares = []  # Store center points of generated squares

        for sq_idx in range(num_squares):
            width = random.randint(min_size, max_size)
            height = random.randint(min_size, max_size)
            start_row = random.randint(
                0, self.map_size[0] - height
            )  # Adjust to fit height
            start_col = random.randint(
                0, self.map_size[1] - width
            )  # Adjust to fit width

            # Fill the square
            for r in range(start_row, start_row + height):
                for col in range(start_col, start_col + width):
                    if random.random() < 0.9:  # 90% empty
                        tile_type = c.TILE_EMPTY
                    else:  # 10% potential enemy or door
                        if random.random() < 0.75:  # 7.5% enemy
                            tile_type = random.choice(c.ENEMY_TILES)
                        else:  # 2.5% door
                            tile_type = c.TILE_DOOR
                    self.maps[batch_idx, r, col] = tile_type

            # Store center of the current square
            center_row = start_row + height // 2
            center_col = start_col + width // 2
            current_center = (center_row, center_col)

            # If not the first square, connect to a random previous square center
            if generated_squares:
                prev_center_row, prev_center_col = random.choice(generated_squares)
                self._connect_points(
                    batch_idx, center_row, center_col, prev_center_row, prev_center_col
                )

            generated_squares.append(current_center)

    def _connect_points(self, batch_idx: int, r1: int, c1: int, r2: int, c2: int):
        """Connects two points (r1, c1) and (r2, c2) with empty tiles (corridor)."""
        row, col = r1, c1
        while row != r2 or col != c2:
            # Move horizontally first, then vertically (L-shape corridor)
            if col != c2:
                col += 1 if col < c2 else -1
            elif row != r2:
                row += 1 if row < r2 else -1

            # Ensure coordinates are within bounds before placing tile
            if 0 <= row < self.map_size[0] and 0 <= col < self.map_size[1]:
                # Only place corridor if the tile isn't already something important (like a door/enemy)
                # Or simply overwrite to ensure connection
                self.maps[batch_idx, row, col] = c.TILE_EMPTY

    def _create_observations(self) -> torch.Tensor:
        """
        Create observation tensor for the entire batch from the current environment state.

        Returns:
            Observation tensor of shape (batch_size, obs_dim).
        """
        # Pre-allocate observation tensor on the correct device
        obs = torch.zeros(
            (self.batch_size, self.obs_dim), dtype=torch.float32, device=self.device
        )

        # Flattened maps part (batch_size, map_height * map_width)
        flat_map_size = self.map_size[0] * self.map_size[1]
        obs[:, :flat_map_size] = self.maps.view(self.batch_size, -1).float()

        # Current positions part (batch_size, 2)
        obs[:, flat_map_size] = self.current_positions[:, 0].float()  # Row
        obs[:, flat_map_size + 1] = self.current_positions[:, 1].float()  # Column

        # Hero data part (batch_size, HERO_TENSOR_SIZE)
        # Assuming hero_data is all zeros for now. If hero data varies per batch,
        # it should be a tensor of shape (batch_size, HERO_TENSOR_SIZE).
        # hero_data = torch.zeros((self.batch_size, c.HERO_TENSOR_SIZE), dtype=torch.float32, device=self.device)
        # obs[:, flat_map_size + 2 : flat_map_size + 2 + c.HERO_TENSOR_SIZE] = hero_data
        # If hero_data is constant zeros, the pre-allocation handles it.

        return obs

    def _create_observation_for_index(self, batch_idx: int) -> torch.Tensor:
        """
        Create observation tensor for a single environment index.

        Args:
            batch_idx: The index of the environment in the batch.

        Returns:
            Observation tensor for the specified index, shape (obs_dim,).
        """
        obs = torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device)
        flat_map_size = self.map_size[0] * self.map_size[1]

        # Flattened map
        obs[:flat_map_size] = self.maps[batch_idx].flatten().float()

        # Current position
        obs[flat_map_size] = self.current_positions[batch_idx, 0].float()
        obs[flat_map_size + 1] = self.current_positions[batch_idx, 1].float()

        # Hero data (assuming zeros)
        # hero_data = torch.zeros(c.HERO_TENSOR_SIZE, dtype=torch.float32, device=self.device)
        # obs[flat_map_size + 2 : flat_map_size + 2 + c.HERO_TENSOR_SIZE] = hero_data

        return obs


# Example usage (optional, for testing)
if __name__ == "__main__":
    env_mode = "TURTLE"
    batch_s = 4
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    max_s = 50

    env = DaedalusEnvironment(
        mode=env_mode,
        batch_size=batch_s,
        device=dev,
        max_steps=max_s,
        map_size=(10, 10),  # Smaller map for easier visualization
    )

    print(f"Environment Mode: {env.mode}")
    print(f"Batch Size: {env.batch_size}")
    print(f"Observation Dim: {env.obs_dim}")
    print(f"Action Space Size: {env.action_space}")
    print(f"Device: {env.device}")
    print(f"Max Steps: {env.max_steps}")

    # Test reset
    print("\nTesting Reset...")
    initial_obs = env.reset()
    print("Initial Obs Shape:", initial_obs.shape)
    print("Initial Obs Device:", initial_obs.device)
    # print("Initial Maps Sample (Env 0):\n", env.maps[0].cpu().numpy())
    # print("Initial Positions Sample (Env 0):", env.current_positions[0].cpu().numpy())

    # Test step
    print("\nTesting Step...")
    # Sample random actions for the batch
    random_actions = np.random.randint(0, env.action_space, size=env.batch_size)
    print("Taking Random Actions:", random_actions)

    next_obs, rewards, dones, truncateds, infos = env.step(random_actions)

    print("Next Obs Shape:", next_obs.shape)
    print("Rewards:", rewards)
    print("Dones:", dones)
    print("Truncateds:", truncateds)
    # print("Infos:", infos)
    # print("Maps after step (Env 0):\n", env.maps[0].cpu().numpy())
    # print("Positions after step (Env 0):", env.current_positions[0].cpu().numpy())
    # print("Step Count:", env.step_count.cpu().numpy())

    # Run a few steps
    print("\nRunning 5 steps...")
    for i in range(5):
        random_actions = np.random.randint(0, env.action_space, size=env.batch_size)
        next_obs, rewards, dones, truncateds, infos = env.step(random_actions)
        print(
            f"Step {i+1}: Rewards: {rewards}, Truncateds: {truncateds}, StepCount: {env.step_count.cpu().numpy()}"
        )
        if np.any(dones) or np.any(truncateds):
            print(f"  -> Episode ended for some environments at step {i+1}.")
            # If you want to see resets happening, check the step counts after this loop

    print("\nTesting step limit truncation...")
    env.reset()  # Start fresh
    for i in range(max_s + 2):  # Go slightly over max_steps
        random_actions = np.random.randint(0, env.action_space, size=env.batch_size)
        next_obs, rewards, dones, truncateds, infos = env.step(random_actions)
        if i == max_s - 2:
            print(
                f"Step {i+1} (before truncation): Truncateds: {truncateds}, StepCount: {env.step_count.cpu().numpy()}"
            )
        if i == max_s - 1:
            print(
                f"Step {i+1} (at truncation): Truncateds: {truncateds}, StepCount: {env.step_count.cpu().numpy()}"
            )
        if i == max_s:
            print(
                f"Step {i+1} (after truncation): Truncateds: {truncateds}, StepCount: {env.step_count.cpu().numpy()}"
            )  # Should have reset step count
