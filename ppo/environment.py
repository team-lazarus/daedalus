# environment.py
import numpy as np
import torch
import random
from typing import Tuple, Optional, Dict, Any, Callable

from config import TrainConfig
from utils import generate_initial_map, create_random_hero_tensor


class MapEnvironment:
    """Simulates the 12x12 map environment."""

    def __init__(self, config: TrainConfig, device: torch.device):
        """Initializes the environment."""
        self.config = config
        self.rows, self.cols = config.map_size
        self.device = device
        self.current_map: Optional[np.ndarray] = None
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.hero_tensor: Optional[torch.Tensor] = None
        self.action_size = config.get_action_size()

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resets the environment to a new initial state."""
        self.current_map, start_pos = generate_initial_map(
            self.config.map_size, self.config.random_walk_steps
        )
        self.agent_pos = start_pos
        self.hero_tensor = create_random_hero_tensor(
            self.config.hero_tensor_size, self.device
        ).unsqueeze(
            0
        )  # Add batch dim for model

        map_tensor = (
            torch.from_numpy(self.current_map)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )  # Add batch and channel dim
        return map_tensor, self.hero_tensor

    def _apply_map_modification(self, pos: Tuple[int, int], mod_type: int) -> None:
        """Applies a map modification action."""
        r, c = pos
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.current_map[r, c] = mod_type

    def _move_agent(self, direction: int) -> None:
        """Moves the agent, handling wrapping for turtle mode."""
        r, c = self.agent_pos
        if direction == 8:  # Up
            r -= 1
        elif direction == 9:  # Left
            c -= 1
        elif direction == 10:  # Down
            r += 1
        elif direction == 11:  # Right
            c += 1

        # Handle wrapping for turtle mode
        if self.config.mode == "turtle":
            r %= self.rows
            c %= self.cols
        else:  # For narrow mode automatic movement (or other future modes)
            r = np.clip(r, 0, self.rows - 1)
            c = np.clip(c, 0, self.cols - 1)

        self.agent_pos = (r, c)

    def _get_next_narrow_position(self) -> None:
        """Calculates the agent's next position automatically in narrow mode."""
        # Simple example: move right, wrap around rows
        r, c = self.agent_pos
        c += 1
        if c >= self.cols:
            c = 0
            r = (r + 1) % self.rows
        self.agent_pos = (r, c)

    def step(self, action: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Performs one step in the environment based on the action and mode.
        Returns the new map tensor, hero tensor, and the previous map state.
        Reward calculation is externalized. Done is always False for this setup.
        """
        if (
            self.current_map is None
            or self.agent_pos is None
            or self.hero_tensor is None
        ):
            raise RuntimeError("Environment must be reset before stepping.")

        prev_map_state = self.current_map.copy()  # Store previous state for reward calc

        if self.config.mode == "narrow":
            if 0 <= action <= 6:  # Map modification actions
                self._apply_map_modification(self.agent_pos, action)
            elif action == 7:  # No-action
                pass
            else:
                raise ValueError(f"Invalid narrow action: {action}")
            # Agent moves automatically after action in narrow mode
            self._get_next_narrow_position()

        elif self.config.mode == "turtle":
            if 0 <= action <= 6:  # Map modification actions
                self._apply_map_modification(self.agent_pos, action)
            elif 8 <= action <= 11:  # Movement actions
                self._move_agent(action)
            elif (
                action == 7
            ):  # Explicit no-op might be needed if action_size=12 but 7 unused
                pass  # Or handle based on exact action mapping
            else:
                raise ValueError(f"Invalid turtle action: {action}")

        elif self.config.mode == "wide":
            # Decode action: action = tile_index * 6 + mod_type
            if not (0 <= action < self.action_size):
                raise ValueError(f"Invalid wide action: {action}")

            mod_type = action % 6
            tile_index = action // 6
            n = self.rows  # Assuming square map
            row = tile_index // n
            col = tile_index % n
            self._apply_map_modification((row, col), mod_type)
            # Agent position might not change in wide mode, or has separate logic
            # Assuming position doesn't change automatically here unless specified

        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

        # Update hero tensor (example: decrement rooms_left, could be more complex)
        # Ensure gradient tracking is handled correctly if hero updates depend on model
        with torch.no_grad():
            new_hero = self.hero_tensor.clone()
            new_hero[0, 4] = max(0, new_hero[0, 4] - 0.01)  # Example decrement
            self.hero_tensor = new_hero

        new_map_tensor = (
            torch.from_numpy(self.current_map)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        return new_map_tensor, self.hero_tensor, prev_map_state

    def get_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the current state (map tensor, hero tensor)."""
        if self.current_map is None or self.hero_tensor is None:
            raise RuntimeError("Environment not initialized.")
        map_tensor = (
            torch.from_numpy(self.current_map)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        return map_tensor, self.hero_tensor
