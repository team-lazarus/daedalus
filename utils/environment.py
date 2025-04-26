import torch
from torchrl.envs.common import EnvBase
from torchrl.data import (
    Composite, 
    Unbounded, 
    Categorical, 
    Binary,
)
from tensordict.tensordict import TensorDict

from typing import Tuple, Dict, Any
import random
import numpy as np
import os

import daedalus.utils.constants as c
from daedalus.critics.critic_approximator import CriticApproximatorMLP

class DaedalusEnvironment(EnvBase):
    def __init__(self, mode: str, batch_size: int, device: str, *, map_size: Tuple[int, int] = (12,12), 
                 critic_path: str = None, max_steps: int = 1024):
        if mode.upper() not in c.POSSIBLE_MODES:
            raise ValueError(f"Mode of environment has to be {c.POSSIBLE_MODES}")

        super().__init__(device=device, batch_size=[batch_size])

        self.mode = mode.upper()
        self._device = device
        self.batch_size = torch.Size([batch_size])
        self.map_size = map_size
        self.max_steps = max_steps
        
        # Current position of the agent (for NARROW and TURTLE modes)
        self.current_positions = None
        
        # Track number of consecutive movement actions for turtle mode
        self.consecutive_moves = None
        
        # Step counter
        self.step_count = None
        
        # Initialize critic approximator
        self.critic = None
        if critic_path:
            self.critic = CriticApproximatorMLP(input_size=map_size[0]*map_size[1], hidden_sizes=[256, 128])
            self.critic.to(device)
            checkpoint = torch.load(critic_path, weights_only=False)
            self.critic.load_state_dict(checkpoint["model_state_dict"])
            self.critic.eval()


        self.obs_dim = (map_size[0] * map_size[1]) + 2 + c.HERO_TENSOR_SIZE
        if self.mode == "NARROW":
            self.action_space = c.MODIFICATION_ACTIONS + len(c.NO_ACTION)
        elif self.mode == "TURTLE":
            self.action_space = c.MODIFICATION_ACTIONS + len(c.MOVE_ACTION)
        elif self.mode == "WIDE":
            self.action_space = c.MODIFICATION_ACTIONS * map_size[0] * map_size[1]

        self.observation_spec = Composite({
            "observation": Unbounded(
                shape=(batch_size, self.obs_dim),
                dtype=torch.float32,
                device=self.device,
            )
        }, shape=self.batch_size)

        self.action_spec = Categorical(
            n=self.action_space,
            shape=(batch_size,1),
            dtype=torch.int64,
            device=self.device
        )
        self.action_spec = self.action_spec.expand(self.batch_size)

        self.reward_spec = Unbounded(
            shape=(batch_size, 1),
            dtype=torch.float32,
            device=self.device
        )
        self.reward_spec = self.reward_spec.expand(self.batch_size)

        self.done_spec = Binary(
            shape=(batch_size, 1),
            device=self.device
        )
        self.done_spec = self.done_spec.expand(self.batch_size)
        
        # Initialize current maps
        self.maps = None

    def _reset(self, indices=None):
        """
        Reset the environment state.
        
        Args:
            indices: Optional indices to reset specific batch items
            
        Returns:
            TensorDict containing observation after reset
        """
        if indices is None or torch.is_tensor(indices[0]):
            indices = torch.arange(self.batch_size[0], device=self.device)
            
        # Initialize maps if not already done or reset specified indices
        if self.maps is None:
            self.maps = torch.zeros(self.batch_size + self.map_size, dtype=torch.int64, device=self.device)
        else:
            # Only reset specified indices
            new_indices = []
            for i,idx_dict in enumerate(indices):
                if torch.is_tensor(idx_dict):
                    idx = idx_dict
                else:
                    idx = idx_dict["_reset"]
                    if torch.equal(idx, torch.tensor([True])):
                        idx = i 
                    else:
                        continue
                    
                self.maps[idx] = torch.zeros(self.map_size, dtype=torch.int64, device=self.device)
                new_indices.append(torch.tensor([idx]))
            indices = new_indices
        
        # Apply procedural generation
        for idx in indices:
            # Randomly choose a generation algorithm
            gen_algorithm = random.choice(["random_walk", "connected_squares"])
            if gen_algorithm == "random_walk":
                self._apply_random_walk(idx)
            else:
                self._apply_connected_squares(idx)
        
        # Initialize or reset agent positions
        if self.current_positions is None:
            self.current_positions = torch.zeros(self.batch_size + (2,), dtype=torch.int64, device=self.device)
        
        # Reset positions for specified indices
        for idx in indices:
            # Random starting position
            i = random.randint(0, self.map_size[0] - 1)
            j = random.randint(0, self.map_size[1] - 1)
            self.current_positions[idx, 0] = i
            self.current_positions[idx, 1] = j
        
        # Reset step counter
        if self.step_count is None:
            self.step_count = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        else:
            self.step_count[indices] = 0
            
        # Reset consecutive moves counter (for TURTLE mode)
        if self.consecutive_moves is None:
            self.consecutive_moves = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        else:
            self.consecutive_moves[indices] = 0
            
        # Create observations
        obs = self._create_observations()
        
        # Create and return a TensorDict instead of a plain dictionary
        return TensorDict({"observation": obs[indices]}, batch_size=torch.Size([len(indices)]))
    
    def _step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: Action tensor with shape matching batch_size + (1,)
            
        Returns:
            TensorDict with observation, reward, done information
        """
        # Increment step counter
        self.step_count += 1
        
        # Flatten action tensor
        action = action.view(-1)
        
        # Process actions based on mode
        rewards = torch.zeros(self.batch_size + (1,), dtype=torch.float32, device=self.device)
        
        for batch_idx in range(self.batch_size[0]):
            act = action[batch_idx]["action"]
            
            if self.mode == "NARROW":
                i, j = self.current_positions[batch_idx]
                
                if act < c.MODIFICATION_ACTIONS:  # Modification action
                    self.maps[batch_idx, i, j] = act
                    self.consecutive_moves[batch_idx] = 0  # Reset consecutive moves counter
                # If act == 7, it's a NO_ACTION, so we do nothing
            
            elif self.mode == "TURTLE":
                i, j = self.current_positions[batch_idx]
                
                if act < c.MODIFICATION_ACTIONS:  # Modification action
                    self.maps[batch_idx, i, j] = act
                    self.consecutive_moves[batch_idx] = 0  # Reset consecutive moves counter
                elif act in c.MOVE_ACTION:  # Movement action
                    self.consecutive_moves[batch_idx] += 1
                    # Apply movement function
                    new_i, new_j = c.MOVE_ACTION[act](i, j, self.map_size[0] - 1)
                    self.current_positions[batch_idx, 0] = new_i
                    self.current_positions[batch_idx, 1] = new_j
                    
                    # Apply punishment for repeated movement
                    if self.consecutive_moves[batch_idx] > 5:
                        punishment = -(1.05 ** self.consecutive_moves[batch_idx].item())
                        rewards[batch_idx, 0] += punishment
            
            elif self.mode == "WIDE":
                # For WIDE mode, we need to calculate which cell to modify
                map_size_prod = self.map_size[0] * self.map_size[1]
                if act < c.MODIFICATION_ACTIONS * map_size_prod:
                    # Calculate which action and which cell
                    modification = act % c.MODIFICATION_ACTIONS
                    cell_idx = act // c.MODIFICATION_ACTIONS
                    i = cell_idx // self.map_size[1]
                    j = cell_idx % self.map_size[1]
                    
                    # Apply modification
                    self.maps[batch_idx, i, j] = modification
        
        # Calculate rewards using critic if available
        if self.critic:
            # Create tensor for critic evaluation
            map_tensor = self.maps.float()
            map_tensor = torch.unsqueeze(map_tensor, 1)
            critic_reward = self.critic.forward(map_tensor)
            rewards += critic_reward.view(self.batch_size + (1,))
        
        # Create observations after actions
        obs = self._create_observations()
        
        # Check if done (max steps reached)
        done = (self.step_count >= self.max_steps).view(self.batch_size + (1,))
        
        # Return a TensorDict instead of a plain dictionary
        return TensorDict({
            "observation": obs,
            "reward": rewards,
            "done": done,
            "terminated" : done.clone(),
            "truncated": done.clone(),  # Same as done for now
        }, batch_size=self.batch_size)
    
    def _apply_random_walk(self, idx):
        """Apply random walk algorithm to generate a map"""
        steps = random.randint(20, 50)  # Number of steps for random walk
        
        # Start at a random position
        i, j = random.randint(0, self.map_size[0] - 1), random.randint(0, self.map_size[1] - 1)
        
        for _ in range(steps):
            # Place a tile
            tile_choice = random.random()
            if tile_choice < 0.8:  # 80% chance
                self.maps[idx, i, j] = 1  # Empty tile
            elif tile_choice < 0.84:  # 4% chance
                self.maps[idx, i, j] = 6  # Door
            else:  # 16% chance
                enemy_type = random.randint(2, 5)  # Enemy types 2-5
                self.maps[idx, i, j] = enemy_type
                
            # Move randomly
            direction = random.randint(0, 3)  # 0: left, 1: up, 2: right, 3: down
            if direction == 0 and j > 0:
                j -= 1
            elif direction == 1 and i > 0:
                i -= 1
            elif direction == 2 and j < self.map_size[1] - 1:
                j += 1
            elif direction == 3 and i < self.map_size[0] - 1:
                i += 1
    
    def _apply_connected_squares(self, idx):
        """Generate a map with connected squares"""
        # Number of squares to generate
        num_squares = random.randint(3, 8)
        
        # Size range for squares
        min_size, max_size = 2, 4
        
        # Generate squares
        for _ in range(num_squares):
            # Random square size
            width = random.randint(min_size, max_size)
            height = random.randint(min_size, max_size)
            
            # Random position (ensuring it fits on the map)
            start_i = random.randint(0, self.map_size[0] - width)
            start_j = random.randint(0, self.map_size[1] - height)
            
            # Fill the square
            for i in range(start_i, start_i + width):
                for j in range(start_j, start_j + height):
                    if random.random() < 0.9:  # 90% empty, 10% potential enemies or doors
                        self.maps[idx, i, j] = 1  # Empty
                    else:
                        if random.random() < 0.75:  # 75% of 10% = 7.5% enemies
                            enemy_type = random.randint(2, 5)
                            self.maps[idx, i, j] = enemy_type
                        else:  # 25% of 10% = 2.5% doors
                            self.maps[idx, i, j] = 6
            
            # If not the first square, connect to a previous square
            if _ > 0:
                # Simple connection - draw a line of empty tiles
                prev_i = random.randint(0, self.map_size[0] - 1)
                prev_j = random.randint(0, self.map_size[1] - 1)
                
                # Find path between points
                current_i, current_j = start_i, start_j
                while current_i != prev_i or current_j != prev_j:
                    # Decide which direction to move
                    if current_i < prev_i:
                        current_i += 1
                    elif current_i > prev_i:
                        current_i -= 1
                    elif current_j < prev_j:
                        current_j += 1
                    elif current_j > prev_j:
                        current_j -= 1
                    
                    # Place corridor (empty tile)
                    self.maps[idx, current_i, current_j] = 1

    def _create_observations(self):
        """Create observation tensor from current state"""
        batch_size = self.batch_size[0]
        obs = torch.zeros(self.batch_size + (self.obs_dim,), dtype=torch.float32, device=self.device)
        
        for batch_idx in range(batch_size):
            # Flatten map
            flattened_map = self.maps[batch_idx].flatten()
            
            # Current position (2 values)
            i, j = self.current_positions[batch_idx]
            
            # Construct observation: [flattened_map, position_i, position_j, hero_data]
            # Assuming hero_data is all zeros for now - this would need to be updated for real implementation
            hero_data = torch.zeros(c.HERO_TENSOR_SIZE, dtype=torch.float32, device=self.device)
            
            # Combine all parts into observation
            obs[batch_idx, :flattened_map.size(0)] = flattened_map.float()
            obs[batch_idx, flattened_map.size(0)] = float(i)
            obs[batch_idx, flattened_map.size(0) + 1] = float(j)
            obs[batch_idx, flattened_map.size(0) + 2:flattened_map.size(0) + 2 + c.HERO_TENSOR_SIZE] = hero_data
            
        return obs
    
    def _set_seed(self, seed=None):
        """Set random seed for reproducibility"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)