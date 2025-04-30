"""
Vectorized level critic module for evaluating 12x12 game levels based on 9 rules.

Evaluates map structure, tile placement, connectivity, and entity counts.
Uses tensor operations for improved batch processing performance.
"""

import torch
from enum import Enum
from typing import Tuple, List, Set, Dict, Optional
import torch.nn.functional as F

# --- Constants ---

MAP_HEIGHT: int = 12
MAP_WIDTH: int = 12
TOTAL_TILES: int = MAP_HEIGHT * MAP_WIDTH


class TileType(Enum):
    """Defines the different types of tiles in the game map."""
    WALL = 0
    EMPTY_FLOOR = 1
    ENEMY_2 = 2
    ENEMY_3 = 3
    ENEMY_4 = 4
    ENEMY_5 = 5
    DOOR = 6


# Tile sets for efficient checking
ENEMY_TILES: Set[int] = {
    TileType.ENEMY_2.value,
    TileType.ENEMY_3.value,
    TileType.ENEMY_4.value,
    TileType.ENEMY_5.value,
}
EMPTY_ENEMY_TILES: Set[int] = {TileType.EMPTY_FLOOR.value} | ENEMY_TILES
EDGE_ALLOWED_TILES: Set[int] = {TileType.WALL.value, TileType.DOOR.value}

# --- Penalties (Negative Rewards) ---
# Rule 1: Invalid edge tiles
PENALTY_INVALID_EDGE_TILE: float = -8.0
# Rule 2: Door count limits (Min 2, Max 4)
PENALTY_TOO_FEW_DOORS: float = -6.0  # New: Min 2 doors
PENALTY_TOO_MANY_DOORS: float = -7.0  # Existing: Max 4 doors
# Rule 3: Non-edge doors or multiple doors per edge
PENALTY_NON_EDGE_DOOR: float = -8.0
PENALTY_DOOR_SAME_EDGE: float = -7.0
# Rule 4: Enemy count limits (Min 1, Max 4)
PENALTY_NO_ENEMIES: float = -9.0  # New: Min 1 enemy
PENALTY_TOO_MANY_ENEMIES: float = -6.0  # Existing: Max 4 enemies
# Rule 5: Disconnected empty/enemy tiles
PENALTY_DISCONNECTED_TILE: float = -2.5
# Rule 6: Door lacks adjacent empty/enemy tile
PENALTY_DOOR_NO_EMPTY_NEIGHBOR: float = -4.0
# Rule 7: < 50% empty/enemy tiles
PENALTY_LOW_EMPTY_RATIO_FACTOR: float = -3


# --- Helper Functions ---

def create_edge_mask() -> torch.Tensor:
    """Create a mask identifying edge positions in a map."""
    mask = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.bool)
    mask[0, :] = True  # Top edge
    mask[-1, :] = True  # Bottom edge
    mask[:, 0] = True  # Left edge
    mask[:, -1] = True  # Right edge
    return mask


def create_edge_type_masks() -> Dict[str, torch.Tensor]:
    """Create masks for the four edge types."""
    edge_masks = {}
    h, w = MAP_HEIGHT, MAP_WIDTH
    
    # Create masks for each edge
    edge_masks["TOP"] = torch.zeros((h, w), dtype=torch.bool)
    edge_masks["TOP"][0, :] = True
    
    edge_masks["BOTTOM"] = torch.zeros((h, w), dtype=torch.bool)
    edge_masks["BOTTOM"][-1, :] = True
    
    edge_masks["LEFT"] = torch.zeros((h, w), dtype=torch.bool)
    edge_masks["LEFT"][1:-1, 0] = True  # Excluding corners
    
    edge_masks["RIGHT"] = torch.zeros((h, w), dtype=torch.bool)
    edge_masks["RIGHT"][1:-1, -1] = True  # Excluding corners
    
    return edge_masks


def create_neighbor_kernels() -> torch.Tensor:
    """Create convolution kernels for finding orthogonal neighbors."""
    # Create a kernel for 4-way connectivity
    kernel = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    kernel[0, 0, 1, 0] = 1  # Left
    kernel[0, 0, 1, 2] = 1  # Right
    kernel[0, 0, 0, 1] = 1  # Top
    kernel[0, 0, 2, 1] = 1  # Bottom
    return kernel


# --- Vectorized Rule Evaluation Functions ---

def evaluate_rule1_edges_vectorized(map_batch: torch.Tensor) -> torch.Tensor:
    """Vectorized calculation of penalty for invalid edge tiles (Rule 1)."""
    batch_size = map_batch.shape[0]
    edge_mask = create_edge_mask().to(map_batch.device)
    
    # Create allowed edge tiles mask
    allowed_mask = torch.zeros_like(map_batch, dtype=torch.bool)
    for tile in EDGE_ALLOWED_TILES:
        allowed_mask |= (map_batch == tile)
    
    # Find edge tiles that are not allowed
    invalid_edges = edge_mask.unsqueeze(0) & ~allowed_mask
    
    # Count invalid edges per map
    invalid_count = invalid_edges.sum(dim=(1, 2))
    
    return invalid_count * PENALTY_INVALID_EDGE_TILE


def evaluate_rule2_door_count(door_counts: torch.Tensor) -> torch.Tensor:
    """Vectorized calculation of penalty for door count limits (Rule 2)."""
    batch_size = door_counts.shape[0]
    penalties = torch.zeros_like(door_counts, dtype=torch.float32)
    
    # Too few doors (< 2)
    too_few_mask = door_counts < 2
    penalties[too_few_mask] += PENALTY_TOO_FEW_DOORS
    
    # Too many doors (> 4)
    excess_doors = torch.clamp(door_counts - 4, min=0)
    penalties += excess_doors * PENALTY_TOO_MANY_DOORS
    
    return penalties


def evaluate_rule3_doors_placement_vectorized(map_batch: torch.Tensor) -> torch.Tensor:
    """Vectorized calculation of penalty for door placement (Rule 3)."""
    batch_size = map_batch.shape[0]
    penalties = torch.zeros(batch_size, device=map_batch.device, dtype=torch.float32)
    
    # Get door positions
    door_positions = (map_batch == TileType.DOOR.value)
    
    # Check for non-edge doors
    edge_mask = create_edge_mask().to(map_batch.device)
    non_edge_doors = door_positions & ~edge_mask.unsqueeze(0)
    non_edge_count = non_edge_doors.sum(dim=(1, 2))
    penalties += non_edge_count * PENALTY_NON_EDGE_DOOR
    
    # Check for multiple doors per edge
    edge_masks = create_edge_type_masks()
    for edge_name, edge_mask in edge_masks.items():
        # Count doors on this edge for each map
        edge_mask = edge_mask.to(map_batch.device)
        doors_on_edge = (door_positions & edge_mask.unsqueeze(0)).sum(dim=(1, 2))
        
        # Count extra doors (beyond 1) for each map
        extra_doors = torch.clamp(doors_on_edge - 1, min=0)
        penalties += extra_doors * PENALTY_DOOR_SAME_EDGE
    
    return penalties


def evaluate_rule4_enemy_count(enemy_counts: torch.Tensor) -> torch.Tensor:
    """Vectorized calculation of penalty for enemy count limits (Rule 4)."""
    batch_size = enemy_counts.shape[0]
    penalties = torch.zeros_like(enemy_counts, dtype=torch.float32)
    
    # No enemies
    no_enemies_mask = enemy_counts == 0
    penalties[no_enemies_mask] += PENALTY_NO_ENEMIES
    
    # Too many enemies (> 4)
    excess_enemies = torch.clamp(enemy_counts - 4, min=0)
    penalties += excess_enemies * PENALTY_TOO_MANY_ENEMIES
    
    return penalties


def evaluate_rule6_door_neighbors_vectorized(map_batch: torch.Tensor) -> torch.Tensor:
    """Vectorized calculation of penalty for doors lacking valid neighbors (Rule 6)."""
    batch_size = map_batch.shape[0]
    
    # Create one-hot encoded maps for the relevant tile types
    door_maps = (map_batch == TileType.DOOR.value).float()
    valid_neighbor_maps = torch.zeros_like(map_batch, dtype=torch.float32)
    
    for tile in EMPTY_ENEMY_TILES:
        valid_neighbor_maps += (map_batch == tile).float()
    
    # Create padding to handle edges correctly during convolution
    padded_valid_maps = F.pad(valid_neighbor_maps, (1, 1, 1, 1), "constant", 0)
    
    # Create a single door connectivity check
    # For each door (value=1), check if any of its neighbors are valid tiles
    neighbor_kernel = create_neighbor_kernels().to(map_batch.device)
    
    # For each map, check if doors have valid neighbors
    valid_neighbor_counts = torch.zeros(batch_size, device=map_batch.device)
    
    for i in range(batch_size):
        # Perform convolution to count valid neighbors for each position
        # The convolution output will have values 0-4 representing count of valid neighbors
        neighbor_count = F.conv2d(
            padded_valid_maps[i:i+1].unsqueeze(1), 
            neighbor_kernel, 
            padding=0
        ).squeeze()
        
        # Get valid neighbor count only at door positions
        door_positions = door_maps[i]
        doors_with_neighbors = (neighbor_count * door_positions) > 0
        
        # Count doors without valid neighbors
        door_count = door_positions.sum()
        doors_with_neighbors_count = doors_with_neighbors.sum()
        doors_without_neighbors = door_count - doors_with_neighbors_count
        
        valid_neighbor_counts[i] = doors_without_neighbors
    
    return valid_neighbor_counts * PENALTY_DOOR_NO_EMPTY_NEIGHBOR


def evaluate_rule7_empty_ratio(empty_enemy_counts: torch.Tensor) -> torch.Tensor:
    """Vectorized calculation of penalty for low empty/enemy ratio (Rule 7)."""
    min_required = TOTAL_TILES // 2
    shortfall = torch.clamp(min_required - empty_enemy_counts, min=0)
    return shortfall * PENALTY_LOW_EMPTY_RATIO_FACTOR


def find_connected_components(map_batch: torch.Tensor, tile_values: Set[int]) -> torch.Tensor:
    """
    Find all connected components with values > 0 for multiple maps.
    Penalizes each connected component which is not the biggest based on its tile count.
    
    Args:
        map_batch: Batch of maps [batch_size, MAP_HEIGHT, MAP_WIDTH]
        tile_values: Set of tile values to consider traversable
        
    Returns:
        A tensor of penalties based on disconnected components
    """
    batch_size = map_batch.shape[0]
    penalties = torch.zeros(batch_size, device=map_batch.device, dtype=torch.float32)
    
    # Create traversable tile maps
    traversable_maps = torch.zeros_like(map_batch, dtype=torch.bool)
    for tile in tile_values:
        traversable_maps |= (map_batch == tile)
    
    # Process each map in the batch
    for b in range(batch_size):
        traversable = traversable_maps[b]
        
        # Skip if no traversable tiles
        if not traversable.sum().item():
            continue
        
        # Track all connected components
        visited = torch.zeros_like(traversable, dtype=torch.bool)
        components = []  # List to store sizes of connected components
        
        # Find all connected components
        for y in range(traversable.shape[0]):
            for x in range(traversable.shape[1]):
                # Skip if not traversable or already visited
                if not traversable[y, x] or visited[y, x]:
                    continue
                
                # Start a new component
                component_size = 0
                queue = [(y, x)]
                visited[y, x] = True
                
                # Flood fill to find this component
                while queue:
                    cy, cx = queue.pop(0)
                    component_size += 1
                    
                    # Check 4-directional neighbors
                    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ny, nx = cy + dy, cx + dx
                        
                        # Check bounds
                        if 0 <= ny < traversable.shape[0] and 0 <= nx < traversable.shape[1]:
                            if traversable[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                
                # Store this component's size
                components.append(component_size)
        
        # If there are connected components
        if components:
            # Find the largest component
            largest_component_size = max(components)
            
            # Calculate penalty for all non-largest components
            total_penalty = 0
            for component_size in components:
                if component_size != largest_component_size:
                    total_penalty += component_size * PENALTY_DISCONNECTED_TILE
            
            penalties[b] = total_penalty
            
            # Debugging info
    
    return penalties
# --- Main Critic Function (Vectorized) ---

def level_critic_vectorized(x: torch.Tensor) -> torch.Tensor:
    """
    Vectorized evaluation of a batch of maps based on 9 rules.
    Compatible as a drop-in replacement for self.critic in RL training loops.
    
    Args:
        x: Input tensor with shape (batch_size, channels, height, width) or (batch_size, height, width)
            
    Returns:
        rewards: Tensor of shape (batch_size,) containing evaluated rewards
    """
    # Handle different input formats
    if x.dim() == 4:  # (batch, channels, height, width)
        map_batch = x.squeeze(1) if x.shape[1] == 1 else x[:, 0]  # Take first channel if multi-channel
    elif x.dim() == 3:  # (batch, height, width)
        map_batch = x
    else:
        raise ValueError(f"Unexpected input shape: {x.shape}. Expected (batch, channels, height, width) or (batch, height, width)")
    
    # Convert to integer type if needed
    if map_batch.dtype != torch.int and map_batch.dtype != torch.long:
        map_batch = map_batch.to(torch.int)
    
    n_batch, height, width = map_batch.shape
    rewards = torch.zeros(n_batch, device=map_batch.device, dtype=torch.float32)

    # Check map dimensions once per batch
    if height != MAP_HEIGHT or width != MAP_WIDTH:
        print(f"Error: Maps must be {MAP_HEIGHT}x{MAP_WIDTH}.")
        rewards.fill_(-1000.0)
        return rewards

    # Pre-calculate counts for the entire batch at once
    door_maps = (map_batch == TileType.DOOR.value)
    door_counts = door_maps.sum(dim=(1, 2))
    
    enemy_maps = torch.zeros_like(map_batch, dtype=torch.bool)
    for enemy_tile in ENEMY_TILES:
        enemy_maps |= (map_batch == enemy_tile)
    enemy_counts = enemy_maps.sum(dim=(1, 2))
    
    empty_enemy_maps = torch.zeros_like(map_batch, dtype=torch.bool)
    for tile in EMPTY_ENEMY_TILES:
        empty_enemy_maps |= (map_batch == tile)
    empty_enemy_counts = empty_enemy_maps.sum(dim=(1, 2))

    # Apply rule evaluations in vectorized form
    # Rule 1: Edge Tiles must be Wall/Door
    rewards += evaluate_rule1_edges_vectorized(map_batch)
    
    # Rule 2: Door Count (Min 2, Max 4)
    # rewards += evaluate_rule2_door_count(door_counts)
    
    # Rule 3: Door Placement (On Edge, Max 1 per Edge)
    # rewards += evaluate_rule3_doors_placement_vectorized(map_batch)
    
    # Rule 4: Enemy Count (Min 1, Max 4)
    rewards += evaluate_rule4_enemy_count(enemy_counts)
    
    # Rule 5: Connectivity of Empty/Enemy Tiles (still partly sequential)
    rewards += find_connected_components(map_batch, EMPTY_ENEMY_TILES)
    
    # Rule 6: Door Adjacency (Must have Empty/Enemy neighbor)
    # rewards += evaluate_rule6_door_neighbors_vectorized(map_batch)
    
    # Rule 7: Empty/Enemy Ratio (Min 50%)
    rewards += evaluate_rule7_empty_ratio(empty_enemy_counts)

    return rewards


# --- Optional Parallel Processing for Batch Splitting ---

def level_critic_parallel(x: torch.Tensor, num_workers: int = 4) -> torch.Tensor:
    """
    Process map batch in parallel using PyTorch's multiprocessing.
    Compatible as a drop-in replacement for self.critic in RL training loops.
    
    Args:
        x: Input tensor with shape (batch_size, channels, height, width) or (batch_size, height, width)
        num_workers: Number of parallel workers
        
    Returns:
        Rewards tensor for the batch
    """
    import torch.multiprocessing as mp
    
    # Handle different input formats
    if x.dim() == 4:  # (batch, channels, height, width)
        map_batch = x.squeeze(1) if x.shape[1] == 1 else x[:, 0]  # Take first channel if multi-channel
    elif x.dim() == 3:  # (batch, height, width)
        map_batch = x
    else:
        raise ValueError(f"Unexpected input shape: {x.shape}. Expected (batch, channels, height, width) or (batch, height, width)")
    
    # Convert to integer type if needed
    if map_batch.dtype != torch.int and map_batch.dtype != torch.long:
        map_batch = map_batch.to(torch.int)
    
    # Function to process a chunk of the batch
    def process_chunk(chunk_idx, chunk, result_queue):
        device = chunk.device
        # Move to CPU for multiprocessing if needed
        if device.type != 'cpu':
            chunk = chunk.cpu()
        # Process the chunk
        rewards = level_critic_vectorized(chunk)
        # Put result in queue with index for reassembly
        result_queue.put((chunk_idx, rewards))
    
    n_batch = map_batch.shape[0]
    
    # Skip parallelization for small batches
    if n_batch < num_workers * 2:
        return level_critic_vectorized(map_batch)
    
    # Setup multiprocessing
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, ignore
            pass
    
    result_queue = mp.Queue()
    processes = []
    
    chunk_size = n_batch // num_workers
    remainder = n_batch % num_workers
    
    # Launch processes for each chunk
    start_idx = 0
    for i in range(num_workers):
        # Calculate chunk size (distribute remainder)
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        # Get chunk
        chunk = map_batch[start_idx:end_idx]
        
        # Launch process
        p = mp.Process(target=process_chunk, args=(i, chunk, result_queue))
        p.start()
        processes.append(p)
        
        start_idx = end_idx
    
    # Collect results
    results = []
    for _ in range(num_workers):
        results.append(result_queue.get())
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Sort results by chunk index and combine
    results.sort(key=lambda x: x[0])
    combined_results = torch.cat([r[1] for r in results])
    
    # Move results back to original device if needed
    if map_batch.device.type != 'cpu':
        combined_results = combined_results.to(map_batch.device)
    
    return combined_results


# Example usage for basic testing
if __name__ == "__main__":
    print("Running basic tests for vectorized level_critic...")

    # Test Case 1: Valid map (Reward near 0)
    test_map_1 = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.int)
    test_map_1[1:-1, 1:-1] = TileType.EMPTY_FLOOR.value
    test_map_1[0, 6] = TileType.DOOR.value  # Top
    test_map_1[-1, 6] = TileType.DOOR.value  # Bottom
    test_map_1[6, 0] = TileType.DOOR.value  # Left
    test_map_1[6, -1] = TileType.DOOR.value  # Right (4 doors total)
    test_map_1[3, 3] = TileType.ENEMY_2.value
    test_map_1[8, 8] = TileType.ENEMY_3.value  # 2 enemies total
    test_map_1[1: -1, 2] = TileType.WALL.value
    test_map_1[2, 1: -1] = TileType.WALL.value
    print(test_map_1)

    # Test Case 2: Multiple errors (Large negative reward)
    test_map_2 = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.int)
    test_map_2[1:5, 1:5] = TileType.EMPTY_FLOOR.value  # Area 1
    test_map_2[7:11, 7:11] = TileType.EMPTY_FLOOR.value  # Area 2 (disconnected)
    test_map_2[0, 0] = TileType.EMPTY_FLOOR.value  # Invalid edge
    test_map_2[0, 1] = TileType.DOOR.value
    test_map_2[0, 2] = TileType.DOOR.value  # Door same edge
    test_map_2[1, 6] = TileType.DOOR.value  # Non-edge door
    test_map_2[-1, 1] = TileType.DOOR.value
    test_map_2[-1, 2] = TileType.DOOR.value
    test_map_2[-1, 3] = TileType.DOOR.value  # 6 doors total (> 4)
    # test_map_2 has 0 enemies initially (violates Rule 4 min)
    test_map_2[5, 0] = TileType.DOOR.value  # Door with no empty neighbor
    test_map_2[4:7, 0:2] = TileType.WALL.value  # Ensure wall neighbors

    # Test Case 3: Low empty ratio, 1 door, 0 enemies (Moderate negative reward)
    test_map_3 = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.int)
    test_map_3[1:4, 1:4] = TileType.EMPTY_FLOOR.value  # Only 9 empty tiles (< 72)
    test_map_3[0, 2] = TileType.DOOR.value  # Only 1 door (< 2)
    # Has 0 enemies (violates Rule 4 min)

    # Test Case 4: Valid structure but only 1 enemy (should be allowed)
    test_map_4 = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.int)
    test_map_4[1:-1, 1:-1] = TileType.EMPTY_FLOOR.value
    test_map_4[0, 6] = TileType.DOOR.value
    test_map_4[-1, 6] = TileType.DOOR.value
    test_map_4[5, 5] = TileType.ENEMY_2.value  # Exactly 1 enemy

    # Test Case 5: Wall map (?)
    test_map_5 = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.int)

    # Create batch in different formats to test the function's flexibility
    map_batch = torch.stack([test_map_1])#, test_map_2, test_map_3, test_map_4, test_map_5])
    
    # Test with standard format (batch, height, width)
    standard_input = map_batch
    
    # Test with (batch, channel, height, width) format used in RL training
    channel_input = torch.unsqueeze(map_batch, dim=1)
    
    from time import time
    
    # Test drop-in functionality with both input formats
    print("\nTesting with standard input format (batch, height, width):")
    start = time()
    rewards_standard = level_critic_vectorized(standard_input)
    end = time()
    print(f"Standard input format time: {end - start:.4f}s")
    
    print("\nTesting with channel input format (batch, channel, height, width):")
    start = time()
    rewards_channel = level_critic_vectorized(channel_input)
    end = time()
    print(f"Channel input format time: {end - start:.4f}s")
    
    # Verify results are the same regardless of input format
    format_diff = torch.abs(rewards_standard - rewards_channel).sum().item()
    print(f"Difference between input formats: {format_diff:.6f}")
    
    # Print results
    for i, label in enumerate(["Valid", "Errors", "Low Empty/Door/Enemy", "Valid 1 Enemy", "Wall Map"]):
        print(f"Map {i+1} ({label}) Reward: {rewards_standard[i].item():.2f}")
    
    # Test GPU acceleration if available
    if torch.cuda.is_available():
        print("\nTesting GPU acceleration:")
        map_batch_gpu = map_batch.cuda()
        channel_input_gpu = channel_input.cuda()
        
        # Warm up GPU
        _ = level_critic_vectorized(channel_input_gpu)
        
        # Test GPU performance
        start = time()
        rewards_gpu = level_critic_vectorized(channel_input_gpu)
        torch.cuda.synchronize()  # Wait for GPU execution to complete
        end = time()
        print(f"GPU processing time: {end - start:.4f}s")
        
        # Compare with CPU
        start = time()
        rewards_cpu = level_critic_vectorized(channel_input)
        end = time()
        print(f"CPU processing time: {end - start:.4f}s")
        
        # Verify GPU results match CPU
        rewards_cpu_from_gpu = rewards_gpu.cpu()
        gpu_cpu_diff = torch.abs(rewards_cpu - rewards_cpu_from_gpu).sum().item()
        print(f"Difference between GPU and CPU results: {gpu_cpu_diff:.6f}")
    else:
        print("\nGPU not available for testing.")
    
    # Test performance with larger batch
    larger_batch = torch.stack([test_map_1, test_map_2, test_map_3, test_map_4, test_map_5] * 50)
    larger_input = torch.unsqueeze(larger_batch, dim=1)
    
    print("\nTesting with larger batch (250 maps):")
    start = time()
    rewards_large = level_critic_vectorized(larger_input)
    end = time()
    print(f"Vectorized time (250 maps): {end - start:.4f}s")
    
    # Test with real-world usage pattern
    print("\nTesting with real-world usage pattern (as in training loop):")
    x = torch.unsqueeze(map_batch, dim=1).float()  # Convert to float as in original code
    start = time()
    critic_rewards = level_critic_vectorized(x).squeeze(-1)  # Remove trailing dim if present
    end = time()
    print(f"Training loop pattern processing time: {end - start:.4f}s")
