# --- START OF FILE level_criticv3.py ---
"""
Level critic module V3 for evaluating 12x12 game levels based on 7 rules.

Evaluates map structure, tile placement, and connectivity constraints.
"""

import torch
from enum import Enum
from collections import deque, defaultdict
from typing import Tuple, List, Set, Dict, Optional, Callable

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
# Rule 2: Too many doors (> 4)
PENALTY_TOO_MANY_DOORS: float = -7.0
# Rule 3: Non-edge doors or multiple doors per edge
PENALTY_NON_EDGE_DOOR: float = -8.0
PENALTY_DOOR_SAME_EDGE: float = -7.0
# Rule 4: Too many enemies (> 4)
PENALTY_TOO_MANY_ENEMIES: float = -6.0
# Rule 5: Disconnected empty/enemy tiles
PENALTY_DISCONNECTED_TILE: float = -0.5
# Rule 6: Door lacks adjacent empty/enemy tile
PENALTY_DOOR_NO_EMPTY_NEIGHBOR: float = -4.0
# Rule 7: < 50% empty/enemy tiles
PENALTY_LOW_EMPTY_RATIO_FACTOR: float = -0.2


# --- Helper Functions ---


def is_valid_position(y: int, x: int) -> bool:
    """Check if coordinates (y, x) are within standard map bounds."""
    return 0 <= y < MAP_HEIGHT and 0 <= x < MAP_WIDTH


def get_neighbors(y: int, x: int) -> List[Tuple[int, int]]:
    """Get valid orthogonal neighbor coordinates for (y, x)."""
    neighbors = []
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ny, nx = y + dy, x + dx
        if is_valid_position(ny, nx):
            neighbors.append((ny, nx))
    return neighbors


def find_tiles(
    map_tensor: torch.Tensor, tile_values: Set[int]
) -> List[Tuple[int, int]]:
    """Find all (y, x) positions of tiles with specified values."""
    positions: List[Tuple[int, int]] = []
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            if map_tensor[y, x].item() in tile_values:
                positions.append((y, x))
    return positions


def bfs_search(
    start_node: Tuple[int, int],
    traversable_nodes: Set[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    """Perform BFS to find reachable nodes within the traversable_nodes set."""
    if start_node not in traversable_nodes:
        return set()

    visited: Set[Tuple[int, int]] = {start_node}
    queue: deque[Tuple[int, int]] = deque([start_node])

    while queue:
        cy, cx = queue.popleft()
        for ny, nx in get_neighbors(cy, cx):
            neighbor_pos = (ny, nx)
            if neighbor_pos in traversable_nodes and neighbor_pos not in visited:
                visited.add(neighbor_pos)
                queue.append(neighbor_pos)
    return visited


def get_edge_for_door(y: int, x: int) -> Optional[str]:
    """Determine which edge a coordinate is on, None if not on edge."""
    # Checks simplified for 12x12 map
    width_m1 = MAP_WIDTH - 1
    height_m1 = MAP_HEIGHT - 1
    if y == 0 and 0 <= x <= width_m1:
        return "TOP"
    if y == height_m1 and 0 <= x <= width_m1:
        return "BOTTOM"
    if x == 0 and 0 < y < height_m1:
        return "LEFT"
    if x == width_m1 and 0 < y < height_m1:
        return "RIGHT"
    return None  # Not on an edge


# --- Rule Evaluation Functions ---


def evaluate_rule1_edges(map_tensor: torch.Tensor) -> float:
    """Calculate penalty for invalid edge tiles (Rule 1)."""
    penalty = 0.0
    h, w = MAP_HEIGHT, MAP_WIDTH
    for y in range(h):
        if map_tensor[y, 0].item() not in EDGE_ALLOWED_TILES:
            penalty += PENALTY_INVALID_EDGE_TILE
        if map_tensor[y, w - 1].item() not in EDGE_ALLOWED_TILES:
            penalty += PENALTY_INVALID_EDGE_TILE
    for x in range(1, w - 1):  # Avoid double counting corners
        if map_tensor[0, x].item() not in EDGE_ALLOWED_TILES:
            penalty += PENALTY_INVALID_EDGE_TILE
        if map_tensor[h - 1, x].item() not in EDGE_ALLOWED_TILES:
            penalty += PENALTY_INVALID_EDGE_TILE
    return penalty


def evaluate_rule3_doors_placement(door_positions: List[Tuple[int, int]]) -> float:
    """Calculate penalty for non-edge doors and multiple doors per edge (Rule 3)."""
    penalty = 0.0
    edge_door_counts: Dict[str, int] = defaultdict(int)
    for dy, dx in door_positions:
        edge = get_edge_for_door(dy, dx)
        if edge is None:
            penalty += PENALTY_NON_EDGE_DOOR
        else:
            edge_door_counts[edge] += 1
            # Penalize immediately if count exceeds 1 for an edge
            if edge_door_counts[edge] > 1:
                penalty += PENALTY_DOOR_SAME_EDGE  # Penalty per extra door on same edge
    return penalty


def evaluate_rule5_connectivity(empty_enemy_positions: List[Tuple[int, int]]) -> float:
    """Calculate penalty for disconnected empty/enemy tiles (Rule 5)."""
    num_empty_enemy = len(empty_enemy_positions)
    if num_empty_enemy <= 1:  # 0 or 1 tile is always connected
        return 0.0

    empty_enemy_set = set(empty_enemy_positions)
    start_node = empty_enemy_positions[0]
    connected_component = bfs_search(start_node, empty_enemy_set)
    unconnected_count = num_empty_enemy - len(connected_component)
    return unconnected_count * PENALTY_DISCONNECTED_TILE


def evaluate_rule6_door_neighbors(
    map_tensor: torch.Tensor, door_positions: List[Tuple[int, int]]
) -> float:
    """Calculate penalty for doors lacking adjacent empty/enemy tiles (Rule 6)."""
    penalty = 0.0
    for dy, dx in door_positions:
        has_valid_neighbor = False
        for ny, nx in get_neighbors(dy, dx):
            if map_tensor[ny, nx].item() in EMPTY_ENEMY_TILES:
                has_valid_neighbor = True
                break
        if not has_valid_neighbor:
            penalty += PENALTY_DOOR_NO_EMPTY_NEIGHBOR
    return penalty


# --- Main Critic Function ---


def level_critic(map_batch: torch.Tensor, hero_batch: torch.Tensor) -> torch.Tensor:
    """Evaluates a batch of 12x12 maps based on 7 rules, ignoring hero state."""
    n_batch, height, width = map_batch.shape
    rewards = torch.zeros(n_batch, device=map_batch.device, dtype=torch.float32)

    if height != MAP_HEIGHT or width != MAP_WIDTH:
        print(f"Error: Maps must be {MAP_HEIGHT}x{MAP_WIDTH}.")
        return torch.full_like(rewards, -1000.0)  # Consistent high penalty

    for i in range(n_batch):
        current_map = map_batch[i]
        current_reward = 0.0

        # Pre-calculate tile positions for efficiency
        door_positions = find_tiles(current_map, {TileType.DOOR.value})
        enemy_positions = find_tiles(current_map, ENEMY_TILES)
        empty_enemy_positions = find_tiles(current_map, EMPTY_ENEMY_TILES)
        num_doors = len(door_positions)
        num_enemies = len(enemy_positions)
        num_empty_enemy = len(empty_enemy_positions)

        # Rule 1: Edge Tiles (Wall/Door Only)
        current_reward += evaluate_rule1_edges(current_map)

        # Rule 2: Max 4 Doors
        if num_doors > 4:
            current_reward += (num_doors - 4) * PENALTY_TOO_MANY_DOORS

        # Rule 3: Doors on Edge, Max 1 per Edge
        current_reward += evaluate_rule3_doors_placement(door_positions)

        # Rule 4: Max 4 Enemies
        if num_enemies > 4:
            current_reward += (num_enemies - 4) * PENALTY_TOO_MANY_ENEMIES

        # Rule 5: Connectivity of Empty/Enemy
        current_reward += evaluate_rule5_connectivity(empty_enemy_positions)

        # Rule 6: Doors Adjacent to Empty/Enemy
        current_reward += evaluate_rule6_door_neighbors(current_map, door_positions)

        # Rule 7: Min 50% Empty/Enemy
        min_required = TOTAL_TILES // 2
        if num_empty_enemy < min_required:
            shortfall = min_required - num_empty_enemy
            current_reward += shortfall * PENALTY_LOW_EMPTY_RATIO_FACTOR

        rewards[i] = current_reward

    return rewards


# Example usage for basic testing
if __name__ == "__main__":
    print("Running basic tests for level_critic_v3...")

    # Test Case 1: Mostly valid map (should have reward near 0)
    test_map_1 = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.int)
    test_map_1[1:-1, 1:-1] = TileType.EMPTY_FLOOR.value
    test_map_1[0, 6] = TileType.DOOR.value  # Top
    test_map_1[-1, 6] = TileType.DOOR.value  # Bottom
    test_map_1[6, 0] = TileType.DOOR.value  # Left
    test_map_1[6, -1] = TileType.DOOR.value  # Right
    test_map_1[3, 3] = TileType.ENEMY_2.value
    test_map_1[8, 8] = TileType.ENEMY_3.value
    test_map_1[3, 8] = TileType.ENEMY_4.value
    test_map_1[8, 3] = TileType.ENEMY_5.value  # 4 enemies, 4 doors, connected

    # Test Case 2: Multiple errors (should have large negative reward)
    test_map_2 = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.int)
    test_map_2[1:5, 1:5] = TileType.EMPTY_FLOOR.value  # Area 1
    test_map_2[7:11, 7:11] = TileType.EMPTY_FLOOR.value  # Area 2 (disconnected)
    test_map_2[0, 0] = TileType.EMPTY_FLOOR.value  # Invalid edge
    test_map_2[0, 1] = TileType.DOOR.value
    test_map_2[0, 2] = TileType.DOOR.value  # Door same edge
    test_map_2[1, 6] = TileType.DOOR.value  # Non-edge door
    test_map_2[-1, 1] = TileType.DOOR.value
    test_map_2[-1, 2] = TileType.DOOR.value
    test_map_2[-1, 3] = TileType.DOOR.value  # 6 doors total
    test_map_2[3, 3] = TileType.ENEMY_2.value
    test_map_2[8, 8] = TileType.ENEMY_3.value
    test_map_2[9, 9] = TileType.ENEMY_4.value
    test_map_2[10, 10] = TileType.ENEMY_5.value
    test_map_2[7, 7] = TileType.ENEMY_2.value  # 5 enemies total
    # Make a door with no empty neighbor (surrounded by walls)
    test_map_2[5, 0] = TileType.DOOR.value
    test_map_2[4, 0] = TileType.WALL.value
    test_map_2[6, 0] = TileType.WALL.value
    test_map_2[5, 1] = TileType.WALL.value

    # Test Case 3: Low empty ratio (should have moderate negative reward)
    test_map_3 = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.int)
    test_map_3[1:4, 1:4] = TileType.EMPTY_FLOOR.value  # Only 9 empty tiles
    test_map_3[0, 2] = TileType.DOOR.value
    test_map_3[4, 0] = TileType.DOOR.value  # Corrected placement for valid adj

    # Create batch and dummy hero data
    map_batch = torch.stack([test_map_1, test_map_2, test_map_3])
    hero_batch = torch.zeros((3, 5))  # Ignored by this critic

    # Evaluate
    rewards = level_critic(map_batch, hero_batch)

    print(f"Map 1 (Valid) Reward: {rewards[0].item():.2f}")
    print(f"Map 2 (Errors) Reward: {rewards[1].item():.2f}")
    print(f"Map 3 (Low Empty) Reward: {rewards[2].item():.2f}")

# --- END OF FILE level_criticv4.py ---
