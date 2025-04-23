# --- START OF FILE level_critic_modified_12x12.py ---

"""
Level critic module for evaluating procedurally generated 12x12 game levels
for a 2D roguelike.

Evaluates based on connectivity, door placement, enemy presence and placement,
edge constraints (edges must be Wall or Door), and structural features
like internal walls.

Tile Definitions:
- 0: Wall
- 1: Empty traversable space
- 2-5: Empty traversable space containing an Enemy (Types 2, 3, 4, 5)
- 6: Door (traversable)

Enemy Types (correspond to tile values 2-5):
- 2: Shoots in 4 directions (place centrally relative to traversable area)
- 3: Long range (place >5 units from doors, near corners <=2 units)
- 4: Short range (place <=3 units from doors or near internal walls)
- 5: High rate of fire (place <=3 units from other enemies)

Hero State Tensor (example): [health, item1, item2, entry_direction_idx, rooms_left]
"""

import torch
import numpy as np
from enum import Enum
from collections import deque, defaultdict
from typing import Tuple, List, Set, Dict, Optional

# --- Constants ---


class TileType(Enum):
    WALL = 0
    EMPTY_FLOOR = 1
    ENEMY_2 = 2
    ENEMY_3 = 3
    ENEMY_4 = 4
    ENEMY_5 = 5
    DOOR = 6


ENEMY_TILES = {
    TileType.ENEMY_2.value,
    TileType.ENEMY_3.value,
    TileType.ENEMY_4.value,
    TileType.ENEMY_5.value,
}
TRAVERSABLE_TILES = {
    TileType.EMPTY_FLOOR.value,
    TileType.ENEMY_2.value,
    TileType.ENEMY_3.value,
    TileType.ENEMY_4.value,
    TileType.ENEMY_5.value,
    TileType.DOOR.value,
}
# FLOOR_TILES represents tiles that are traversable but are NOT doors or walls.
# These are the tiles explicitly forbidden on the absolute map border.
FLOOR_TILES = {
    TileType.EMPTY_FLOOR.value,
    TileType.ENEMY_2.value,
    TileType.ENEMY_3.value,
    TileType.ENEMY_4.value,
    TileType.ENEMY_5.value,
}


class Entry(Enum):
    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3


# Reward/Penalty Constants (Tunable - Kept same values, might need re-tuning for 12x12)
REWARD_CONTIGUOUS_FULL = 2.0
PENALTY_DISCONNECTED_TILE = -0.5
PENALTY_MULTIPLE_LARGE_AREAS = -3.0
REWARD_ENTRY_DOOR = 1.0
PENALTY_ENEMY_RATIO_NORMAL = -0.5
PENALTY_ENEMY_RATIO_LOW_HEALTH = -0.35
PENALTY_INVALID_EDGE_TILE = -6.0  # Penalty for floor/enemy tiles (1-5) on border
PENALTY_NO_ENEMIES = -7.0
PENALTY_INACCESSIBLE_DOOR = -6.0
PENALTY_NON_EDGE_DOOR = -6.0
PENALTY_TOO_MANY_DOORS = -4.0
PENALTY_DOORS_SAME_EDGE = -5.0
REWARD_TRAVERSABLE_TILE = 0.05
REWARD_FLOATING_WALL = 0.25
REWARD_ENEMY_2_CENTRAL = 0.8
REWARD_ENEMY_3_FAR_FROM_DOOR = 0.6
REWARD_ENEMY_3_NEAR_CORNER = 0.4
REWARD_ENEMY_4_NEAR_DOOR = 0.6
REWARD_ENEMY_4_NEAR_FLOATING_WALL = 0.3
REWARD_ENEMY_5_NEAR_OTHER_ENEMY = 0.3
REWARD_ENEMY_3_4_TOGETHER = 0.7

# --- Helper Functions ---


def is_valid_position(y: int, x: int, height: int, width: int) -> bool:
    return 0 <= y < height and 0 <= x < width


def is_traversable(tile_value: int) -> bool:
    """Checks if tile is 1, 2, 3, 4, 5, or 6."""
    return tile_value in TRAVERSABLE_TILES


def is_enemy(tile_value: int) -> bool:
    """Checks if tile is 2, 3, 4, or 5."""
    return tile_value in ENEMY_TILES


def is_floor(tile_value: int) -> bool:
    """Checks if tile is 1, 2, 3, 4, or 5 (i.e., not Wall or Door)."""
    return tile_value in FLOOR_TILES


def get_neighbors(
    y: int, x: int, height: int, width: int, diagonals: bool = False
) -> List[Tuple[int, int]]:
    neighbors = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            if not diagonals and abs(dy) + abs(dx) != 1:
                continue
            ny, nx = y + dy, x + dx
            if is_valid_position(ny, nx, height, width):
                neighbors.append((ny, nx))
    return neighbors


def manhattan_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    y1, x1 = pos1
    y2, x2 = pos2
    return abs(y1 - y2) + abs(x1 - x2)


# --- BFS and Connectivity ---


def bfs_search(
    start_nodes: List[Tuple[int, int]],
    map_tensor: torch.Tensor,
    valid_tile_checker,
    get_neighbors_func=lambda pos, h, w: get_neighbors(
        pos[0], pos[1], h, w, diagonals=False
    ),
) -> Set[Tuple[int, int]]:
    height, width = map_tensor.shape
    visited = set()
    queue = deque()
    for sy, sx in start_nodes:
        # Check validity before accessing map_tensor
        if is_valid_position(sy, sx, height, width):
            tile_val = map_tensor[sy, sx].item()
            if valid_tile_checker(tile_val):
                if (sy, sx) not in visited:
                    visited.add((sy, sx))
                    queue.append((sy, sx))

    reachable_nodes = set(visited)  # Initialize with valid start nodes
    while queue:
        cy, cx = queue.popleft()
        for ny, nx in get_neighbors_func((cy, cx), height, width):
            # is_valid_position check is implicit in get_neighbors_func
            tile_val = map_tensor[ny, nx].item()
            if (ny, nx) not in visited and valid_tile_checker(tile_val):
                visited.add((ny, nx))
                queue.append((ny, nx))
                reachable_nodes.add((ny, nx))
    return reachable_nodes


def find_contiguous_areas(map_tensor: torch.Tensor) -> List[Set[Tuple[int, int]]]:
    height, width = map_tensor.shape
    visited_global = set()
    contiguous_areas = []
    for y in range(height):
        for x in range(width):
            if (y, x) not in visited_global and is_traversable(map_tensor[y, x].item()):
                # Use the updated bfs_search which handles start node validation
                area = bfs_search([(y, x)], map_tensor, is_traversable)
                if area:
                    contiguous_areas.append(area)
                    visited_global.update(
                        area
                    )  # Mark all found tiles as visited globally
    contiguous_areas.sort(key=len, reverse=True)
    return contiguous_areas


# --- Map Feature Extraction ---


def find_tiles_by_type(
    map_tensor: torch.Tensor, tile_values: Set[int]
) -> List[Tuple[int, int]]:
    height, width = map_tensor.shape
    positions = []
    for y in range(height):
        for x in range(width):
            if map_tensor[y, x].item() in tile_values:
                positions.append((y, x))
    return positions


def find_doors(map_tensor: torch.Tensor) -> List[Tuple[int, int]]:
    return find_tiles_by_type(map_tensor, {TileType.DOOR.value})


def find_enemies(
    map_tensor: torch.Tensor,
) -> Tuple[Dict[int, List[Tuple[int, int]]], List[Tuple[int, int]]]:
    height, width = map_tensor.shape
    enemies_by_type = defaultdict(list)
    all_enemy_positions = []
    for y in range(height):
        for x in range(width):
            tile_val = map_tensor[y, x].item()
            if is_enemy(tile_val):
                pos = (y, x)
                enemies_by_type[tile_val].append(pos)
                all_enemy_positions.append(pos)
    return enemies_by_type, all_enemy_positions


def find_floating_walls(map_tensor: torch.Tensor) -> List[Tuple[int, int]]:
    height, width = map_tensor.shape
    all_wall_pos = find_tiles_by_type(map_tensor, {TileType.WALL.value})
    if not all_wall_pos:
        return []

    # Find border wall tiles to start the search from
    border_walls = []
    for y in range(height):
        for x in range(width):
            # Check if it's a wall AND on the border
            if (y == 0 or y == height - 1 or x == 0 or x == width - 1) and map_tensor[
                y, x
            ].item() == TileType.WALL.value:
                border_walls.append((y, x))

    # If no border walls exist, all walls are floating (edge case for maps entirely enclosed by non-walls)
    if not border_walls:
        # However, our evaluate_edge_tiles should heavily penalize such a map anyway.
        # Assuming a valid map has border walls.
        pass

    # Find all wall tiles reachable from the border walls (using 4-way connectivity for walls)
    edge_connected_walls = bfs_search(
        border_walls, map_tensor, lambda tile: tile == TileType.WALL.value
    )

    # Floating walls are all walls minus edge-connected walls
    floating_walls = [pos for pos in all_wall_pos if pos not in edge_connected_walls]
    return floating_walls


def get_traversable_centroid(map_tensor: torch.Tensor) -> Optional[Tuple[float, float]]:
    height, width = map_tensor.shape
    positions = []
    for y in range(height):
        for x in range(width):
            if is_traversable(map_tensor[y, x].item()):
                positions.append((y, x))
    if not positions:
        return None
    # Use float division
    avg_y = sum(p[0] for p in positions) / len(positions)
    avg_x = sum(p[1] for p in positions) / len(positions)
    return (avg_y, avg_x)


# --- Evaluation Functions ---


def evaluate_edge_tiles(map_tensor: torch.Tensor) -> float:
    """Penalizes maps with non-wall/non-door tiles (1-5) on the border."""
    height, width = map_tensor.shape
    reward = 0.0
    violation_count = 0
    for y in range(height):
        for x in range(width):
            # Check only border tiles
            if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                tile_val = map_tensor[y, x].item()
                # Penalize if it's a floor or enemy tile (1-5)
                if is_floor(tile_val):
                    violation_count += 1
                    # print(f"Debug: Invalid edge tile {tile_val} at ({y},{x})") # Optional debug print

    # Apply penalty per violation
    reward += violation_count * PENALTY_INVALID_EDGE_TILE
    return reward


def evaluate_contiguity_and_connection(
    map_tensor: torch.Tensor, contiguous_areas: List[Set[Tuple[int, int]]]
) -> Tuple[float, int, int]:
    """Evaluates map connectivity based on traversable tiles (1-6)."""
    reward = 0.0
    primary_area_size = 0
    total_traversable_count = count_by_condition(map_tensor, is_traversable)

    if not contiguous_areas:
        # If there are no traversable tiles, it's technically contiguous (empty set)
        # but likely undesirable. The low total_traversable_count will result
        # in low base reward and issues with enemy ratios etc.
        # No specific penalty here, handled by other metrics.
        return reward, 0, total_traversable_count

    primary_area_size = len(contiguous_areas[0])
    disconnected_tiles = total_traversable_count - primary_area_size

    # Penalty for any disconnected traversable tiles (Rule 1 implicit check)
    reward += disconnected_tiles * PENALTY_DISCONNECTED_TILE

    # Reward if fully contiguous
    if len(contiguous_areas) == 1 and disconnected_tiles == 0:
        reward += REWARD_CONTIGUOUS_FULL
    # Penalty if multiple large areas exist
    elif len(contiguous_areas) > 1:
        second_largest_size = len(contiguous_areas[1])
        # Ensure primary_area_size > 0 to avoid division by zero or weird ratios
        if primary_area_size > 0 and second_largest_size >= 0.8 * primary_area_size:
            reward += PENALTY_MULTIPLE_LARGE_AREAS

    return reward, primary_area_size, total_traversable_count


def get_edge_location(y: int, x: int, height: int, width: int) -> Optional[Entry]:
    """Determine which edge a position is on. Returns None if not on edge."""
    is_on_top = y == 0
    is_on_bottom = y == height - 1
    is_on_left = x == 0
    is_on_right = x == width - 1

    # Check corners first for consistent assignment (assign to Top/Bottom)
    if is_on_top and is_on_left:
        return Entry.TOP
    if is_on_top and is_on_right:
        return Entry.TOP
    if is_on_bottom and is_on_left:
        return Entry.BOTTOM
    if is_on_bottom and is_on_right:
        return Entry.BOTTOM

    # Check edges
    if is_on_top:
        return Entry.TOP
    if is_on_bottom:
        return Entry.BOTTOM
    if is_on_left:
        return Entry.LEFT
    if is_on_right:
        return Entry.RIGHT

    return None  # Not on any edge


def evaluate_doors(
    door_positions: List[Tuple[int, int]],
    primary_area: Set[Tuple[int, int]],
    height: int,
    width: int,
    entry_direction: Entry,
) -> float:
    """Evaluates door count, placement, edge rules, connection, and entry matching."""
    reward = 0.0
    num_doors = len(door_positions)

    # Penalty for too many doors
    if num_doors > 4:
        reward += PENALTY_TOO_MANY_DOORS * (num_doors - 4)

    edge_counts = defaultdict(int)
    has_matching_entry_door = False

    for dy, dx in door_positions:
        edge = get_edge_location(dy, dx, height, width)

        # Check: Is door actually on an edge?
        if edge is None:
            reward += PENALTY_NON_EDGE_DOOR
            # print(f"Debug: Non-edge door at ({dy},{dx})") # Optional Debug
            continue  # Skip other checks for this invalid door

        # Check Rule 3: Door must be connected to the main traversable area
        is_connected = False
        # A door tile itself is traversable, so check if it's in the primary area
        if (dy, dx) in primary_area:
            is_connected = True
        # If the door itself isn't in the primary (e.g., BFS didn't start there),
        # check if it has a neighbor in the primary area. This is crucial.
        else:
            for ny, nx in get_neighbors(dy, dx, height, width, diagonals=False):
                # Make sure neighbor is within bounds before checking primary_area
                if (
                    is_valid_position(ny, nx, height, width)
                    and (ny, nx) in primary_area
                ):
                    is_connected = True
                    break  # Found a connection

        if not is_connected:
            reward += PENALTY_INACCESSIBLE_DOOR
            # print(f"Debug: Inaccessible door at ({dy},{dx})") # Optional Debug
            continue  # Skip other checks for this invalid door

        # Count doors per edge for Rule 2 check
        edge_counts[edge] += 1

        # Reward for matching entry direction (only the first one found)
        if edge == entry_direction and not has_matching_entry_door:
            reward += REWARD_ENTRY_DOOR
            has_matching_entry_door = True

    # Check Rule 2: Multiple doors on the same edge
    for edge, count in edge_counts.items():
        if count > 1:
            reward += PENALTY_DOORS_SAME_EDGE * (
                count - 1
            )  # Penalize per extra door on same edge
            # print(f"Debug: Multiple doors ({count}) on edge {edge.name}") # Optional Debug

    return reward


def evaluate_enemies(
    map_tensor: torch.Tensor,
    enemies_by_type: Dict[int, List[Tuple[int, int]]],
    all_enemy_positions: List[Tuple[int, int]],
    total_traversable_tiles: int,
    health: float,
    door_positions: List[Tuple[int, int]],
    floating_walls: List[Tuple[int, int]],
    traversable_centroid: Optional[Tuple[float, float]],
) -> float:
    """Evaluates enemy count (Rule 4), ratio, and placement rewards."""
    reward = 0.0
    height, width = map_tensor.shape
    enemy_count = len(all_enemy_positions)

    # Rule 4 Violation: Must have at least 1 enemy
    if enemy_count == 0:
        return PENALTY_NO_ENEMIES  # Return early with penalty

    # --- Enemy Ratio Penalty ---
    if total_traversable_tiles > 0:
        enemy_ratio = enemy_count / total_traversable_tiles
        desired_ratio = 0.2 if health < 3 else 0.25
        penalty_per_excess = (
            PENALTY_ENEMY_RATIO_LOW_HEALTH if health < 3 else PENALTY_ENEMY_RATIO_NORMAL
        )
        # Calculate penalty only if ratio is exceeded
        if enemy_ratio > desired_ratio:
            # Calculate how many enemies would be allowed
            allowed_enemies = int(desired_ratio * total_traversable_tiles)
            excess_enemies = enemy_count - allowed_enemies
            # Apply penalty only if there's actual excess (excess_enemies > 0)
            if excess_enemies > 0:
                reward += excess_enemies * penalty_per_excess
    # Consider edge case: total_traversable_tiles == 0 but enemy_count > 0 (shouldn't happen if enemies are traversable)
    elif enemy_count > 0:
        # Extremely high ratio, apply max penalty? Or rely on other map errors?
        # Let's assume this indicates other map flaws.
        pass

    # --- Enemy Placement Rewards ---
    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    enemy_pos_set = set(all_enemy_positions)  # For quick proximity checks if needed

    for enemy_type, positions in enemies_by_type.items():
        for pos in positions:
            # Reward 2: Enemy 2 (Value 2) central
            if (
                enemy_type == TileType.ENEMY_2.value
                and traversable_centroid is not None
            ):
                dist_to_centroid = manhattan_distance(pos, traversable_centroid)
                # Reward if within threshold distance
                if dist_to_centroid < 3.0:
                    # Reward scales linearly, max reward at dist 0
                    reward += REWARD_ENEMY_2_CENTRAL * (1.0 - dist_to_centroid / 3.0)

            # Reward 3: Enemy 3 (Value 3) far from doors, near corner
            elif enemy_type == TileType.ENEMY_3.value:
                min_dist_to_door = float("inf")
                if door_positions:  # Check if there are any doors
                    min_dist_to_door = min(
                        manhattan_distance(pos, d_pos) for d_pos in door_positions
                    )
                # Reward only if distance > 5
                if min_dist_to_door > 5.0:
                    # Scaled reward: more reward the further away, capped slightly
                    reward += REWARD_ENEMY_3_FAR_FROM_DOOR * min(
                        1.0,
                        (min_dist_to_door - 5.0)
                        / 5.0,  # Scale over next 5 units distance
                    )

                # Check proximity to corners (dist <= 2)
                min_dist_to_corner = min(
                    manhattan_distance(pos, corner) for corner in corners
                )
                if min_dist_to_corner <= 2.0:
                    reward += REWARD_ENEMY_3_NEAR_CORNER

            # Reward 4: Enemy 4 (Value 4) near doors or floating walls
            elif enemy_type == TileType.ENEMY_4.value:
                # Check near doors (dist <= 3)
                min_dist_to_door = float("inf")
                if door_positions:  # Check if doors exist
                    min_dist_to_door = min(
                        manhattan_distance(pos, d_pos) for d_pos in door_positions
                    )
                if min_dist_to_door <= 3.0:
                    # Reward inversely proportional to distance
                    reward += REWARD_ENEMY_4_NEAR_DOOR * (
                        1.0 - min_dist_to_door / 3.5  # Slightly gentler scaling
                    )

                # Check proximity to floating walls (dist <= 1, i.e., adjacent)
                min_dist_to_floating_wall = float("inf")
                if floating_walls:  # Check if floating walls exist
                    min_dist_to_floating_wall = min(
                        manhattan_distance(pos, w_pos) for w_pos in floating_walls
                    )
                # Reward if adjacent (distance 1) or on top (distance 0 - unlikely but possible)
                if min_dist_to_floating_wall <= 1.0:
                    reward += REWARD_ENEMY_4_NEAR_FLOATING_WALL

            # Reward 5: Enemy 5 (Value 5) near other enemies
            elif enemy_type == TileType.ENEMY_5.value:
                nearby_enemies = 0
                # Compare against all other enemy positions
                for other_pos in all_enemy_positions:
                    if pos == other_pos:
                        continue  # Don't compare enemy to itself
                    # Check if within Manhattan distance 3
                    if manhattan_distance(pos, other_pos) <= 3.0:
                        nearby_enemies += 1
                # Add reward for each nearby enemy found
                reward += nearby_enemies * REWARD_ENEMY_5_NEAR_OTHER_ENEMY

    # Reward 6: Enemy 3 and Enemy 4 together
    enemy3_positions = enemies_by_type.get(TileType.ENEMY_3.value, [])
    enemy4_positions = enemies_by_type.get(TileType.ENEMY_4.value, [])

    # Store rewarded pairs to avoid double counting (e.g., E3 near two E4s counts pair reward twice)
    rewarded_pairs = set()

    # Only check if both types of enemies exist in the map
    if enemy3_positions and enemy4_positions:
        for e3_pos in enemy3_positions:
            for e4_pos in enemy4_positions:
                # Create a canonical representation of the pair to store in the set
                pair = tuple(sorted((e3_pos, e4_pos)))
                if pair in rewarded_pairs:
                    continue  # Already rewarded this specific pair

                # Check if Manhattan distance is within 4
                if manhattan_distance(e3_pos, e4_pos) <= 4.0:
                    reward += REWARD_ENEMY_3_4_TOGETHER
                    rewarded_pairs.add(pair)  # Mark this pair as rewarded

    return reward


def evaluate_structure(floating_walls: List[Tuple[int, int]]) -> float:
    """Evaluates structural features like floating walls."""
    # Reward 1: Floating walls (simple count * reward per wall)
    return len(floating_walls) * REWARD_FLOATING_WALL


def count_by_condition(map_tensor: torch.Tensor, condition_func) -> int:
    """Helper to count tiles matching a condition function."""
    count = 0
    height, width = map_tensor.shape
    # Ensure we iterate over the full map tensor
    map_data = map_tensor.cpu().numpy() if map_tensor.is_cuda else map_tensor.numpy()
    for y in range(height):
        for x in range(width):
            if condition_func(map_data[y, x]):  # Use numpy indexing if tensor is large
                count += 1
    return count


# --- Main Critic Function ---


def level_critic(map_batch: torch.Tensor, hero_batch: torch.Tensor) -> torch.Tensor:
    """
    Evaluates a batch of 12x12 game levels based on multiple criteria.
    """
    n_batch, height, width = map_batch.shape
    rewards = torch.zeros(n_batch, device=map_batch.device)

    # Dimension check updated for 12x12
    if height != 12 or width != 12:
        print(
            f"Error: Expected 12x12 maps, got {height}x{width}. Aborting evaluation for this batch."
        )
        # Return a very low reward to strongly discourage wrong size input
        return torch.full((n_batch,), -100.0, device=map_batch.device)

    for i in range(n_batch):
        # Extract current map and hero state for this iteration
        current_map = map_batch[i]
        hero_state = hero_batch[i]
        health = hero_state[0].item()
        try:
            # Ensure entry index is valid before converting
            entry_idx_raw = hero_state[3].item()
            entry_idx = int(entry_idx_raw)
            if not (0 <= entry_idx < len(Entry)):
                raise ValueError("Invalid entry index")
            entry_direction = list(Entry)[entry_idx]
        except (IndexError, ValueError, TypeError):
            # Default to TOP if index is out of bounds, not an int, or otherwise invalid
            # print(f"Warning: Invalid hero entry index '{entry_idx_raw}' for batch item {i}. Defaulting to TOP.")
            entry_direction = Entry.TOP

        # Initialize reward for this map
        current_reward = 0.0

        # --- Pre-calculate map features ---
        # These are calculated once and passed to evaluation functions
        contiguous_areas = find_contiguous_areas(current_map)
        # Handle case where there are no traversable areas found
        primary_area = contiguous_areas[0] if contiguous_areas else set()
        door_positions = find_doors(current_map)
        enemies_by_type, all_enemy_positions = find_enemies(current_map)
        floating_walls = find_floating_walls(current_map)
        traversable_centroid = get_traversable_centroid(current_map)

        # --- Evaluate Base Traversable Area Reward ---
        # Small reward for simply having traversable space
        base_traversable_count = len(primary_area) + sum(
            len(area) for area in contiguous_areas[1:]
        )  # More accurate count
        current_reward += base_traversable_count * REWARD_TRAVERSABLE_TILE

        # --- Evaluate Edge Tile Rule ---
        # Penalizes floor/enemy tiles (1-5) on the border
        current_reward += evaluate_edge_tiles(current_map)

        # --- Evaluate Connectivity (Rule 1) ---
        # Rewards full connectivity, penalizes disconnected tiles and multiple large areas
        contiguity_reward, primary_area_size, total_traversable = (
            evaluate_contiguity_and_connection(current_map, contiguous_areas)
        )
        # Note: total_traversable is recalculated inside evaluate_contiguity_and_connection
        current_reward += contiguity_reward

        # --- Evaluate Doors (Rules 2, 3, Entry Bonus) ---
        # Checks count, edge placement, connection to primary area, non-duplication on edges
        current_reward += evaluate_doors(
            door_positions, primary_area, height, width, entry_direction
        )

        # --- Evaluate Enemies (Rule 4, Ratio, Placement Rewards 2-6) ---
        # Checks for presence, ratio to traversable space, and specific placement logic
        current_reward += evaluate_enemies(
            current_map,
            enemies_by_type,
            all_enemy_positions,
            total_traversable,  # Use count from connectivity check
            health,
            door_positions,
            floating_walls,
            traversable_centroid,
        )

        # --- Evaluate Structure (Floating Walls - Reward 1) ---
        # Rewards walls not connected to the border
        current_reward += evaluate_structure(floating_walls)

        # --- Assign final calculated reward for this map ---
        rewards[i] = current_reward

    return rewards


# Keep create_test_hero function as it might be useful for the external test script
def create_test_hero(health=8.0, entry_idx=Entry.RIGHT.value) -> torch.Tensor:
    """Creates a sample hero state tensor for testing."""
    # Structure: [health, item1, item2, entry_idx, rooms_left]
    # Ensure correct types (float for health, items, rooms_left; int/float for entry_idx)
    return torch.tensor([float(health), 1.0, 0.0, float(entry_idx), 5.0])


# --- END OF FILE level_critic_modified_12x12.py ---
