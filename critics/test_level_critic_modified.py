# --- START OF FILE test_critic_12x12.py ---

import torch
import numpy as np
import random
from level_critic_modifiedv3 import *

MAP_HEIGHT = 12
MAP_WIDTH = 12
NUM_TEST_MAPS = 1000  # How many random maps to generate and test


def create_random_map(height: int = MAP_HEIGHT, width: int = MAP_WIDTH) -> torch.Tensor:
    """Creates a random 12x12 map for testing the critic."""
    # Start with all walls
    m = torch.full((height, width), TileType.WALL.value, dtype=torch.int)

    # Carve out a basic traversable area (ensure border remains wall initially)
    # Example: Simple rectangular room
    room_y_start, room_y_end = random.randint(1, height // 2 - 1), random.randint(
        height // 2, height - 2
    )
    room_x_start, room_x_end = random.randint(1, width // 2 - 1), random.randint(
        width // 2, width - 2
    )
    m[room_y_start : room_y_end + 1, room_x_start : room_x_end + 1] = (
        TileType.EMPTY_FLOOR.value
    )

    # Get list of potential floor positions (where we just carved)
    floor_coords = []
    for r in range(room_y_start, room_y_end + 1):
        for c in range(room_x_start, room_x_end + 1):
            floor_coords.append((r, c))

    # Add some random floating walls inside the carved area
    num_internal_walls = random.randint(
        height // 2, height * width // 20
    )  # Adjust density as needed
    for _ in range(num_internal_walls):
        if floor_coords:
            idx = random.randrange(len(floor_coords))
            wy, wx = floor_coords.pop(
                idx
            )  # Remove coord so we don't place enemy/door here later
            m[wy, wx] = TileType.WALL.value

    # Add random enemies in the remaining floor area
    num_enemies = random.randint(
        1, height * width // 15
    )  # Ensure at least 1 enemy, adjust density
    enemy_types = [
        TileType.ENEMY_2.value,
        TileType.ENEMY_3.value,
        TileType.ENEMY_4.value,
        TileType.ENEMY_5.value,
    ]
    for _ in range(num_enemies):
        if floor_coords:
            idx = random.randrange(len(floor_coords))
            ey, ex = floor_coords.pop(idx)  # Use a floor coord
            m[ey, ex] = random.choice(enemy_types)

    # Add 1-4 doors on the edges (replacing existing walls)
    num_doors = random.randint(1, 4)
    possible_door_edges = []
    # Top edge (y=0, x=1..width-2)
    possible_door_edges.extend([(0, x) for x in range(1, width - 1)])
    # Bottom edge (y=h-1, x=1..width-2)
    possible_door_edges.extend([(height - 1, x) for x in range(1, width - 1)])
    # Left edge (y=1..height-2, x=0)
    possible_door_edges.extend([(y, 0) for y in range(1, height - 1)])
    # Right edge (y=1..height-2, x=w-1)
    possible_door_edges.extend([(y, width - 1) for y in range(1, height - 1)])

    # Shuffle and pick distinct locations
    random.shuffle(possible_door_edges)
    door_locations_added = set()
    doors_placed = 0
    for dy, dx in possible_door_edges:
        if doors_placed >= num_doors:
            break
        # Basic check: ensure door is placed on a wall (should be true by design)
        if m[dy, dx] == TileType.WALL.value:
            # More robust check: ensure door has at least one traversable neighbor inside
            has_traversable_neighbor = False
            for ny, nx in [(dy + 1, dx), (dy - 1, dx), (dy, dx + 1), (dy, dx - 1)]:
                if (
                    1 <= ny < height - 1
                    and 1 <= nx < width - 1
                    and is_traversable(m[ny, nx].item())
                ):
                    has_traversable_neighbor = True
                    break
            if has_traversable_neighbor:
                m[dy, dx] = TileType.DOOR.value
                door_locations_added.add((dy, dx))
                doors_placed += 1

    return m


def create_random_hero() -> torch.Tensor:
    """Creates a random hero state for testing."""
    entry_idx = random.choice([e.value for e in Entry])
    # [health, item1, item2, entry_idx, rooms_left]
    return torch.tensor([0.0, 0.0, float(entry_idx), 5.0])


if __name__ == "__main__":
    print(
        f"--- Testing Level Critic on {NUM_TEST_MAPS} Random {MAP_HEIGHT}x{MAP_WIDTH} Maps ---"
    )

    for i in range(NUM_TEST_MAPS):
        print(f"\n--- Test Map {i+1} ---")

        # Generate map and hero
        random_map = create_random_map(MAP_HEIGHT, MAP_WIDTH)
        random_hero = create_random_hero()

        # Prepare for critic (add batch dimension)
        map_batch = random_map.unsqueeze(0)
        hero_batch = random_hero.unsqueeze(0)

        # Get hero info for printing
        hero_entry = Entry(int(random_hero[2].item()))

        # Evaluate the map
        reward = level_critic(map_batch, hero_batch)

        if reward > 0:

            # Print map and hero info
            print("Generated Map:")
            print(random_map.numpy())

            # Print reward
            print(f"\nCritic Reward: {reward.item():.3f}")
            print("-" * (MAP_WIDTH * 3))  # Separator

    print("\n--- Testing Complete ---")

# --- END OF FILE test_critic_12x12.py ---
