import pytest
import torch
import numpy as np
from enum import Enum

# Import the module containing level_critic
# Adjust the import to match your module structure
from .level_critic import (
    level_critic,
    Entry,
    WALL,
    EMPTY_MIN,
    EMPTY_MAX,
    DOOR,
    ENEMY_MIN,
    ENEMY_MAX,
    evaluate_contiguity,
    evaluate_disconnected_tiles,
    evaluate_doors,
    evaluate_enemy_count,
    find_contiguous_areas,
    count_enemies,
    count_traversable_tiles,
    find_door_positions,
    evaluate_traversable_tiles,
)


@pytest.fixture
def sample_map():
    """Create a simple test map with a single contiguous area."""
    # Create a 5x7 map
    test_map = torch.zeros((5, 7), dtype=torch.int)

    # Create a path through the center
    for i in range(1, 6):
        test_map[2, i] = 1  # Empty tile

    # Add some enemies
    test_map[1, 2] = 3  # Enemy
    test_map[3, 4] = 4  # Enemy

    # Add a door at the right edge
    test_map[2, 6] = DOOR

    return test_map


@pytest.fixture
def sample_hero_right_entry():
    """Create a hero state with RIGHT entry."""
    # [health, item1, item2, entry, rooms_left]
    return torch.tensor([8.0, 1.0, 0.0, 1.0, 5.0])  # Entry.RIGHT is index 1


@pytest.fixture
def sample_hero_low_health():
    """Create a hero state with low health."""
    # [health, item1, item2, entry, rooms_left]
    return torch.tensor([2.0, 1.0, 0.0, 1.0, 5.0])  # Health = 2, Entry.RIGHT


def test_level_critic_basic(sample_map, sample_hero_right_entry):
    """Test the basic functionality of level_critic with a simple map."""
    # Create batch dimensions
    map_batch = sample_map.unsqueeze(0)
    hero_batch = sample_hero_right_entry.unsqueeze(0)

    # Get rewards
    rewards = level_critic(map_batch, hero_batch)

    # Check that the output is a tensor with the correct shape
    assert isinstance(rewards, torch.Tensor)
    assert rewards.shape == (1,)

    # Basic reward should be positive for this well-formed map
    assert rewards[0].item() > 0


def test_traversable_tiles_reward(sample_map):
    """Test that traversable tiles are rewarded correctly."""
    # Count traversable tiles in the sample map
    traversable_count = sum(
        1 for y in range(5) for x in range(7) if 1 <= sample_map[y, x] <= 6
    )

    # Calculate expected reward
    expected_reward = 0.2 * traversable_count

    # Get actual reward
    actual_reward = evaluate_traversable_tiles(sample_map)

    assert abs(actual_reward - expected_reward) < 1e-6


def test_contiguity_reward(sample_map, sample_hero_right_entry):
    """Test that a fully contiguous map gets the proper reward."""
    # Create batch dimensions
    map_batch = sample_map.unsqueeze(0)
    hero_batch = sample_hero_right_entry.unsqueeze(0)

    # Find contiguous areas
    contiguous_areas = find_contiguous_areas(sample_map)

    # Should have exactly one contiguous area
    assert len(contiguous_areas) == 1

    # Check that contiguity reward is +2.0
    contiguity_reward, _ = evaluate_contiguity(contiguous_areas)
    assert contiguity_reward == 2.0


def test_multiple_contiguous_areas():
    """Test that multiple significant contiguous areas are penalized."""
    # Create a map with two separate areas
    test_map = torch.zeros((7, 7), dtype=torch.int)

    # First area (3x3)
    test_map[1:4, 1:4] = 1

    # Second area (3x3)
    test_map[4:7, 4:7] = 1

    # Find contiguous areas
    contiguous_areas = find_contiguous_areas(test_map)

    # Should have exactly two contiguous areas
    assert len(contiguous_areas) == 2

    # Check that contiguity reward includes -3.0 penalty for multiple areas
    contiguity_reward, _ = evaluate_contiguity(contiguous_areas)
    assert contiguity_reward == -3.0


def test_disconnected_tiles_penalty():
    """Test penalty for disconnected tiles."""
    # Create a map with a main area and some disconnected tiles
    test_map = torch.zeros((6, 6), dtype=torch.int)

    # Main area (4x4)
    test_map[1:5, 1:5] = 1

    # Disconnected tile
    test_map[0, 0] = 1

    # Find contiguous areas for primary area size
    contiguous_areas = find_contiguous_areas(test_map)
    primary_area_size = len(contiguous_areas[0])

    # Count total traversable tiles
    total_traversable = count_traversable_tiles(test_map)

    # Should have 17 total traversable tiles (16 in main + 1 disconnected)
    assert total_traversable == 17
    assert primary_area_size == 16

    # Check penalty: -0.5 per disconnected tile
    disconnected_penalty = evaluate_disconnected_tiles(
        primary_area_size, total_traversable
    )
    assert disconnected_penalty == -0.5


def test_door_correct_entry(sample_map, sample_hero_right_entry):
    """Test reward for door matching hero entry direction."""
    # Door is at right edge, hero enters from right
    doors = find_door_positions(sample_map)
    primary_area = find_contiguous_areas(sample_map)[0]
    height, width = sample_map.shape

    # Entry direction from hero tensor (RIGHT = 1)
    entry_idx = int(sample_hero_right_entry[3].item())
    entry = list(Entry)[entry_idx]

    # Check door reward
    door_reward = evaluate_doors(doors, primary_area, height, width, entry)
    assert door_reward == 1.0  # +1 for door in correct entry direction


def test_door_wrong_entry():
    """Test no reward for door not matching hero entry direction."""
    # Create a map with a door at the bottom
    test_map = torch.zeros((5, 5), dtype=torch.int)
    test_map[1:4, 1:4] = 1  # Empty area
    test_map[4, 2] = DOOR  # Door at BOTTOM

    # Create hero with RIGHT entry
    hero = torch.tensor([8.0, 1.0, 0.0, 1.0, 5.0])  # Entry.RIGHT
    entry_idx = int(hero[3].item())
    entry = list(Entry)[entry_idx]

    # Find door and area
    doors = find_door_positions(test_map)
    primary_area = find_contiguous_areas(test_map)[0]
    height, width = test_map.shape

    # Check door reward (should be 0, no bonus for wrong direction)
    door_reward = evaluate_doors(doors, primary_area, height, width, entry)
    assert door_reward == 0.0


def test_too_many_doors():
    """Test penalty for too many doors."""
    # Create a map with 5 doors (more than 4)
    test_map = torch.zeros((7, 7), dtype=torch.int)
    test_map[2:5, 2:5] = 1  # Empty area

    # Add doors at edges
    test_map[0, 3] = DOOR  # Top
    test_map[6, 3] = DOOR  # Bottom
    test_map[3, 0] = DOOR  # Left
    test_map[3, 6] = DOOR  # Right
    test_map[0, 0] = DOOR  # Corner (5th door)

    # Create hero
    hero = torch.tensor([8.0, 1.0, 0.0, 1.0, 5.0])
    entry_idx = int(hero[3].item())
    entry = list(Entry)[entry_idx]

    # Find doors and area
    doors = find_door_positions(test_map)
    primary_area = find_contiguous_areas(test_map)[0]
    height, width = test_map.shape

    # Check door reward (should include -3.0 penalty for >4 doors)
    door_reward = evaluate_doors(doors, primary_area, height, width, entry)
    assert door_reward < -2.0  # Should include the penalty


def test_enemy_count_normal_health(sample_map, sample_hero_right_entry):
    """Test enemy count evaluation with normal health."""
    enemy_count = count_enemies(sample_map)
    traversable = count_traversable_tiles(sample_map)
    health = sample_hero_right_entry[0].item()

    # Get reward
    enemy_reward = evaluate_enemy_count(enemy_count, traversable, health)

    # Map has 2 enemies, ~7 traversable tiles, so ratio is ~0.29
    # With normal health (>= 3), threshold is 0.25
    # Should be negative as we exceed the threshold
    assert enemy_reward < 0


def test_enemy_count_low_health(sample_map, sample_hero_low_health):
    """Test enemy count evaluation with low health."""
    enemy_count = count_enemies(sample_map)
    traversable = count_traversable_tiles(sample_map)
    health = sample_hero_low_health[0].item()

    # Get reward
    enemy_reward = evaluate_enemy_count(enemy_count, traversable, health)

    # With low health (< 3), threshold is 0.2, penalty should be higher
    assert enemy_reward < -1.0  # Should include stronger penalty for low health


def test_level_critic_empty_map():
    """Test level_critic with an empty map (all walls)."""
    # Create an empty map (all walls)
    empty_map = torch.zeros((5, 5), dtype=torch.int)
    hero = torch.tensor([8.0, 1.0, 0.0, 1.0, 5.0])

    # Create batch dimensions
    map_batch = empty_map.unsqueeze(0)
    hero_batch = hero.unsqueeze(0)

    # Get rewards
    rewards = level_critic(map_batch, hero_batch)

    # Reward should be close to 0 (no traversable tiles)
    assert abs(rewards[0].item()) < 1e-6


def test_level_critic_complete():
    """Test level_critic with multiple maps in a batch."""
    # Create two different maps
    map1 = torch.zeros((5, 5), dtype=torch.int)
    map1[1:4, 1:4] = 1  # Good connected map
    map1[2, 4] = DOOR  # Door at right

    map2 = torch.zeros((5, 5), dtype=torch.int)
    map2[1, 1:4] = 1  # Smaller connected area
    map2[3, 1:4] = 1  # Second disconnected area
    map2[1, 4] = DOOR  # Door at right

    # Create heroes
    hero1 = torch.tensor([8.0, 1.0, 0.0, 1.0, 5.0])  # RIGHT entry
    hero2 = torch.tensor([2.0, 1.0, 0.0, 1.0, 5.0])  # RIGHT entry, low health

    # Stack into batches
    map_batch = torch.stack([map1, map2])
    hero_batch = torch.stack([hero1, hero2])

    # Get rewards
    rewards = level_critic(map_batch, hero_batch)

    # Check shapes
    assert rewards.shape == (2,)

    # First map should have higher reward than second map
    assert rewards[0].item() > rewards[1].item()


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main(["-v", "test_level_critic.py"])
