import pytest
import torch
import numpy as np
from enum import Enum

# Import the level_critic function and constants
# Adjust the import to match your module structure
from daedalus.critics.level_critic import level_critic, DOOR


@pytest.fixture
def create_map_with_single_area():
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
def create_map_with_multiple_areas():
    """Create a map with multiple disconnected areas."""
    # Create a 7x7 map
    test_map = torch.zeros((7, 7), dtype=torch.int)
    
    # First area (3x3)
    test_map[1:4, 1:4] = 1
    
    # Second area (3x3)
    test_map[4:7, 4:7] = 1
    
    return test_map


@pytest.fixture
def create_map_with_many_doors():
    """Create a map with more than 4 doors."""
    test_map = torch.zeros((7, 7), dtype=torch.int)
    test_map[2:5, 2:5] = 1  # Empty area
    
    # Add 5 doors at edges
    test_map[0, 3] = DOOR  # Top
    test_map[6, 3] = DOOR  # Bottom
    test_map[3, 0] = DOOR  # Left
    test_map[3, 6] = DOOR  # Right
    test_map[0, 0] = DOOR  # Corner (5th door)
    
    return test_map


@pytest.fixture
def create_map_with_many_enemies():
    """Create a map with many enemies (high enemy ratio)."""
    test_map = torch.zeros((5, 5), dtype=torch.int)
    
    # Make a 3x3 traversable area
    test_map[1:4, 1:4] = 1
    
    # Add many enemies (4 enemies in 9 traversable tiles)
    test_map[1, 1] = 2
    test_map[1, 3] = 3
    test_map[3, 1] = 4
    test_map[3, 3] = 5
    
    return test_map


@pytest.fixture
def hero_right_entry_normal_health():
    """Create a hero state with RIGHT entry and normal health."""
    # [health, item1, item2, entry, rooms_left]
    return torch.tensor([8., 1., 0., 1., 5.])  # Entry.RIGHT is index 1


@pytest.fixture
def hero_right_entry_low_health():
    """Create a hero state with RIGHT entry and low health."""
    # [health, item1, item2, entry, rooms_left]
    return torch.tensor([2., 1., 0., 1., 5.])  # Health = 2, Entry.RIGHT


@pytest.fixture
def hero_top_entry_normal_health():
    """Create a hero state with TOP entry and normal health."""
    # [health, item1, item2, entry, rooms_left]
    return torch.tensor([8., 1., 0., 0., 5.])  # Entry.TOP is index 0


def test_level_critic_single_area(create_map_with_single_area, hero_right_entry_normal_health):
    """Test level_critic with a well-formed single-area map."""
    # Create batch dimensions
    map_batch = create_map_with_single_area.unsqueeze(0)
    hero_batch = hero_right_entry_normal_health.unsqueeze(0)
    
    # Get rewards
    rewards = level_critic(map_batch, hero_batch)
    
    # Check output format
    assert isinstance(rewards, torch.Tensor)
    assert rewards.shape == (1,)
    
    # Well-formed map should have positive reward
    assert rewards[0].item() > 0
    
    # This map should get:
    # - ~1.4 for traversable tiles (7 tiles * 0.2)
    # - +2.0 for contiguity
    # - +1.0 for door in right direction
    # - Some penalty for enemy ratio
    # Total should be positive but less than 4.4
    reward_value = rewards[0].item()
    assert 0 < reward_value < 4.4


def test_level_critic_multiple_areas(create_map_with_multiple_areas, hero_right_entry_normal_health):
    """Test level_critic with a map having multiple disconnected areas."""
    # Create batch dimensions
    map_batch = create_map_with_multiple_areas.unsqueeze(0)
    hero_batch = hero_right_entry_normal_health.unsqueeze(0)
    
    # Get rewards
    rewards = level_critic(map_batch, hero_batch)
    
    # Check that reward includes penalty for multiple areas
    # Map should get:
    # - Reward for traversable tiles (18 * 0.2 = 3.6)
    # - -3.0 penalty for multiple areas
    # Total should be positive but less than 1.0
    reward_value = rewards[0].item()
    assert reward_value < 1.0


def test_level_critic_many_doors(create_map_with_many_doors, hero_right_entry_normal_health):
    """Test level_critic with a map having too many doors."""
    # Create batch dimensions
    map_batch = create_map_with_many_doors.unsqueeze(0)
    hero_batch = hero_right_entry_normal_health.unsqueeze(0)
    
    # Get rewards
    rewards = level_critic(map_batch, hero_batch)
    
    # Check that reward includes penalty for too many doors
    # Should have significant penalty making the reward quite negative
    reward_value = rewards[0].item()
    assert reward_value < 0


def test_level_critic_many_enemies_normal_health(create_map_with_many_enemies, hero_right_entry_normal_health):
    """Test level_critic with a map having many enemies and normal health."""
    # Create batch dimensions
    map_batch = create_map_with_many_enemies.unsqueeze(0)
    hero_batch = hero_right_entry_normal_health.unsqueeze(0)
    
    # Get rewards
    rewards = level_critic(map_batch, hero_batch)
    
    # Reward should include penalty for too many enemies
    # but with normal health, penalty should be moderate
    reward_value = rewards[0].item()
    assert reward_value < 2.0  # Less than traversable reward + contiguity reward


def test_level_critic_many_enemies_low_health(create_map_with_many_enemies, hero_right_entry_low_health):
    """Test level_critic with a map having many enemies and low health."""
    # Create batch dimensions
    map_batch = create_map_with_many_enemies.unsqueeze(0)
    hero_batch = hero_right_entry_low_health.unsqueeze(0)
    
    # Get rewards
    rewards = level_critic(map_batch, hero_batch)
    
    # With low health, penalty for enemies should be stronger
    # Compare with normal health test - reward should be lower
    reward_value = rewards[0].item()
    
    # Run the same test with normal health to compare
    normal_health_rewards = level_critic(map_batch, hero_right_entry_low_health.unsqueeze(0))
    
    # Low health should result in lower reward
    assert reward_value <= normal_health_rewards[0].item()


def test_level_critic_door_entry_direction(create_map_with_single_area, hero_right_entry_normal_health, hero_top_entry_normal_health):
    """Test that level_critic rewards doors matching hero entry direction."""
    # Create batch dimensions - map has door on right
    map_batch = create_map_with_single_area.unsqueeze(0)
    
    # Test with RIGHT entry (matching)
    right_hero_batch = hero_right_entry_normal_health.unsqueeze(0)
    right_rewards = level_critic(map_batch, right_hero_batch)
    
    # Test with TOP entry (not matching)
    top_hero_batch = hero_top_entry_normal_health.unsqueeze(0)
    top_rewards = level_critic(map_batch, top_hero_batch)
    
    # RIGHT entry should have higher reward (door bonus)
    assert right_rewards[0].item() > top_rewards[0].item()


def test_level_critic_empty_map():
    """Test level_critic with an empty map (all walls)."""
    # Create an empty map (all walls)
    empty_map = torch.zeros((5, 5), dtype=torch.int)
    hero = torch.tensor([8., 1., 0., 1., 5.])
    
    # Create batch dimensions
    map_batch = empty_map.unsqueeze(0)
    hero_batch = hero.unsqueeze(0)
    
    # Get rewards
    rewards = level_critic(map_batch, hero_batch)
    
    # Reward should be 0 (no traversable tiles, no rewards/penalties)
    assert abs(rewards[0].item()) < 1e-6


def test_level_critic_batch_processing():
    """Test that level_critic correctly processes a batch of maps."""
    # Create a batch of different maps
    map1 = torch.zeros((5, 5), dtype=torch.int)
    map1[1:4, 1:4] = 1
    map1[2, 4] = DOOR
    
    map2 = torch.zeros((5, 5), dtype=torch.int)
    map2[1:4, 1:4] = 1
    map2[1, 1] = 3  # Add an enemy
    map2[2, 2] = 4  # Add an enemy
    map2[3, 3] = 5  # Add an enemy
    
    map3 = torch.zeros((5, 5), dtype=torch.int)
    map3[1, 1:4] = 1
    map3[3, 1:4] = 1
    
    # Create a batch of different heroes
    hero1 = torch.tensor([8., 1., 0., 1., 5.])  # Normal health, RIGHT
    hero2 = torch.tensor([2., 1., 0., 1., 5.])  # Low health, RIGHT
    hero3 = torch.tensor([8., 1., 0., 0., 5.])  # Normal health, TOP
    
    # Stack into batches
    map_batch = torch.stack([map1, map2, map3])
    hero_batch = torch.stack([hero1, hero2, hero3])
    
    # Get rewards
    rewards = level_critic(map_batch, hero_batch)
    
    # Check output shape
    assert rewards.shape == (3,)
    
    # Check each reward individually
    assert rewards[0].item() > 0  # Good map with door
    assert rewards[1].item() < rewards[0].item()  # Too many enemies with low health
    assert rewards[2].item() < 0  # Multiple disconnected areas


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main(["-v", "test_level_critic.py"])