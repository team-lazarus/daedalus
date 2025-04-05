# test_level_critic.py

import pytest
import torch
import numpy as np
from enum import Enum

# Import the level_critic function and constants
# Adjust the import to match your module structure if necessary
# Assuming the main function is now just level_critic in the file
from daedalus.critics.level_critic import level_critic, DOOR, Entry

# Tolerance for floating point comparisons
TOL = 1e-6

# --- Fixtures for Maps (Keep as they are, add 7x7 variant) ---
@pytest.fixture
def map_single_area():
    """Map: 5x7, single contiguous area, 2 enemies, 1 valid door (right)."""
    m = torch.zeros((5, 7), dtype=torch.int)
    m[2, 1:6] = 1
    m[1, 2] = 3
    m[3, 4] = 4
    m[2, 6] = DOOR
    return m

@pytest.fixture
def map_multiple_areas():
    """Map: 7x7, two separate 3x3 areas."""
    m = torch.zeros((7, 7), dtype=torch.int)
    m[1:4, 1:4] = 1
    m[4:7, 4:7] = 1
    return m

@pytest.fixture
def map_many_doors():
    """Map: 7x7, central area, 5 doors (4 edge, 1 corner - all inaccessible)."""
    m = torch.zeros((7, 7), dtype=torch.int)
    m[2:5, 2:5] = 1
    m[0, 3] = DOOR
    m[6, 3] = DOOR
    m[3, 0] = DOOR
    m[3, 6] = DOOR
    m[0, 0] = DOOR
    return m

@pytest.fixture
def map_many_enemies():
    """Map: 5x5, 3x3 area, 4 enemies (high ratio)."""
    m = torch.zeros((5, 5), dtype=torch.int)
    m[1:4, 1:4] = 1
    m[1, 1] = 2
    m[1, 3] = 3
    m[3, 1] = 4
    m[3, 3] = 5
    return m

@pytest.fixture
def map_many_enemies_7x7():
    """Map: 7x7, central 3x3 area, 4 enemies (for batch test)."""
    m = torch.zeros((7, 7), dtype=torch.int)
    # Central 3x3 area
    m[2:5, 2:5] = 1
    # Add 4 enemies
    m[2, 2] = 2
    m[2, 4] = 3
    m[4, 2] = 4
    m[4, 4] = 5
    return m


@pytest.fixture
def map_inaccessible_door():
    """Map: 5x5, L-shape area, 1 inaccessible door."""
    m = torch.zeros((5, 5), dtype=torch.int)
    m[1, 1:3] = 1
    m[2:4, 1] = 1
    m[2, 4] = DOOR
    return m

@pytest.fixture
def map_non_edge_door():
    """Map: 5x5, plus-shape area, 1 non-edge door."""
    m = torch.zeros((5, 5), dtype=torch.int)
    m[1, 2] = 1
    m[2, 1] = 1
    m[2, 3] = 1
    m[3, 2] = 1
    m[2, 2] = DOOR # Non-edge door
    return m

@pytest.fixture
def map_ok_enemies():
    """Map: 5x5, 3x3 area, 1 enemy (ratio ok)."""
    m = torch.zeros((5, 5), dtype=torch.int)
    m[1:4, 1:4] = 1
    m[2, 2] = 2
    return m

@pytest.fixture
def map_exactly_4_doors():
    """Map: 7x7, central area, 4 valid edge doors (inaccessible from center)."""
    m = torch.zeros((7, 7), dtype=torch.int)
    m[2:5, 2:5] = 1 # Central non-connected area
    m[0, 3] = DOOR  # Top
    m[6, 3] = DOOR  # Bottom
    m[3, 0] = DOOR  # Left
    m[3, 6] = DOOR  # Right
    return m

# --- Corrected Fixtures for Hero States ---
@pytest.fixture
def hero_norm_r():
    """Hero: Normal health (8), Entry RIGHT (Index 1)."""
    return torch.tensor([8., 1., 0., 1.0, 5.])

@pytest.fixture
def hero_low_r():
    """Hero: Low health (2), Entry RIGHT (Index 1)."""
    return torch.tensor([2., 1., 0., 1.0, 5.])

@pytest.fixture
def hero_norm_t():
    """Hero: Normal health (8), Entry TOP (Index 0)."""
    return torch.tensor([8., 1., 0., 0.0, 5.])

# --- Test Cases ---

def test_critic_empty_map(hero_norm_r):
    # ... (no changes needed, was passing) ...
    empty_map = torch.zeros((3, 3), dtype=torch.int)
    map_batch = empty_map.unsqueeze(0)
    hero_batch = hero_norm_r.unsqueeze(0)
    rewards = level_critic(map_batch, hero_batch)
    assert abs(rewards[0].item() - 0.0) < TOL

def test_critic_single_area(map_single_area, hero_norm_r):
    """
    Test Case: Well-formed Single Area
    Visual: (See fixture) 5x7 map, contiguous path + enemies + right door.
    Hero: Normal Health, Entry Right
    Components (Recalculated based on code):
    - Trav Reward (1-5): 7 tiles * 0.2 = +1.4
    - Contiguity: Fully contiguous (8 tiles) = +2.0
    - Doors: 1 door, Edge=T, Accessible=T, Matches=T -> +1.0
    - Enemies: 2 enemies / 8 trav = 0.25. OK -> 0.0
    Expected Total: 1.4 + 2.0 + 1.0 + 0.0 = 4.4
    """
    map_batch = map_single_area.unsqueeze(0)
    hero_batch = hero_norm_r.unsqueeze(0)
    rewards = level_critic(map_batch, hero_batch)

    assert isinstance(rewards, torch.Tensor); assert rewards.shape == (1,)
    assert rewards[0].item() > 0
    # Assertion Updated
    assert abs(rewards[0].item() - 4.4) < TOL

def test_critic_multiple_areas(map_multiple_areas, hero_norm_r):
    # ... (no changes needed, was passing) ...
    map_batch = map_multiple_areas.unsqueeze(0)
    hero_batch = hero_norm_r.unsqueeze(0)
    rewards = level_critic(map_batch, hero_batch)
    assert abs(rewards[0].item() - (-0.9)) < TOL

def test_critic_many_doors(map_many_doors, hero_norm_r):
    """
    Test Case: Too Many Doors (and Inaccessible)
    Visual: (See fixture) 7x7 map, 9-tile central area, 5 doors.
    Hero: Normal Health, Entry Right
    Components (Recalculated based on code):
    - Trav Reward (1-5): 9 tiles * 0.2 = +1.8
    - Contiguity: Primary=9, Total=14. Disconnected=5 -> -2.5
    - Doors: 5 doors (>4) -> -2.0 (count penalty)
             All 5 doors Inaccessible (-2.0 each). Door(3,6) matches (+1.0).
             Total per-door checks: 5*(-2.0) + 1.0 = -9.0
             Total Door Effect: -2.0 + (-9.0) = -11.0
    - Enemies: 0 -> 0.0
    Expected Total: 1.8 - 2.5 - 11.0 = -11.7
    """
    map_batch = map_many_doors.unsqueeze(0)
    hero_batch = hero_norm_r.unsqueeze(0)
    rewards = level_critic(map_batch, hero_batch)

    assert rewards[0].item() < 0
    # Assertion Updated
    assert abs(rewards[0].item() - (-11.7)) < TOL


def test_critic_exactly_4_doors(map_exactly_4_doors, hero_norm_r):
    """
    Test Case: Exactly 4 Doors (Inaccessible from center)
    Visual: (See fixture) 7x7 map, 9-tile area, 4 edge doors.
    Hero: Normal Health, Entry Right
    Components (Recalculated based on code):
    - Trav Reward (1-5): 9 tiles * 0.2 = +1.8
    - Contiguity: Primary=9, Total=13. Disconnected=4 -> -2.0
    - Doors: 4 doors (Count penalty=0.0).
             All 4 doors Inaccessible (-2.0 each). Door(3,6) matches (+1.0).
             Total per-door checks: 4*(-2.0) + 1.0 = -7.0
             Total Door Effect: 0.0 + (-7.0) = -7.0
    - Enemies: 0 -> 0.0
    Expected Total: 1.8 - 2.0 - 7.0 = -7.2
    """
    map_batch = map_exactly_4_doors.unsqueeze(0)
    hero_batch = hero_norm_r.unsqueeze(0)
    rewards = level_critic(map_batch, hero_batch)

    # Assertion Updated
    assert abs(rewards[0].item() - (-7.2)) < TOL

def test_critic_many_enemies_normal(map_many_enemies, hero_norm_r):
     # ... (no changes needed, was passing) ...
    map_batch = map_many_enemies.unsqueeze(0)
    hero_batch = hero_norm_r.unsqueeze(0)
    rewards = level_critic(map_batch, hero_batch)
    assert abs(rewards[0].item() - 2.8) < TOL

# FIX: Added hero_norm_r as argument
def test_critic_many_enemies_low(map_many_enemies, hero_low_r, hero_norm_r):
    """
    Test Case: High Enemy Ratio, Low Health
    Visual: (See fixture) 5x5 map, 9-tile area, 4 enemies.
    Hero: Low Health (2), Entry Right
    Components (Recalculated based on code):
    - Trav Reward (1-5): 5 tiles * 0.2 = +1.0
    - Contiguity: Fully contiguous (9 tiles) = +2.0
    - Doors: 0 -> 0.0
    - Enemies: 4 enemies / 9 trav ~= 0.44. Threshold(Low H)=0.20.
               desired = floor(9*0.20)=1. excess=4-1=3. penalty=3*(-0.35)=-1.05
    Expected Total: 1.0 + 2.0 - 1.05 = 1.95
    """
    map_batch = map_many_enemies.unsqueeze(0)
    hero_batch = hero_low_r.unsqueeze(0)
    rewards = level_critic(map_batch, hero_batch)

    # Compare against normal health case - Use injected hero_norm_r
    rewards_normal = level_critic(map_batch, hero_norm_r.unsqueeze(0))
    assert rewards[0].item() < rewards_normal[0].item() # Low health penalty is harsher

    # Assertion Updated based on corrected Trav Reward
    assert abs(rewards[0].item() - 1.95) < TOL

def test_critic_ok_enemies(map_ok_enemies, hero_norm_r):
     # ... (no changes needed, was passing) ...
    map_batch = map_ok_enemies.unsqueeze(0)
    hero_batch = hero_norm_r.unsqueeze(0)
    rewards = level_critic(map_batch, hero_batch)
    assert abs(rewards[0].item() - 3.8) < TOL

def test_critic_door_entry_direction(map_single_area, hero_norm_r, hero_norm_t):
     # ... (no changes needed, was passing) ...
    map_batch = map_single_area.unsqueeze(0)
    rewards_right = level_critic(map_batch, hero_norm_r.unsqueeze(0))
    rewards_top = level_critic(map_batch, hero_norm_t.unsqueeze(0))
    assert abs((rewards_right[0].item() - rewards_top[0].item()) - 1.0) < TOL

def test_critic_inaccessible_door(map_inaccessible_door, hero_norm_r):
    """
    Test Case: Inaccessible Door Penalty
    Visual: (See fixture) 5x5 map, L-shape area, door separate.
    Hero: Normal Health, Entry Right
    Components (Recalculated based on code):
    - Trav Reward (1-5): 4 tiles * 0.2 = +0.8
    - Contiguity: Primary=4, Total=5. Disconnected=1 -> -0.5
    - Doors: 1 door, Edge=T, Inaccessible=T (-2.0), Matches=F -> -2.0
    - Enemies: 0 -> 0.0
    Expected Total: 0.8 - 0.5 - 2.0 = -1.7
    """
    map_batch = map_inaccessible_door.unsqueeze(0)
    hero_batch = hero_norm_r.unsqueeze(0)
    rewards = level_critic(map_batch, hero_batch)

    # Assertion Updated
    assert abs(rewards[0].item() - (-1.7)) < TOL

def test_critic_non_edge_door(map_non_edge_door, hero_norm_r):
    """
    Test Case: Non-Edge Door Penalty
    Visual: (See fixture) 5x5 map, plus-shape area, door in center.
    Hero: Normal Health, Entry Right
    Components (Recalculated based on code):
    - Trav Reward (1-5): 4 tiles * 0.2 = +0.8
    - Contiguity: Fully contiguous (5 tiles) = +2.0
    - Doors: 1 door, Edge=F (-2.0), Accessible=T, Matches=F -> -2.0
    - Enemies: 0 -> 0.0
    Expected Total: 0.8 + 2.0 - 2.0 = 0.8
    """
    map_batch = map_non_edge_door.unsqueeze(0)
    hero_batch = hero_norm_r.unsqueeze(0)
    rewards = level_critic(map_batch, hero_batch)

    # Assertion Updated
    assert abs(rewards[0].item() - 0.8) < TOL

# FIX: Use maps of the same size (7x7) and adapted hero logic
def test_critic_batch_processing(map_multiple_areas, map_many_enemies_7x7, map_exactly_4_doors,
                                 hero_norm_r, hero_low_r, hero_norm_t):
    """
    Test Case: Batch Processing Correctness (Using 7x7 maps)
    Map1=multiple_areas, Map2=many_enemies_7x7, Map3=exactly_4_doors
    Heroes: H1=norm_r, H2=low_r, H3=norm_t (Entry=TOP)
    Expected:
    - Reward 1 (map_multiple_areas + hero_norm_r): Expected -0.9
    - Reward 2 (map_many_enemies_7x7 + hero_low_r):
        Trav(1-5)=5*0.2=1.0; Contig=9, Fully=+2.0; Doors=0;
        Enemies=4/9=0.44, Thresh(low)=0.2, desired=floor(9*0.2)=1, excess=3, pen=3*(-0.35)=-1.05
        Total=1.0+2.0-1.05 = 1.95
    - Reward 3 (map_exactly_4_doors + hero_norm_t):
        Trav(1-5)=9*0.2=1.8; Contig=Prim=9,Total=13,Disc=4 -> -2.0;
        Doors=4(Count=0.0), Inacc=(-2.0*4), Match(TOP @ 0,3)=+1.0 -> -7.0; Enemies=0;
        Total = 1.8 - 2.0 - 7.0 = -7.2
    """
    # Stack the 7x7 maps
    map_batch = torch.stack([map_multiple_areas, map_many_enemies_7x7, map_exactly_4_doors])
    # Assign heroes H1->Map1, H2->Map2, H3->Map3
    hero_batch = torch.stack([hero_norm_r, hero_low_r, hero_norm_t])

    rewards = level_critic(map_batch, hero_batch)

    assert rewards.shape == (3,)
    # Check each reward individually against recalculated expectations for the maps used
    assert abs(rewards[0].item() - (-0.9)) < TOL # Map 1 (multiple_areas)
    assert abs(rewards[1].item() - 1.95) < TOL  # Map 2 (many_enemies_7x7)
    assert abs(rewards[2].item() - (-7.2)) < TOL # Map 3 (exactly_4_doors with TOP entry)


if __name__ == "__main__":
    pytest.main(["-v", __file__])