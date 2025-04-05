# Define the Entry enum as requested
from enum import Enum


class Entry(Enum):
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"


# Define action constants for clarity
NO_ACTION = 0
SET_EMPTY = 1
# Assuming enemy types correspond to actions 2, 3, 4, 5
PLACE_ENEMY_1 = 2
PLACE_ENEMY_2 = 3
PLACE_ENEMY_3 = 4
PLACE_ENEMY_4 = 5
PLACE_DOOR = 6

# Define movement actions for Turtle strategy
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
