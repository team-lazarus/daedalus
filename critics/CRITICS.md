## Level critic module for evaluating procedurally generated game levels.

This module provides functions to analyze and score game levels based on
various criteria such as connectivity, enemy placement, and door positioning.

- +2 reward if the whole map is contigious (values [1,5] have tiles [1-5] on up, left, right, bottom to them)
- -0.5 reward for each tile [1,6] placed away from the primary contigious area
- -3 reward for more than one primary contigious area (two or more areas with similar area to each other)
- +1 reward for placing a door in the direction the hero is going to enter from (is hero vector has Entry.right, then a door on the right should be rewarded)
- -0.5 reward for each extra enemy than desired (enemies [2,5] to tile [1] ratio should be lower than 0.25),
- -0.35 reward for each extra enemy given that the user has low health (low health < 3, then ratio should be 0.2)
- -2 reward for placing more than 4 doors, or inaccessible doors, or doors which are not at the left, right, bottom or up edge of the map
- +0.2 reward for each traversible tile placed
