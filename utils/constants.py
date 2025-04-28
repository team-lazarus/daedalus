TILE_EMPTY = 1
ENEMY_TILES = [2, 3, 4, 5]
TILE_DOOR = 6
MODIFICATION_ACTIONS = 7
NO_ACTION = {7}
MOVE_ACTION = {
    7: lambda i, j, s: (max(0, i - 1), j),  # left
    8: lambda i, j, s: (i, max(0, j - 1)),
    9: lambda i, j, s: (min(s-1, i + 1), j),
    10: lambda i, j, s: (i, min(s-1, j + 1)),
}
POSSIBLE_MODES = {"NARROW", "TURTLE", "WIDE"}
HERO_TENSOR_SIZE = 7