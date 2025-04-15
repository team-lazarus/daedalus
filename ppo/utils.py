# utils.py
import torch
import numpy as np
import random
import os
from typing import Tuple, Callable, Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from config import TrainConfig # Assuming config.py is in the same directory

# --- Map and Hero Generation ---
def create_random_hero_tensor(size: int, device: torch.device) -> torch.Tensor:
    """Creates a random hero tensor."""
    # [health(1-10), item1(0/1), item2(0/1), direction(0-3), rooms_left(0-10)]
    health = torch.randint(1, 11, (1,), device=device)
    item1 = torch.randint(0, 2, (1,), device=device)
    item2 = torch.randint(0, 2, (1,), device=device)
    direction = torch.randint(0, 4, (1,), device=device)
    rooms_left = torch.randint(0, 11, (1,), device=device)
    # Ensure tensor size matches config.hero_tensor_size if different logic needed
    if size != 5:
        # Adjust logic if hero tensor definition changes
         raise ValueError(f"Hero tensor size mismatch. Expected 5, got {size}")

    # Concatenate and ensure float type for the network
    hero = torch.cat([health, item1, item2, direction, rooms_left]).float()
    return hero # Shape: [5]

def generate_initial_map(map_size: Tuple[int, int], walk_steps: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Generates an initial map via random walk."""
    rows, cols = map_size
    grid = np.zeros((rows, cols), dtype=int) # Start with all walls (0)
    
    # Start at a random position
    start_pos = current_pos = (random.randint(0, rows - 1), random.randint(0, cols - 1))

    for _ in range(walk_steps):
        # Choose tile type: 75% empty (1), 25% enemy ([2,5])
        tile_type = 1 if random.random() < 0.75 else random.randint(2, 5)
        grid[current_pos] = tile_type

        # Move to a random neighbor (staying within bounds)
        possible_moves = []
        r, c = current_pos
        if r > 0: possible_moves.append((r - 1, c))
        if r < rows - 1: possible_moves.append((r + 1, c))
        if c > 0: possible_moves.append((r, c - 1))
        if c < cols - 1: possible_moves.append((r, c + 1))

        if not possible_moves: break # Should not happen on reasonable map size
        current_pos = random.choice(possible_moves)

    # Ensure start position is empty if overwritten
    grid[start_pos] = 1

    return grid, start_pos


# --- Rich Printing ---
TILE_COLORS = {
    0: "grey50",  # Wall
    1: "white",   # Empty
    2: "red",     # Enemy
    3: "red",     # Enemy
    4: "red",     # Enemy
    5: "red",     # Enemy
    6: "green",   # Door
}
DEFAULT_COLOR = "bright_black" # For unknown tile types

def format_map_rich(grid: np.ndarray, title: str = "Map") -> Panel:
    """Formats a map numpy array into a Rich Panel."""
    rows, cols = grid.shape
    table = Table.grid(expand=False)
    for _ in range(cols):
        table.add_column()

    for r in range(rows):
        row_cells = []
        for c in range(cols):
            tile = grid[r, c]
            color = TILE_COLORS.get(tile, DEFAULT_COLOR)
            # Represent tile value, maybe add player marker later if needed
            cell_content = f"[{color}]{tile}[/{color}]"
            row_cells.append(cell_content)
        table.add_row(*row_cells)

    return Panel(table, title=title, border_style="blue")

# --- Checkpointing ---
def save_checkpoint(state: Dict[str, Any], filepath: str) -> None:
    """Saves training state to a file."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath: str, device: torch.device) -> Optional[Dict[str, Any]]:
    """Loads training state from a file."""
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found: {filepath}")
        return None
    try:
        checkpoint = torch.load(filepath, map_location=device)
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint from {filepath}: {e}")
        return None

# --- Device Handling ---
def get_device(requested: str = "auto") -> torch.device:
    """Gets the appropriate torch device."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif requested == "cpu":
        return torch.device("cpu")
    else: # 'auto' or fallback
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")