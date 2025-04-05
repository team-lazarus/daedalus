import os
import daedalus.models.mdp.constants as c

MAP_SIZE = (10, 10)  # Slightly larger map
HERO_TENSOR_SIZE = 5  # Must match hero_ranges structure below
PERCENTAGE_MOD_LIMIT = 0.25  # Modify up to 25% of the map
AGENT_STRATEGY = "turtle"  # narrow, turtle, or wide

NUM_EPISODES = 50  # Number of training episodes
MAX_STEPS_PER_EPISODE = 200  # Max steps for agent to reach % mod limit
METRIC_UPDATE_FREQ = 5  # Save metrics CSV every 5 episodes
RESULTS_DIR = "training_results/mdp"  # Directory to save results
METRICS_FILE = os.path.join(RESULTS_DIR, f"metrics_{AGENT_STRATEGY}.csv")
AGENT_STATE_FILE = os.path.join(RESULTS_DIR, f"agent_state_{AGENT_STRATEGY}.json")

HERO_RANGES = {
    # Name: List or range of values
    "health": range(5, 16),  # Health 5-15
    "item1": [False, True],  # Has item 1?
    "item2": [False, True],  # Has item 2?
    "entry": list(c.Entry),  # Entry point enum
    "rooms_left": range(1, 6),  # Number of rooms left 1-5
}
