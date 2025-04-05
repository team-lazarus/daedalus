import os
import torch
import logging
from rich.console import Console
from rich.logging import RichHandler

from daedalus.agents import MDPTrainer
from daedalus.critics import level_critic
from daedalus.models.mdp import MDPAgent
import daedalus.hyperparameters as h

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
# Get the logger instance
log = logging.getLogger("rich")
# Create a console object for direct printing if needed
console = Console()

if __name__ == "__main__":
    # --- Configuration ---
    # Define ranges for hero state generation (ensure order matches critic/tensor needs
    # Verify HERO_TENSOR_SIZE matches the number of keys in hero_ranges
    assert h.HERO_TENSOR_SIZE == len(
        h.HERO_RANGES
    ), f"HERO_TENSOR_SIZE ({h.HERO_TENSOR_SIZE}) must match the number of hero parameters ({len(h.hero_ranges)})"

    # --- Create Agent ---
    agent = MDPAgent(
        size=h.MAP_SIZE,
        hero_tensor_size=h.HERO_TENSOR_SIZE,
        critic=level_critic,
        strategy=h.AGENT_STRATEGY,
        percentage_change=h.PERCENTAGE_MOD_LIMIT,
        initial_epsilon=h.INITIAL_EPSILON,
        epsilon_decay=h.EPSILON_DECAY,  # Decay slightly faster
        min_epsilon=h.MIN_EPSILON,
    )

    # --- Optional: Load Agent State ---
    # Ensure the directory exists before trying to load
    if os.path.exists(h.AGENT_STATE_FILE):
        try:
            log.info(f"Attempting to load agent state from: {h.AGENT_STATE_FILE}")
            agent.load_state(h.AGENT_STATE_FILE)
            log.info("Agent state loaded successfully. Continuing training.")
        except FileNotFoundError:
            # This case is handled inside load_state now, but good practice
            log.info(
                f"Saved agent state file not found at {h.AGENT_STATE_FILE}. Starting fresh training."
            )
        except Exception as e:
            log.warning(
                f"Could not load agent state from {h.AGENT_STATE_FILE}: {e}. Starting fresh."
            )
    else:
        log.info(
            f"No saved agent state file found at {h.AGENT_STATE_FILE}. Starting fresh training."
        )

    # --- Create Trainer ---
    trainer = MDPTrainer(
        agent=agent,
        map_size=h.MAP_SIZE,
        hero_param_ranges=h.HERO_RANGES,
        num_episodes=h.NUM_EPISODES,
        max_steps_per_episode=h.MAX_STEPS_PER_EPISODE,
        metrics_csv_path=h.METRICS_FILE,
        metric_update_steps=h.METRIC_UPDATE_FREQ,
        agent_save_path=h.AGENT_STATE_FILE,
    )

    # --- Run Training ---
    try:
        trainer.train()
        console.print(
            f"\n[bold green]Training complete. Check metrics ({h.METRICS_FILE}) and agent state ({h.AGENT_STATE_FILE}).[/bold green]"
        )
    except Exception as e:
        log.exception("An error occurred during training.")
        console.print(f"[bold red]Training failed: {e}[/bold red]")

    # --- Example: How to load and potentially use later ---
    # print("\n--- Loading Agent Example ---")
    # agent_loaded = MDPAgent( # Re-initialize with same *base* config
    #     size=MAP_SIZE,
    #     hero_tensor_size=HERO_TENSOR_SIZE,
    #     critic=simple_critic, # Critic needs to be provided again
    #     strategy=AGENT_STRATEGY,
    #     percentage_change=PERCENTAGE_MOD_LIMIT,
    # )
    # try:
    #     agent_loaded.load_state(AGENT_STATE_FILE)
    #     print(f"Loaded agent epsilon: {agent_loaded.epsilon:.4f}")
    #     # Now agent_loaded has the epsilon value from the end of training
    #     # You could potentially run more episodes or use it for inference/generation
    # except FileNotFoundError:
    #     print(f"Could not find saved state file {AGENT_STATE_FILE} to load.")
    # except Exception as e:
    #      print(f"Error loading saved state: {e}")
