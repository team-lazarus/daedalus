# trainer.py
import torch
import numpy as np
import os
import time
import argparse
import yaml
from collections import deque
from typing import Callable, Dict, Any, Optional, List

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from tqdm import tqdm # Can use either rich Progress or tqdm

from config import TrainConfig, load_config_from_yaml, dump_config_to_yaml
from utils import get_device, save_checkpoint, load_checkpoint, format_map_rich
from environment import MapEnvironment
from agent import PPOAgent
from ppo import compute_advantages_gae, ppo_update
from model import ActorCritic # Only needed if loading checkpoint requires class definition


# Define the type hint for the external critic function
CriticFuncType = Callable[[np.ndarray, np.ndarray], torch.Tensor]

class PPOTrainer:
    """Orchestrates the PPO training process."""

    def __init__(self, config: TrainConfig, critic_func: CriticFuncType):
        """Initializes trainer components."""
        self.config = config
        self.critic_func = critic_func
        self.device = get_device(config.device)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(config.seed)

        self.env = MapEnvironment(config, self.device)
        self.agent = PPOAgent(config, self.device)

        self.console = Console()
        os.makedirs(config.output_dir, exist_ok=True)

        self.start_epoch = 0
        self.cumulative_reward = 0.0
        self.episode_rewards = deque(maxlen=100) # For tracking recent avg reward

        # Load checkpoint if provided
        if config.checkpoint_path:
            self._load_trainer_state(config.checkpoint_path)

        self.console.print(f"Using device: [bold {self.device.type}]{self.device}[/]")
        self.console.print(f"Output directory: [cyan]{config.output_dir}[/]")
        dump_config_to_yaml(config, os.path.join(config.output_dir, "config.yaml"))

    def _collect_rollouts(self) -> Dict[str, Any]:
        """Collects trajectories for one batch."""
        rollout_data = {
            "maps": [], "heroes": [], "actions": [], "log_probs": [],
            "rewards": [], "dones": [], "values": [], "prev_maps": []
        }
        map_tensor, hero_tensor = self.env.get_state() # Get initial state

        # Need to store initial state value for GAE calculation start
        with torch.no_grad():
            _, initial_value = self.agent.actor_critic(map_tensor, hero_tensor)
            rollout_data["values"].append(initial_value.squeeze().cpu()) # Store V(s_0)

        current_cumulative_reward = 0.0

        for _ in range(self.config.batch_size):
            map_tensor, hero_tensor = self.env.get_state() # Get current state S_t
            action, log_prob, value = self.agent.select_action(map_tensor, hero_tensor)

            # Execute action A_t in environment to get S_{t+1} and previous map
            next_map_tensor, next_hero_tensor, prev_map_state = self.env.step(action)

            # Get reward R_t using external critic function
            # Critic needs prev_state and modified_state as numpy arrays [N, H, W]
            # We are collecting step-by-step, so N=1
            # Ensure map tensors are correctly formatted for the critic
            # Assuming critic function handles device placement if needed
            # map_tensor shape: [1, 1, H, W] -> need [1, H, W] numpy
            map_np = map_tensor.squeeze(0).squeeze(0).cpu().numpy()
            next_map_np = next_map_tensor.squeeze(0).squeeze(0).cpu().numpy()
            reward = self.critic_func(np.expand_dims(prev_map_state, axis=0),
                                      np.expand_dims(next_map_np, axis=0)) # Pass as [1, H, W]
            reward = reward.squeeze().item() # Expects [1,] tensor, get scalar float

            # Store transition (S_t, H_t, A_t, log_prob_t, R_t, V(S_t))
            rollout_data["maps"].append(map_tensor.cpu()) # Store on CPU to save GPU memory
            rollout_data["heroes"].append(hero_tensor.cpu())
            rollout_data["actions"].append(torch.tensor(action, dtype=torch.long))
            rollout_data["log_probs"].append(log_prob.cpu())
            rollout_data["rewards"].append(torch.tensor(reward, dtype=torch.float32))
            rollout_data["dones"].append(torch.tensor(0.0, dtype=torch.float32)) # Assuming non-terminating env for now
            # Value V(S_t) was calculated during action selection
            rollout_data["values"].append(value.cpu())
            rollout_data["prev_maps"].append(prev_map_state) # Store prev numpy map if needed later

            current_cumulative_reward += reward

        # Get value of the final state S_N for GAE calculation
        with torch.no_grad():
            final_map, final_hero = self.env.get_state()
            _, final_value = self.agent.actor_critic(final_map, final_hero)
            rollout_data["values"].append(final_value.squeeze().cpu()) # Store V(S_N)

        # Stack collected data into tensors
        for key in ["maps", "heroes", "actions", "log_probs", "rewards", "dones", "values"]:
            # Special handling for maps: [B, 1, H, W]
            if key == "maps":
                 rollout_data[key] = torch.cat(rollout_data[key], dim=0)
            # Special handling for heroes: [B, Hero_dim]
            elif key == "heroes":
                 rollout_data[key] = torch.cat(rollout_data[key], dim=0)
            # Values have N+1 entries
            elif key == "values":
                 rollout_data[key] = torch.stack(rollout_data[key])
            else: # Others are [B]
                 rollout_data[key] = torch.stack(rollout_data[key])

        # Store the initial map of this rollout for printing comparison
        rollout_data["initial_map_for_print"] = rollout_data["prev_maps"][0]
        # Store the final map of this rollout for printing comparison
        rollout_data["final_map_for_print"] = self.env.current_map.copy()

        return rollout_data, current_cumulative_reward / self.config.batch_size

    def _save_trainer_state(self, epoch: int):
        """Saves the trainer's state."""
        state = {
            'epoch': epoch,
            'agent_state': self.agent.save_state(),
            'cumulative_reward': self.cumulative_reward,
            'episode_rewards': list(self.episode_rewards), # Save deque as list
            # Add anything else needed to resume training (e.g., RNG state)
        }
        filepath = os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch}.pth")
        save_checkpoint(state, filepath)

    def _load_trainer_state(self, filepath: str):
        """Loads the trainer's state."""
        checkpoint = load_checkpoint(filepath, self.device)
        if checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            self.agent.load_state(checkpoint['agent_state'])
            self.cumulative_reward = checkpoint.get('cumulative_reward', 0.0)
            saved_rewards = checkpoint.get('episode_rewards', [])
            self.episode_rewards = deque(saved_rewards, maxlen=self.episode_rewards.maxlen)
            self.console.print(f"Resuming training from epoch {self.start_epoch}")
        else:
            self.console.print("[yellow]Could not load checkpoint, starting from scratch.[/yellow]")


    def train(self):
        """Runs the main training loop."""
        self.console.print("[bold green]Starting PPO Training...[/bold green]")
        map_tensor, _ = self.env.reset() # Initial reset
        initial_map_print = map_tensor.squeeze().cpu().numpy() # Get initial map for first print

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[metrics]}"), # Custom metrics field
            console=self.console,
            transient=False, # Keep the bar after completion
        )

        task_id = progress.add_task("[cyan]Training Epochs", total=self.config.num_epochs, metrics="")

        start_time = time.time()
        with progress:
            for epoch in range(self.start_epoch, self.config.num_epochs):
                epoch_start_time = time.time()

                # Collect rollouts
                rollout_data, avg_reward_batch = self._collect_rollouts()
                self.episode_rewards.append(avg_reward_batch)
                self.cumulative_reward += rollout_data["rewards"].sum().item()

                # Compute advantages and value targets
                advantages, value_targets = compute_advantages_gae(
                    rewards=rollout_data["rewards"],
                    values=rollout_data["values"], # Has N+1 values
                    dones=rollout_data["dones"], # All zeros currently
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda
                )
                # Add advantages and targets to batch dict for update function
                rollout_data['advantages'] = advantages
                rollout_data['value_targets'] = value_targets

                # Perform PPO updates
                avg_policy_loss, avg_value_loss, avg_entropy = ppo_update(
                    self.agent, rollout_data, self.config, self.device
                )

                epoch_duration = time.time() - epoch_start_time
                avg_recent_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0

                # Update progress bar metrics
                metrics_str = (
                    f"AvgRew: {avg_recent_reward:.3f} | "
                    f"PolLoss: {avg_policy_loss:.3f} | "
                    f"ValLoss: {avg_value_loss:.3f} | "
                    f"Entropy: {avg_entropy:.3f}"
                )
                progress.update(task_id, advance=1, metrics=metrics_str)


                # Logging and Checkpointing
                if (epoch + 1) % self.config.save_interval == 0:
                    self.console.print(f"\n--- Epoch {epoch + 1} Summary ---")
                    self.console.print(f"Time: {epoch_duration:.2f}s")
                    self.console.print(f"Avg Reward (last 100 batches): {avg_recent_reward:.4f}")
                    self.console.print(f"Cumulative Reward: {self.cumulative_reward:.2f}")
                    self.console.print(f"Policy Loss: {avg_policy_loss:.4f}")
                    self.console.print(f"Value Loss: {avg_value_loss:.4f}")
                    self.console.print(f"Policy Entropy: {avg_entropy:.4f}")

                    # Print maps
                    if epoch == self.start_epoch: # First save interval
                         self.console.print(format_map_rich(initial_map_print, title="Initial Map (Epoch 0)"))
                    else:
                         # Print map from start of the *last completed* rollout batch for comparison
                         self.console.print(format_map_rich(rollout_data["initial_map_for_print"], title=f"Map Start (Epoch {epoch+1})"))

                    self.console.print(format_map_rich(rollout_data["final_map_for_print"], title=f"Map End (Epoch {epoch+1})"))

                    # Save checkpoint and metrics
                    self._save_trainer_state(epoch)
                    # Save metrics to a file (e.g., CSV or JSON)
                    metrics_data = {
                        'epoch': epoch + 1,
                        'avg_reward_batch': avg_reward_batch,
                        'avg_recent_reward': avg_recent_reward,
                        'cumulative_reward': self.cumulative_reward,
                        'policy_loss': avg_policy_loss,
                        'value_loss': avg_value_loss,
                        'entropy': avg_entropy,
                        'time_elapsed': time.time() - start_time
                    }
                    # Append to a metrics file
                    metrics_file = os.path.join(self.config.output_dir, "metrics.csv")
                    # (Implementation for appending to CSV/JSON not shown here for brevity)


        self.console.print(f"\n[bold green]Training finished in {time.time() - start_time:.2f} seconds.[/bold green]")


# --- Example Critic Function (Replace with your actual logic) ---
def example_critic_function(prev_states: np.ndarray, modified_states: np.ndarray) -> torch.Tensor:
    """
    Placeholder critic function. Calculates reward based on state changes.
    Input: prev_states [N, H, W], modified_states [N, H, W] (numpy arrays)
    Output: rewards [N,] (torch tensor on CPU/GPU)
    """
    rewards = []
    for prev, mod in zip(prev_states, modified_states):
        # Example: reward for changing a wall (0) to empty (1)
        change_reward = np.sum((prev == 0) & (mod == 1)) * 0.1
        # Example: penalty for creating walls
        penalty = np.sum((prev != 0) & (mod == 0)) * -0.05
        # Example: reward for placing enemies away from walls
        enemy_reward = 0
        enemy_indices = np.argwhere((prev != mod) & (mod >= 2) & (mod <= 5)) # Find new enemies
        for r, c in enemy_indices:
            is_near_wall = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < mod.shape[0] and 0 <= nc < mod.shape[1]:
                        if mod[nr, nc] == 0: # Is near a wall
                            is_near_wall = True
                            break
                if is_near_wall: break
            if not is_near_wall:
                enemy_reward += 0.02 # Reward placing enemy not near wall


        total_reward = change_reward + penalty + enemy_reward
        rewards.append(total_reward)

    # Return as a torch tensor on the appropriate device (assuming CPU here)
    return torch.tensor(rewards, dtype=torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent for Map Generation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    # Allow overriding specific config values via command line (optional)
    parser.add_argument("--mode", type=str, choices=["narrow", "turtle", "wide"], help="Override environment mode.")
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs.")
    parser.add_argument("--load", type=str, help="Path to checkpoint file to load.")
    parser.add_argument("--output", type=str, help="Directory to save outputs.")
    parser.add_argument("--temp", type=float, help="Override action selection temperature (0-1).")


    args = parser.parse_args()

    # Load base configuration
    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = TrainConfig() # Use defaults

    # Override config with command-line arguments if provided
    if args.mode: config.mode = args.mode
    if args.lr: config.lr = args.lr
    if args.epochs: config.num_epochs = args.epochs
    if args.load: config.checkpoint_path = args.load
    if args.output: config.output_dir = args.output
    if args.temp is not None: config.temperature = args.temp # Handle 0.0 case


    # --- IMPORTANT: Replace this with your actual critic function ---
    critic = example_critic_function
    # --------------------------------------------------------------

    trainer = PPOTrainer(config, critic_func=critic)
    trainer.train()