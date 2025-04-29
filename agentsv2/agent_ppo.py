import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

import os
import csv
import time
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque, namedtuple

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskID,
)
from rich import print as rprint

# Assuming Daedalus environment and model imports are correct
from daedalus.models.neural_network import DaedalusActionPredictor
from daedalus.utils.environment import DaedalusEnvironment
import daedalus.utils.constants as c
from daedalus.critics.critic_approximator import CriticConfig

# Define a structure for storing trajectory steps
TrajectoryStep = namedtuple(
    "TrajectoryStep", ["observation", "action", "log_prob", "value", "reward", "done"]
)

# Define a structure for storing update results
UpdateMetrics = namedtuple(
    "UpdateMetrics", ["policy_loss", "value_loss", "entropy_loss"]
)

# --- Constants ---
LOGGING_WINDOW = 50  # Number of episodes for rolling averages
NUMERICAL_STABILITY_EPS = 1e-8
DEFAULT_LOG_LEVEL = logging.INFO


class ValueNetwork(nn.Module):
    """Value network (Critic) for PPO."""

    def __init__(self, in_size: int):
        super().__init__()
        self.in_size = in_size
        self.input_layer = nn.Linear(in_size, 256)
        self.hidden_1 = nn.Linear(256, 512)
        self.hidden_2 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Predicts the value of a given observation."""
        x = F.tanh(self.input_layer(observation))
        x = F.tanh(self.hidden_1(x))
        x = F.tanh(self.hidden_2(x))
        value = self.output_layer(x)
        return value


class PPOTrainer:
    """PPO Trainer for Daedalus environment using PyTorch without torchrl."""

    def __init__(
        self,
        env_mode: str,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        critic_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "ppo_checkpoints",
        log_dir: str = "ppo_logs",
        map_size: Tuple[int, int] = (12, 12),
        critic_path: Optional[str] = None,
        steps_per_episode: int = 256,
        update_interval: int = 128,
        num_episodes: int = 50000,
        mini_batch_factor: int = 4,
        log_level: int = DEFAULT_LOG_LEVEL,
        log_window_size: int = LOGGING_WINDOW,
    ):
        self.env_mode = env_mode
        self.num_episodes = num_episodes
        self.device = torch.device(device)
        self.env_batch_size = batch_size
        self.steps_per_episode = steps_per_episode
        self.update_interval = update_interval
        self.ppo_epochs = ppo_epochs
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.log_window_size = log_window_size

        self._setup_logging(log_level)
        self.logger.info(f"Initializing PPOTrainer on device: {self.device}")

        if self.update_interval % mini_batch_factor != 0:
            msg = (
                f"update_interval ({self.update_interval}) must be divisible by "
                f"mini_batch_factor ({mini_batch_factor})"
            )
            self.logger.error(msg)
            raise ValueError(msg)
        self.mini_batch_size = (
            self.env_batch_size * self.update_interval
        ) // mini_batch_factor
        self.logger.info(f"Update Interval: {self.update_interval} steps")
        self.logger.info(f"PPO Epochs per Update: {self.ppo_epochs}")
        self.logger.info(f"Minibatch Size: {self.mini_batch_size}")

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self._setup_directories()

        self.env = self._initialize_environment(env_mode, map_size, critic_path)
        self.actor, self.critic = self._initialize_networks()
        self.actor_optimizer, self.critic_optimizer = self._initialize_optimizers(
            learning_rate
        )

        self.rollout_buffer = deque(maxlen=self.update_interval * self.env_batch_size)
        self._initialize_metrics()

        self.console = Console()
        self.current_observation: Optional[torch.Tensor] = None
        self.last_observation_for_update: Optional[torch.Tensor] = None
        self.last_done_for_update: Optional[torch.Tensor] = None

    def _setup_logging(self, log_level: int) -> None:
        """Configures the logger."""
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger("PPOTrainer")

    def _setup_directories(self) -> None:
        """Creates necessary directories for checkpoints and logs."""
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
            self.logger.info(f"Log directory: {self.log_dir}")
        except OSError as e:
            self.logger.error(f"Failed to create directories: {e}")
            raise

    def _initialize_environment(
        self, env_mode: str, map_size: Tuple[int, int], critic_path: Optional[str]
    ) -> DaedalusEnvironment:
        """Initializes the Daedalus environment."""
        self.logger.info(
            f"Initializing Daedalus Environment (Mode: {env_mode}, Size: {map_size})"
        )
        try:
            env = DaedalusEnvironment(
                mode=env_mode,
                batch_size=self.env_batch_size,
                device=self.device.type,  # Env expects 'cuda' or 'cpu' string
                map_size=map_size,
                critic_path=critic_path,
                max_steps=self.steps_per_episode,
            )
            self.logger.info("Environment initialized successfully.")
            return env
        except Exception as e:
            self.logger.exception(f"Failed to initialize environment: {e}")
            raise

    def _initialize_networks(self) -> Tuple[DaedalusActionPredictor, ValueNetwork]:
        """Initializes the actor and critic networks."""
        self.logger.info("Initializing Actor and Critic networks.")
        try:
            obs_size = self.env.obs_dim
            action_size = self.env.action_space
            actor = DaedalusActionPredictor(obs_size, action_size).to(self.device)
            critic = ValueNetwork(obs_size).to(self.device)
            self.logger.info(f"Actor Network: {type(actor).__name__}")
            self.logger.info(f"Critic Network: {type(critic).__name__}")
            return actor, critic
        except Exception as e:
            self.logger.exception(f"Failed to initialize networks: {e}")
            raise

    def _initialize_optimizers(
        self, learning_rate: float
    ) -> Tuple[optim.Optimizer, optim.Optimizer]:
        """Initializes the optimizers for actor and critic."""
        self.logger.info(f"Initializing Optimizers (Adam, LR: {learning_rate})")
        try:
            actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=learning_rate, eps=NUMERICAL_STABILITY_EPS
            )
            critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=learning_rate, eps=NUMERICAL_STABILITY_EPS
            )
            return actor_optimizer, critic_optimizer
        except Exception as e:
            self.logger.exception(f"Failed to initialize optimizers: {e}")
            raise

    def _initialize_metrics(self) -> None:
        """Initializes dictionaries and deques for tracking metrics."""
        self.metrics = {
            "episode": 0,
            "all_avg_rewards": [],
            "all_min_rewards": [],
            "all_max_rewards": [],
            "all_avg_lengths": [],
            "all_policy_losses": [],
            "all_value_losses": [],
            "all_entropies": [],
        }
        # Deques for rolling averages shown in progress bar
        self.reward_deque = deque(maxlen=self.log_window_size)
        self.length_deque = deque(maxlen=self.log_window_size)
        self.policy_loss_deque = deque(maxlen=self.log_window_size)
        self.value_loss_deque = deque(maxlen=self.log_window_size)
        self.entropy_deque = deque(maxlen=self.log_window_size)

    def _reset_environment(self) -> torch.Tensor:
        """Resets the environment and returns the initial observation tensor."""
        try:
            initial_state_dict = self.env.reset()
            # Assuming reset returns a dict like {'observation': tensor, ...}
            # Adapt this if the environment returns something else
            if (
                isinstance(initial_state_dict, dict)
                and "observation" in initial_state_dict
            ):
                observation = initial_state_dict["observation"]
            elif isinstance(initial_state_dict, torch.Tensor):
                # Handle case where env might return just the tensor
                observation = initial_state_dict
            else:
                raise TypeError(
                    f"Unexpected environment reset output type: {type(initial_state_dict)}"
                )

            if not isinstance(observation, torch.Tensor):
                observation = torch.tensor(
                    observation, dtype=torch.float32, device=self.device
                )
            elif observation.device != self.device:
                observation = observation.to(self.device)

            if (
                observation.shape[0] != self.env_batch_size
                or len(observation.shape) < 2
            ):
                raise ValueError(
                    f"Unexpected observation shape after reset: {observation.shape}"
                )

            return observation

        except Exception as e:
            self.logger.exception(f"Error during environment reset: {e}")
            raise RuntimeError("Failed to reset environment") from e

    def _step_environment(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Steps the environment and processes the output."""
        try:
            # Environment might expect actions on CPU and as numpy array
            actions_np = actions.cpu().numpy()
            next_obs_raw, rewards_raw, dones_raw, truncateds_raw, infos = self.env.step(
                actions_np
            )

            # Ensure outputs are tensors on the correct device
            if isinstance(next_obs_raw, dict) and "observation" in next_obs_raw:
                next_observation = next_obs_raw["observation"]
            elif isinstance(next_obs_raw, (np.ndarray, torch.Tensor)):
                next_observation = next_obs_raw
            else:
                raise TypeError(
                    f"Unexpected next_observation type: {type(next_obs_raw)}"
                )

            if not isinstance(next_observation, torch.Tensor):
                next_observation = torch.tensor(
                    next_observation, dtype=torch.float32, device=self.device
                )
            elif next_observation.device != self.device:
                next_observation = next_observation.to(self.device)

            rewards = torch.tensor(rewards_raw, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones_raw, dtype=torch.bool, device=self.device)
            truncateds = torch.tensor(
                truncateds_raw, dtype=torch.bool, device=self.device
            )

            # Combine dones and truncateds for terminal condition in buffer logic if needed,
            # but use original 'dones' for GAE calculation mask.
            # terminals = dones | truncateds

            return next_observation, rewards, dones, truncateds, infos

        except Exception as e:
            self.logger.exception(f"Error during environment step: {e}")
            # Return dummy values indicating termination to prevent infinite loops
            dummy_obs = (
                torch.zeros_like(self.current_observation)
                if self.current_observation is not None
                else torch.zeros(
                    (self.env_batch_size, self.env.obs_dim), device=self.device
                )
            )
            dummy_rewards = torch.zeros(self.env_batch_size, device=self.device)
            dummy_dones = torch.ones(
                self.env_batch_size, dtype=torch.bool, device=self.device
            )
            dummy_truncateds = torch.zeros(
                self.env_batch_size, dtype=torch.bool, device=self.device
            )
            dummy_infos = {}
            return dummy_obs, dummy_rewards, dummy_dones, dummy_truncateds, dummy_infos

    def visualize_maps(self, maps: Optional[torch.Tensor], title: str) -> None:
        """Visualize maps using rich."""
        if maps is None:
            self.logger.warning(
                f"Cannot visualize maps for '{title}', maps tensor is None."
            )
            return

        color_map = {
            0: "[grey27]0[/grey27]",  # Empty/Wall
            1: "[white]1[/white]",  # Path
            2: "[red]2[/red]",  # Enemy
            3: "[red]3[/red]",  # Damaged Enemy?
            4: "[red]4[/red]",  # Dead Enemy?
            5: "[red]5[/red]",  # Player (if present)
            6: "[green]6[/green]",  # Door
        }

        self.console.print(f"\n{title}")

        num_maps_to_show = min(3, maps.shape[0])
        for i in range(num_maps_to_show):
            table = Table(
                title=f"Map {i+1}",
                show_header=False,
                show_lines=False,
                box=None,
                padding=0,
            )

            map_height, map_width = maps.shape[1], maps.shape[2]
            for _ in range(map_width):
                table.add_column()  # No header text needed

            for j in range(map_height):
                row = []
                for k in range(map_width):
                    cell_value = int(maps[i, j, k].item())
                    row.append(color_map.get(cell_value, f"[cyan]{cell_value}[/cyan]"))
                table.add_row(*row)

            self.console.print(table)
            rprint("")  # Use rich print for spacing

    def _log_metrics(self, episode: int) -> None:
        """Log metrics to CSV file."""
        filename = os.path.join(self.log_dir, "training_metrics.csv")
        is_new_file = not os.path.exists(filename)

        try:
            with open(filename, "a", newline="") as file:
                writer = csv.writer(file)
                if is_new_file:
                    self.logger.info(f"Creating new metrics log file: {filename}")
                    writer.writerow(
                        [
                            "episode",
                            "avg_reward",
                            "min_reward",
                            "max_reward",
                            "avg_episode_length",
                            "avg_policy_loss",
                            "avg_value_loss",
                            "avg_entropy",
                        ]
                    )

                # Get the metrics for the *last completed* episode
                avg_policy_loss = (
                    np.mean(self.metrics["all_policy_losses"][-1])
                    if self.metrics["all_policy_losses"]
                    and self.metrics["all_policy_losses"][-1]
                    else 0
                )
                avg_value_loss = (
                    np.mean(self.metrics["all_value_losses"][-1])
                    if self.metrics["all_value_losses"]
                    and self.metrics["all_value_losses"][-1]
                    else 0
                )
                avg_entropy = (
                    np.mean(self.metrics["all_entropies"][-1])
                    if self.metrics["all_entropies"]
                    and self.metrics["all_entropies"][-1]
                    else 0
                )

                writer.writerow(
                    [
                        episode,
                        self.metrics["all_avg_rewards"][-1],
                        self.metrics["all_min_rewards"][-1],
                        self.metrics["all_max_rewards"][-1],
                        self.metrics["all_avg_lengths"][-1],
                        avg_policy_loss,
                        avg_value_loss,
                        avg_entropy,
                    ]
                )
        except IOError as e:
            self.logger.error(f"Failed to write metrics to CSV {filename}: {e}")
        except IndexError:
            self.logger.warning(
                f"Attempted to log metrics for episode {episode}, but metrics lists are empty."
            )

    def _save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "episode": episode,
            "metrics": self.metrics,  # Save complete metrics history
        }

        filename = os.path.join(self.checkpoint_dir, f"checkpoint_episode_{episode}.pt")
        try:
            torch.save(checkpoint, filename)
            self.logger.info(
                f"Checkpoint saved successfully at episode {episode} to {filename}"
            )
            self.console.print(f"[green]Checkpoint saved at episode {episode}[/green]")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {filename}: {e}")

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return the next episode number to start from."""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(
                f"Checkpoint file not found: {checkpoint_path}. Starting training from scratch."
            )
            self.console.print(
                f"[yellow]Checkpoint file not found: {checkpoint_path}. Starting from scratch.[/yellow]"
            )
            return 0

        self.logger.info(f"Attempting to load checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )

            self.metrics = checkpoint.get("metrics", self.metrics)
            loaded_episode = checkpoint.get("episode", -1)
            start_episode = loaded_episode + 1

            # Repopulate deques for rolling averages from loaded metrics history
            if self.metrics["all_avg_rewards"]:
                self.reward_deque.extend(
                    self.metrics["all_avg_rewards"][-self.log_window_size :]
                )
            if self.metrics["all_avg_lengths"]:
                self.length_deque.extend(
                    self.metrics["all_avg_lengths"][-self.log_window_size :]
                )
            # Note: Need to handle nested lists for losses/entropy correctly
            # This part might need adjustment based on how losses are stored per episode
            if self.metrics["all_policy_losses"]:
                avg_losses = [
                    np.mean(ep_losses) if ep_losses else 0
                    for ep_losses in self.metrics["all_policy_losses"]
                ]
                self.policy_loss_deque.extend(avg_losses[-self.log_window_size :])
            if self.metrics["all_value_losses"]:
                avg_losses = [
                    np.mean(ep_losses) if ep_losses else 0
                    for ep_losses in self.metrics["all_value_losses"]
                ]
                self.value_loss_deque.extend(avg_losses[-self.log_window_size :])
            if self.metrics["all_entropies"]:
                avg_entropies = [
                    np.mean(ep_entropies) if ep_entropies else 0
                    for ep_entropies in self.metrics["all_entropies"]
                ]
                self.entropy_deque.extend(avg_entropies[-self.log_window_size :])

            self.logger.info(
                f"Checkpoint loaded successfully from episode {loaded_episode}. Resuming training from episode {start_episode}."
            )
            self.console.print(
                f"[green]Loaded checkpoint from episode {loaded_episode}. Resuming training.[/green]"
            )
            return start_episode

        except FileNotFoundError:
            self.logger.error(
                f"Checkpoint file not found at {checkpoint_path} despite existence check."
            )
            self.console.print(
                f"[red]Error: Checkpoint file disappeared: {checkpoint_path}[/red]"
            )
            return 0
        except Exception as e:
            self.logger.exception(e)
            return 0

    def select_action(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects actions, calculates log probabilities, and estimates values."""
        if observation is None:
            raise ValueError("Cannot select action, observation is None.")
        if observation.device != self.device:
            observation = observation.to(self.device)
        

        try:
            self.actor.eval()  # Set actor to evaluation mode for consistency
            self.critic.eval()  # Set critic to evaluation mode for consistency
            with torch.no_grad():
                logits = self.actor(observation)
                self.logits = logits
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = self.critic(observation).squeeze(-1)
            self.actor.train()  # Set back to training mode
            self.critic.train()  # Set back to training mode
            return actions, log_probs, values
        except Exception as e:
            self.logger.exception(
                f"Error during action selection/value estimation: {e}"
            )
            raise

    def calculate_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates Generalized Advantage Estimation (GAE) and returns."""
        # Ensure all tensors are on the correct device
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        next_values = next_values.to(self.device)
        dones = dones.to(self.device)

        num_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards, device=self.device)
        last_gae_lam = 0.0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                v_next = next_values
            else:
                v_next = values[t + 1]

            # Use 'dones' (true termination) for GAE mask
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * v_next * mask - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_gae_lam * mask
            last_gae_lam = advantages[t]

        returns = advantages + values
        return advantages, returns

    def update_policy(self) -> Optional[UpdateMetrics]:
        """Performs the PPO update using data collected in the rollout_buffer."""
        if len(self.rollout_buffer) < self.update_interval:
            self.logger.warning(
                "Update called with buffer size %d, less than update_interval %d. Skipping update.",
                len(self.rollout_buffer),
                self.update_interval,
            )
            self.console.print(
                "[yellow]Warning: update_policy called with incomplete buffer.[/yellow]"
            )
            return None

        if (
            self.last_observation_for_update is None
            or self.last_done_for_update is None
        ):
            self.logger.error(
                "Missing last observation or done state needed for GAE calculation. Skipping update."
            )
            return None

        self.logger.debug("Starting PPO policy update.")

        # 1. Prepare Data from Buffer
        try:
            observations = torch.stack(
                [step.observation for step in self.rollout_buffer]
            ).to(self.device)
            actions = torch.stack([step.action for step in self.rollout_buffer]).to(
                self.device
            )
            old_log_probs = torch.stack(
                [step.log_prob for step in self.rollout_buffer]
            ).to(self.device)
            values = torch.stack([step.value for step in self.rollout_buffer]).to(
                self.device
            )
            rewards = torch.stack([step.reward for step in self.rollout_buffer]).to(
                self.device
            )
            dones = torch.stack([step.done for step in self.rollout_buffer]).to(
                self.device
            )
        except Exception as e:
            self.logger.exception(f"Error stacking data from rollout buffer: {e}")
            self.rollout_buffer.clear()  # Clear potentially corrupted buffer
            return None

        # 2. Calculate GAE and Returns
        try:
            self.critic.eval()  # Ensure critic is in eval mode for final value estimate
            with torch.no_grad():
                next_values = self.critic(self.last_observation_for_update).squeeze(-1)
                # Mask value based on the *actual termination* flag of the last step
                next_values = next_values * (1.0 - self.last_done_for_update.float())
            self.critic.train()  # Back to train mode

            advantages, returns = self.calculate_gae(
                rewards, values, next_values, dones
            )

        except Exception as e:
            self.logger.exception(f"Error calculating GAE/Returns: {e}")
            self.rollout_buffer.clear()
            return None

        # 3. Flatten and Normalize Advantages
        num_samples = self.update_interval * self.env_batch_size
        try:
            observations = observations.view(num_samples, -1)
            actions = actions.view(num_samples)
            old_log_probs = old_log_probs.view(num_samples)
            advantages = advantages.view(num_samples)
            returns = returns.view(num_samples)
            returns = (returns - returns.mean()) / (
                returns.std() + NUMERICAL_STABILITY_EPS
            )
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + NUMERICAL_STABILITY_EPS
            )
        except Exception as e:
            self.logger.exception(f"Error reshaping or normalizing data: {e}")
            self.rollout_buffer.clear()
            return None

        all_policy_losses, all_value_losses, all_entropy_losses = [], [], []

        # 4. PPO Optimization Loop
        self.actor.train()
        self.critic.train()
        indices = np.arange(num_samples)

        try:
            for epoch in range(self.ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, num_samples, self.mini_batch_size):
                    end = start + self.mini_batch_size
                    mb_indices = indices[start:end]
                    if len(mb_indices) == 0:
                        continue

                    mb_obs = observations[mb_indices]
                    mb_actions = actions[mb_indices]
                    mb_old_log_probs = old_log_probs[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_returns = returns[mb_indices]

                    # --- Forward pass ---
                    logits = self.actor(mb_obs)
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                    new_values = self.critic(mb_obs).squeeze(-1)

                    # --- Calculate Losses ---
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(
                            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                        )
                        * mb_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(new_values, mb_returns)

                    # --- Optimization Step ---
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    actor_loss_component = policy_loss - self.entropy_coef * entropy
                    value_loss_component = self.critic_coef * value_loss
                    total_loss = (
                        actor_loss_component + value_loss_component
                    )  # Combine for potential full backward pass if needed

                    # Separate backward passes for clarity and potential debugging
                    actor_loss_component.backward(
                        retain_graph=False
                    )  # Can set retain_graph=False if value loss uses separate graph
                    value_loss_component.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm
                    )

                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                    all_policy_losses.append(policy_loss.item())
                    all_value_losses.append(value_loss.item())
                    all_entropy_losses.append(entropy.item())

        except Exception as e:
            self.logger.exception(
                f"Error during PPO optimization loop (Epoch {epoch}, Batch Start {start}): {e}"
            )
            # Clear buffer and return None to indicate update failure
            self.rollout_buffer.clear()
            return None

        self.rollout_buffer.clear()
        self.logger.debug("PPO policy update finished.")

        avg_policy_loss = np.mean(all_policy_losses) if all_policy_losses else 0
        avg_value_loss = np.mean(all_value_losses) if all_value_losses else 0
        avg_entropy_loss = np.mean(all_entropy_losses) if all_entropy_losses else 0

        return UpdateMetrics(
            policy_loss=avg_policy_loss,
            value_loss=avg_value_loss,
            entropy_loss=avg_entropy_loss,
        )

    def train(
        self, checkpoint_interval: int = 100, resume_from: Optional[str] = None
    ) -> None:
        """Main training loop."""
        self.logger.info(f"Starting PPO training for {self.num_episodes} episodes.")
        start_episode = 0
        if resume_from:
            start_episode = self._load_checkpoint(resume_from)
        if start_episode >= self.num_episodes:
            self.logger.warning(
                "Start episode %d is >= total episodes %d. No training needed.",
                start_episode,
                self.num_episodes,
            )
            return

        total_episodes = self.num_episodes

        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("Avg R: {task.fields[avg_r]:.2f}"),
            TextColumn("Avg Len: {task.fields[avg_l]:.1f}"),
            TextColumn("P Loss: {task.fields[p_loss]:.3f}"),
            TextColumn("V Loss: {task.fields[v_loss]:.3f}"),
            TimeRemainingColumn(),
        ]

        
        try:
            with Progress(
                *progress_columns, console=self.console, transient=False
            ) as progress:
                episode_task = progress.add_task(
                    "[cyan]Training Progress",
                    total=total_episodes,
                    completed=start_episode,
                    avg_r=0.0,
                    avg_l=0.0,
                    p_loss=0.0,
                    v_loss=0.0,  # Initial fields
                )

                for episode in range(start_episode, total_episodes):
                    self.current_observation = self._reset_environment()
                    previous_rewards = self.env.get_initial_rewards()

                    self.metrics["episode"] = episode
                    episode_rewards_sum = np.zeros(self.env_batch_size)
                    episode_true_lengths = np.zeros(
                        self.env_batch_size, dtype=int
                    )  # Length until done/truncated
                    episode_active = np.ones(
                        self.env_batch_size, dtype=bool
                    )  # Track active envs

                    ep_policy_losses, ep_value_losses, ep_entropies = [], [], []

                    self.visualize_maps(
                        getattr(self.env, "maps", None),
                        f"Initial Maps (Start of Training)",
                    )

                    # --- Rollout Phase ---
                    self.logger.debug(f"Starting Episode {episode}")
                    steps_collected_in_episode = 0

                    while steps_collected_in_episode < self.steps_per_episode:
                        # 1. Select action and get value estimate
                        actions, log_probs, values = self.select_action(
                            self.current_observation
                        )

                        # 2. Step the environment
                        next_observation, rewards, dones, truncateds, infos = (
                            self._step_environment(actions)
                        )
                        rewards, previous_rewards = rewards - previous_rewards, rewards
                        steps_collected_in_episode += 1
                        if steps_collected_in_episode == self.steps_per_episode - 1:
                            self.visualize_maps(
                                getattr(self.env, "maps", None),
                                f"Final Maps (Episode {episode} End)",
                            )

                        terminals = dones | truncateds  # Combined flag for resets etc.

                        # 3. Store transition
                        step_data = TrajectoryStep(
                            observation=self.current_observation.cpu(),  # Store on CPU to save GPU memory
                            action=actions.cpu(),
                            log_prob=log_probs.cpu(),
                            value=values.cpu(),
                            reward=rewards.cpu(),
                            done=dones.cpu(),  # Store original 'done' for GAE
                        )
                        self.rollout_buffer.append(step_data)

                        # 4. Update episode trackers
                        episode_rewards_sum += rewards.cpu().numpy() * episode_active / self.steps_per_episode
                        episode_true_lengths += 1 * episode_active
                        newly_finished = terminals.cpu().numpy() & episode_active
                        episode_active[newly_finished] = False  # Mark finished envs

                        # 5. Prepare for next step / Reset finished envs
                        # Handle resets: if an env finished, its next_observation is the reset obs
                        # The environment class should handle this internally based on dones/truncateds
                        self.current_observation = next_observation

                        # 6. Check if update interval reached
                        if len(self.rollout_buffer) == self.update_interval:
                            self.logger.debug(
                                f"Horizon of {self.update_interval} steps reached at Ep {episode}, Step {steps_collected_in_episode}."
                            )
                            self.console.print(
                                f"[dim]Horizon reached ({self.update_interval} steps). Performing PPO update...[/dim]"
                            )

                            # Store last observation/done needed for GAE bootstrap
                            self.last_observation_for_update = (
                                self.current_observation.clone()
                            )  # Important: clone tensor
                            self.last_done_for_update = (
                                dones.clone()
                            )  # Use 'done' flag before reset

                            # Perform PPO Update
                            update_metrics = (
                                self.update_policy()
                            )  # This clears the buffer

                            if update_metrics:
                                ep_policy_losses.append(update_metrics.policy_loss)
                                ep_value_losses.append(update_metrics.value_loss)
                                ep_entropies.append(update_metrics.entropy_loss)
                                self.policy_loss_deque.append(
                                    update_metrics.policy_loss
                                )
                                self.value_loss_deque.append(update_metrics.value_loss)
                                self.entropy_deque.append(update_metrics.entropy_loss)
                            else:
                                self.logger.warning(
                                    f"PPO update skipped or failed at Ep {episode}, Step {steps_collected_in_episode}."
                                )
                                # Append NaN or 0 to maintain list length if needed for logging
                                ep_policy_losses.append(np.nan)
                                ep_value_losses.append(np.nan)
                                ep_entropies.append(np.nan)

                        # If all envs finished the episode early (optional check)
                        if not episode_active.any():
                            self.logger.info(
                                f"All environments finished early in Episode {episode} after {np.max(episode_true_lengths)} steps."
                            )
                            # Break inner loop if desired, but fixed steps_per_episode is simpler
                            # break

                    # --- End of Episode ---
                    # Calculate and store episode metrics
                    avg_ep_reward = np.mean(episode_rewards_sum)
                    min_ep_reward = np.min(episode_rewards_sum)
                    max_ep_reward = np.max(episode_rewards_sum)
                    # Use the true lengths for reporting, average over envs that actually ran
                    valid_lengths = episode_true_lengths[episode_true_lengths > 0]
                    avg_ep_length = (
                        np.mean(valid_lengths) if len(valid_lengths) > 0 else 0
                    )

                    self.metrics["all_avg_rewards"].append(avg_ep_reward)
                    self.metrics["all_min_rewards"].append(min_ep_reward)
                    self.metrics["all_max_rewards"].append(max_ep_reward)
                    self.metrics["all_avg_lengths"].append(avg_ep_length)
                    self.metrics["all_policy_losses"].append(
                        [loss for loss in ep_policy_losses if not np.isnan(loss)]
                    )  # Store list of update losses
                    self.metrics["all_value_losses"].append(
                        [loss for loss in ep_value_losses if not np.isnan(loss)]
                    )
                    self.metrics["all_entropies"].append(
                        [loss for loss in ep_entropies if not np.isnan(loss)]
                    )

                    self.reward_deque.append(avg_ep_reward)
                    self.length_deque.append(avg_ep_length)
                    # Note: Loss/Entropy deques already updated during the step loop

                    # Log to CSV
                    self._log_metrics(episode)

                    # Update Progress Bar
                    avg_r_disp = (
                        np.mean(self.reward_deque) if self.reward_deque else 0.0
                    )
                    avg_l_disp = (
                        np.mean(self.length_deque) if self.length_deque else 0.0
                    )
                    avg_p_loss_disp = (
                        np.mean(self.policy_loss_deque)
                        if self.policy_loss_deque
                        else 0.0
                    )
                    avg_v_loss_disp = (
                        np.mean(self.value_loss_deque) if self.value_loss_deque else 0.0
                    )
                    progress.update(
                        episode_task,
                        advance=1,
                        refresh=True,
                        avg_r=avg_r_disp,
                        avg_l=avg_l_disp,
                        p_loss=avg_p_loss_disp,
                        v_loss=avg_v_loss_disp,
                    )

                    # Display episode summary in console
                    self.console.print(f"\n[bold]Episode {episode} Summary[/bold]")
                    self.console.print(
                        f"  Avg Reward: {avg_ep_reward:.3f} (Min: {min_ep_reward:.3f}, Max: {max_ep_reward:.3f})"
                    )
                    self.console.print(f"  Avg Length: {avg_ep_length:.2f}")
                    avg_ep_p_loss = (
                        np.mean(ep_policy_losses)
                        if ep_policy_losses and not all(np.isnan(ep_policy_losses))
                        else np.nan
                    )
                    avg_ep_v_loss = (
                        np.mean(ep_value_losses)
                        if ep_value_losses and not all(np.isnan(ep_value_losses))
                        else np.nan
                    )
                    self.console.print(
                        f"  Avg Policy Loss (updates): {avg_ep_p_loss:.4f}"
                    )
                    self.console.print(
                        f"  Avg Value Loss (updates): {avg_ep_v_loss:.4f}"
                    )

                    # Visualize final maps for the episode

                    # Save checkpoint periodically
                    if episode > 0 and episode % checkpoint_interval == 0:
                        self._save_checkpoint(episode)

            # --- End of Training ---
            final_desc = "[green]Training Finished"
            progress.update(episode_task, description=final_desc)

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user.")
            self.console.print("\n[yellow]Training interrupted by user.[/yellow]")
            final_desc = "[yellow]Training Interrupted"
            progress.update(episode_task, description=final_desc)
        except Exception as e:
            self.logger.exception("An unexpected error occurred during training.")
            self.console.print(
                f"\n[bold red]An unexpected error occurred: {e}[/bold red]"
            )
            final_desc = "[red]Training Failed"
            progress.update(episode_task, description=final_desc)
        finally:
            # Save final checkpoint regardless of how training ended
            last_episode = self.metrics["episode"]
            self.logger.info(f"Saving final checkpoint at episode {last_episode}.")
            self._save_checkpoint(last_episode)
            self.console.print(
                "[bold green]Training finished or stopped. Final checkpoint saved.[/bold green]"
            )
    
    def get_transition_data(self):
        for episode in range(1):
            self.current_observation = self._reset_environment()
            previous_rewards = self.env.get_initial_rewards()

            episode_rewards_sum = np.zeros(self.env_batch_size)
            episode_true_lengths = np.zeros(
                self.env_batch_size, dtype=int
            )  # Length until done/truncated
            episode_active = np.ones(
                self.env_batch_size, dtype=bool
            )  # Track active envs

            ep_policy_losses, ep_value_losses, ep_entropies = [], [], []

            self.visualize_maps(
                getattr(self.env, "maps", None),
                f"Initial Maps (Start of Training)",
            )

            # --- Rollout Phase ---
            self.logger.debug(f"Starting Episode {episode}")
            steps_collected_in_episode = 0

            actions = None
            rewards = previous_rewards
            logits = None
            position_final = self.env.current_positions[0]

            while steps_collected_in_episode < self.steps_per_episode+1:
                # 1. Select action and get value estimate
                if steps_collected_in_episode > 0:
                    actions, log_probs, values = self.select_action(
                        self.current_observation
                    )
                    logits = self.logits[0]

                    # 2. Step the environment
                    next_observation, rewards, dones, truncateds, infos = (
                        self._step_environment(actions)
                    )

                    if steps_collected_in_episode == self.steps_per_episode - 1:
                        self.visualize_maps(
                            getattr(self.env, "maps", None),
                            f"Final Maps (Episode {episode} End)",
                        )

                steps_collected_in_episode += 1
                yield (position_final.tolist() ,actions.tolist() if actions is not None else None, rewards[0].tolist(), logits.tolist() if logits is not None else None)
    
    def export_transition_data(self):
        transitions = []
        for posn, action, reward, log_probs in self.get_transition_data():
            transitions.append({
                "position" : posn,
                "action" : action,
                "reward" : reward,
                "log_probs": log_probs
            })

        data = {
            "initial_state" : self.env.maps[0].tolist(),
            "transitions" : transitions
        }

        import json
        
        json.dump(data, open("transitions.json", mode="w+"))


if __name__ == "__main__":
    trainer = PPOTrainer(
        env_mode="TURTLE",
        batch_size=8192 // 256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.85,
        clip_epsilon=0.2,
        critic_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir="ppo_daedalus_checkpoints",
        log_dir="ppo_daedalus_logs",
        critic_path="latest_checkpoint.pth",  # "latest_checkpoint.pth" # Set path if needed
        map_size=(12, 12),
        steps_per_episode=256,  # Total steps collected across envs per episode
        update_interval=32,  # Perform PPO update every 128 steps
        num_episodes=50000,
        mini_batch_factor=4,
        log_level=logging.INFO,  # Change to logging.DEBUG for more detail
        log_window_size=50,
    )

    # Example: Start training, save every 500 episodes
    # Optionally resume: resume_from="ppo_daedalus_checkpoints/checkpoint_episode_XXX.pt"
    trainer.export_transition_data()
    #trainer.train(checkpoint_interval=500, resume_from="ppo_daedalus_checkpoints/checkpoint_episode_779.pt")
