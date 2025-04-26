import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor

import os
import csv
import time
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from typing import Dict, List, Tuple, Optional, Any, Union

from daedalus.models.neural_network import DaedalusActionPredictor
from daedalus.utils.environment import DaedalusEnvironment
import daedalus.utils.constants as c

from daedalus.critics.critic_approximator import CriticConfig

class ValueNetwork(nn.Module):
    """Value network for PPO critic."""
    
    def __init__(self, in_size: int):
        super().__init__()
        self.in_size = in_size

        self.in_keys = ["observation"]
        
        self.input_layer = nn.Linear(in_size, 256)
        self.hidden_1 = nn.Linear(256, 512)
        self.hidden_2 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, 1)
        
    def forward(self, tensordict) -> torch.Tensor:
        x = tensordict["observation"]

        x = F.tanh(self.input_layer(x))
        x = F.tanh(self.hidden_1(x))
        x = F.tanh(self.hidden_2(x))
        y = self.output_layer(x)

        tensordict = tensordict.set("state_value", y)

        return tensordict


class PPOTrainer:
    """PPO Trainer for Daedalus environment."""
    
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
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        map_size: Tuple[int, int] = (12, 12),
        critic_path: Optional[str] = None,
        steps_per_episode: int = 256,
        update_interval: int = 256,
        num_episodes: int = 100
    ):
        self.num_episodes = num_episodes
        self.device = device
        self.batch_size = batch_size
        self.steps_per_episode = steps_per_episode
        self.update_interval = update_interval
        self.ppo_epochs = ppo_epochs
        self.max_grad_norm = max_grad_norm
        
        # Setup directories
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize environment
        self.env = DaedalusEnvironment(
            mode=env_mode,
            batch_size=batch_size,
            device=device,
            map_size=map_size,
            critic_path=critic_path,
            max_steps=steps_per_episode
        )
        
        # Initialize networks
        obs_size = self.env.obs_dim
        action_size = self.env.action_space
        
        self.actor = DaedalusActionPredictor(obs_size, action_size).to(device)
        self.critic = ValueNetwork(obs_size).to(device)
        
        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Setup PPO components
        if env_mode.upper() in c.POSSIBLE_MODES:
            self.policy = ProbabilisticActor(
                module=self.actor,
                spec=self.env.action_spec,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=Categorical,
                distribution_kwargs={},
                return_log_prob=True,
            )
        self.gae = GAE(
            gamma=gamma,
            lmbda=gae_lambda,
            value_network=self.critic,
            average_gae=True
        )
        
        self.loss_module = ClipPPOLoss(
            actor=self.policy,
            critic=self.critic,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            value_coef=critic_coef,
            normalize_advantage=True
        )
        
        
        # Setup data collector
        self.frames_per_batch = update_interval * batch_size
        self.total_frames = steps_per_episode * batch_size * num_episodes
        self._device = device 
        self.collector = SyncDataCollector(
            self.env,
            self.policy,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            device=device,
        )
        self.collector_iterator = self.collector.iterator()
        
        # Setup replay buffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(update_interval * batch_size),
            sampler=SamplerWithoutReplacement()
        )
        
        # Initialize metrics tracking
        self.metrics = {
            "episode": 0,
            "avg_reward": [],
            "min_reward": [],
            "max_reward": [],
            "avg_episode_length": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": []
        }
        
        # Setup console for rich output
        self.console = Console()
    
    def visualize_maps(self, maps: torch.Tensor, title: str) -> None:
        """Visualize maps using rich."""
        # Convert first 3 maps to colored representation
        color_map = {
            0: "[black]0[/black]",    # Gray
            1: "[white]1[/white]",  # White
            2: "[red]2[/red]",      # Red (enemy)
            3: "[red]3[/red]",      # Red (enemy)
            4: "[red]4[/red]",      # Red (enemy)
            5: "[red]5[/red]",      # Red (enemy)
            6: "[green]6[/green]"  # Bright green (door)
        }
        
        self.console.print(f"\n{title}")
        
        # Show first 3 maps
        for i in range(min(3, maps.shape[0])):
            table = Table(title=f"Map {i+1}", show_lines=False, box=None)
            
            # Add columns
            for j in range(maps.shape[2]):
                table.add_column(str(j))
            
            # Add rows
            for j in range(maps.shape[1]):
                row = []
                for k in range(maps.shape[2]):
                    cell_value = maps[i, j, k].item()
                    row.append(color_map.get(cell_value, f"[cyan]{cell_value}[/cyan]"))
                table.add_row(*row)
            
            self.console.print(table)
            print("")
    
    def _log_metrics(self, episode: int) -> None:
        """Log metrics to CSV file."""
        filename = os.path.join(self.log_dir, "training_metrics.csv")
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "episode", "avg_reward", "min_reward", "max_reward", 
                    "avg_episode_length", "policy_loss", "value_loss", "entropy"
                ])
        
        # Append metrics
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                episode,
                self.metrics["avg_reward"][-1],
                self.metrics["min_reward"][-1],
                self.metrics["max_reward"][-1],
                self.metrics["avg_episode_length"][-1],
                self.metrics["policy_loss"][-1] if self.metrics["policy_loss"] else 0,
                self.metrics["value_loss"][-1] if self.metrics["value_loss"] else 0,
                self.metrics["entropy"][-1] if self.metrics["entropy"] else 0
            ])
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "policy_state_dict": self.policy.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "episode": episode,
            "metrics": self.metrics
        }
        
        filename = os.path.join(self.checkpoint_dir, f"checkpoint_episode_{episode}.pt")
        torch.save(checkpoint, filename)
        self.console.print(f"[green]Checkpoint saved at episode {episode}[/green]")
    
    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return episode number."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        
        # Load policy if available in checkpoint
        if "policy_state_dict" in checkpoint:
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
        
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        
        self.metrics = checkpoint["metrics"]
        episode = checkpoint["episode"]
        
        self.console.print(f"[green]Loaded checkpoint from episode {episode}[/green]")
        return episode
    
    def update_policy(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy using PPO."""
        # Compute advantages and returns
        batch["next", "reward"] = torch.unsqueeze(batch["next", "reward"], 2)
        
        with torch.no_grad():
            output = self.gae(batch)
            advantages, returns  = output["advantage"], output["state_value"]
        
        # Add advantages and returns to batch
        batch["advantages"] = advantages
        batch["returns"] = returns
        
        # Track loss metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # Create mini-batches
        mini_batch_size = self.batch_size * self.update_interval // 4
        for _ in range(self.ppo_epochs):
            # Sample mini-batches
            for mini_batch in self.replay_buffer.sample(mini_batch_size):
                # Compute loss
                loss_vals = self.loss_module(mini_batch)
                loss = (
                    loss_vals["loss_objective"] +
                    loss_vals["loss_critic"] +
                    loss_vals["loss_entropy"]
                )
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                
                # Apply gradients
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Track losses
                policy_losses.append(loss_vals["loss_objective"].item())
                value_losses.append(loss_vals["loss_critic"].item())
                entropy_losses.append(loss_vals["loss_entropy"].item())
        
        # Return average loss metrics
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses)
        }
    
    def train(self, checkpoint_interval: int = 10, resume_from: Optional[str] = None) -> None:
        """Train the agent for specified number of episodes."""
        num_episodes = self.num_episodes
        start_episode = 0
        
        # Load checkpoint if specified
        if resume_from:
            start_episode = self._load_checkpoint(resume_from)
        
        # Main training loop
        for episode in range(start_episode, start_episode + num_episodes):
            self.metrics["episode"] = episode
            episode_rewards = []
            episode_lengths = []
            
            # Reset environment and collector
            env_state = self.env.reset()
            self.collector.reset()
            
            # Show initial maps
            self.visualize_maps(self.env.maps, "Initial Maps (Episode Start)")
            
            progress_bar = tqdm(range(self.steps_per_episode // self.update_interval), 
                               desc=f"Episode {episode}/{start_episode + num_episodes - 1}")
            
            # Run episode with multiple updates
            for update_step in progress_bar:
                try:
                    # Collect data
                    batch = next(self.collector_iterator)
                    
                    # Store batch in replay buffer
                    self.replay_buffer.extend(batch)
                    
                    # Update policy
                    loss_metrics = self.update_policy(batch)
                    
                    # Track rewards and episode length
                    rewards = batch["next", "reward"].view(-1, self.env.batch_size[0]).sum(0)
                    episode_rewards.append(rewards.mean().item())
                    
                    # Update progress bar with metrics
                    progress_bar.set_postfix({
                        "avg_reward": f"{np.mean(episode_rewards):.4f}",
                        "policy_loss": f"{loss_metrics['policy_loss']:.4f}",
                        "value_loss": f"{loss_metrics['value_loss']:.4f}"
                    })
                except StopIteration as e:
                    print(e)
                    self.console.print("[yellow]Collector iteration finished early[/yellow]")
                    break

            """
            self.collector = SyncDataCollector(
                self.env,
                self.policy,
                frames_per_batch=self.frames_per_batch,
                total_frames=self.total_frames,
                device=self._device,
            )
            self.collector_iterator = self.collector.iterator()
            """
            
            # Show final maps
            self.visualize_maps(self.env.maps, "Final Maps (Episode End)")
            
            # Record metrics
            self.metrics["avg_reward"].append(np.mean(episode_rewards))
            self.metrics["min_reward"].append(np.min(episode_rewards) if episode_rewards else 0)
            self.metrics["max_reward"].append(np.max(episode_rewards) if episode_rewards else 0)
            self.metrics["avg_episode_length"].append(np.mean(episode_lengths) if episode_lengths else self.steps_per_episode)
            
            if loss_metrics:
                self.metrics["policy_loss"].append(loss_metrics["policy_loss"])
                self.metrics["value_loss"].append(loss_metrics["value_loss"])
                self.metrics["entropy"].append(loss_metrics["entropy"])
            
            # Log metrics
            self._log_metrics(episode)
            
            # Display episode summary
            self.console.print(f"\n[bold]Episode {episode} Summary[/bold]")
            self.console.print(f"Average Reward: {self.metrics['avg_reward'][-1]:.4f}")
            self.console.print(f"Min/Max Reward: {self.metrics['min_reward'][-1]:.4f}/{self.metrics['max_reward'][-1]:.4f}")
            
            # Save checkpoint
            if episode % checkpoint_interval == 0 or episode == start_episode + num_episodes - 1:
                self._save_checkpoint(episode)


if __name__ == "__main__":
    # Example usage
    trainer = PPOTrainer(
        env_mode="TURTLE",
        batch_size=32,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        critic_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir="checkpoints",
        log_dir="logs",
        critic_path="latest_checkpoint.pth",
        map_size=(12, 12),
        steps_per_episode=256,
        update_interval=128,
        num_episodes=100,
    )
    
    trainer.train(
        checkpoint_interval=10
    )