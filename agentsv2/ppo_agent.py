# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

# Import the distribution class with an alias
from torch.distributions import Categorical as TorchDistributionCategorical
import torch.nn.functional as F
from daedalus.critics.critic_approximator import (
    CriticConfig,
)  # Keep if used elsewhere, unused in trainer

# TorchRL components
# Import the spec classes, aliasing the spec Categorical
from torchrl.data import DiscreteTensorSpec, Categorical as TorchRLCategoricalSpec
from torchrl.envs.common import EnvBase
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor
from tensordict import TensorDict

# Standard libraries
import os
import csv
import time
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from typing import Dict, Tuple, Optional, Any, Iterator

# Daedalus specific imports
# Ensure these paths are correct relative to where you run the script
from daedalus.models.neural_network import DaedalusActionPredictor
from daedalus.utils.environment import DaedalusEnvironment
import daedalus.utils.constants as c


# --- Value Network ---
class ValueNetwork(nn.Module):
    """Simple MLP Critic Network for PPO."""

    def __init__(self, in_size: int):
        """Initializes the Value Network layers."""
        super().__init__()
        self.in_size = in_size
        # Define keys expected/produced by this network
        self.in_keys = ["observation"]  # Input key expected from TensorDict
        self.out_keys = ["state_value"]  # Output key written to TensorDict

        self.input_layer = nn.Linear(in_size, 256)
        self.hidden_1 = nn.Linear(256, 512)
        self.hidden_2 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, 1)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass to estimate state value."""
        x = tensordict[self.in_keys[0]]
        x = F.tanh(self.input_layer(x))
        x = F.tanh(self.hidden_1(x))
        x = F.tanh(self.hidden_2(x))
        value = self.output_layer(x)
        tensordict.set(self.out_keys[0], value)
        return tensordict


# --- PPO Trainer ---
class PPOTrainer:
    """PPO Trainer optimized for CUDA usage with Daedalus environment."""

    # Type hints for core components initialized later
    env: DaedalusEnvironment
    policy: ProbabilisticActor
    critic_model: ValueNetwork
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    gae_estimator: GAE
    loss_module: ClipPPOLoss
    collector: SyncDataCollector
    replay_buffer: ReplayBuffer

    def __init__(
        self,
        env_mode: str,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        critic_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        num_episodes: int = 1000,
        steps_per_episode: int = 1024,
        update_interval_steps: int = 256,
        device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints_cuda",
        log_dir: str = "logs_cuda",
        map_size: Tuple[int, int] = (12, 12),
        env_critic_path: Optional[str] = None,
    ):
        """Initializes the PPO Trainer configuration and components."""
        self.config = locals()  # Store config args
        del self.config["self"]

        self.device = self._setup_device(device_str)
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.frames_per_batch = batch_size * update_interval_steps
        self.total_frames = steps_per_episode * num_episodes
        self.update_interval_steps = update_interval_steps
        self.ppo_epochs = ppo_epochs
        self.max_grad_norm = max_grad_norm

        self.num_updates_per_episode = self._calculate_updates_per_episode(
            steps_per_episode, self.frames_per_batch
        )
        # Adjust total frames if steps_per_episode wasn't divisible
        # Re-calculate based on the actual number of updates
        actual_steps_per_episode = self.num_updates_per_episode * self.frames_per_batch
        if actual_steps_per_episode != steps_per_episode:
            rprint(
                f"[WARN] Steps per episode adjusted to {actual_steps_per_episode} to be multiple of frames per batch."
            )
            self.steps_per_episode = actual_steps_per_episode
            self.total_frames = self.steps_per_episode * num_episodes

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self._setup_directories()

        self._setup_components(
            env_mode,
            map_size,
            env_critic_path,
            gamma,
            gae_lambda,
            learning_rate,
            clip_epsilon,
            entropy_coef,
            critic_coef,
        )

        self.metrics = self._reset_metrics()
        self.console = Console()
        rprint(
            f"[bold blue]Trainer initialized. Running on device: {self.device}[/bold blue]"
        )

    # --- Initialization Helpers ---

    def _setup_device(self, device_str: str) -> torch.device:
        """Determines and returns the torch device."""
        if device_str.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_str)
        if device_str.startswith("cuda"):
            rprint(
                "[yellow]CUDA requested but not available. Falling back to CPU.[/yellow]"
            )
        return torch.device("cpu")

    def _setup_directories(self) -> None:
        """Creates checkpoint and log directories if they don't exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _calculate_updates_per_episode(
        self, steps_per_ep: int, frames_per_batch: int
    ) -> int:
        """Calculates the number of updates per episode, using ceiling division."""
        if frames_per_batch <= 0:
            raise ValueError("frames_per_batch must be positive.")
        # Ensure at least one update
        return max(1, (steps_per_ep + frames_per_batch - 1) // frames_per_batch)

    def _setup_components(
        self,
        env_mode: str,
        map_size: Tuple[int, int],
        env_critic_path: Optional[str],
        gamma: float,
        gae_lambda: float,
        learning_rate: float,
        clip_epsilon: float,
        entropy_coef: float,
        critic_coef: float,
    ) -> None:
        """Initializes all major training components."""
        self._init_environment(env_mode, map_size, env_critic_path)
        self._init_models()
        self._init_policy(env_mode)
        self._init_torchrl_modules(
            gamma, gae_lambda, clip_epsilon, entropy_coef, critic_coef
        )
        self._init_optimizers(learning_rate)
        self._init_collector()
        self._init_replay_buffer()

    def _init_environment(
        self, mode: str, map_size: Tuple[int, int], critic_path: Optional[str]
    ) -> None:
        """Initializes the Daedalus environment."""
        self.env = DaedalusEnvironment(
            mode=mode,
            batch_size=self.batch_size,
            device=self.device,
            map_size=map_size,
            critic_path=critic_path,
            max_steps=self.update_interval_steps,
        )

    def _init_models(self) -> None:
        """Initializes the actor and critic neural network models."""
        if not hasattr(self.env, "observation_spec") or not hasattr(
            self.env, "action_spec"
        ):
            raise RuntimeError(
                "Environment specs not initialized before model creation."
            )

        obs_size = self.env.observation_spec["observation"].shape[-1]
        # Use aliased spec class for check
        if not isinstance(
            self.env.action_spec, (DiscreteTensorSpec, TorchRLCategoricalSpec)
        ):
            raise TypeError(
                f"Expected DiscreteTensorSpec or TorchRLCategoricalSpec action_spec, got {type(self.env.action_spec)}"
            )
        action_size = self.env.action_spec.space.n

        self.actor_model = DaedalusActionPredictor(obs_size, action_size).to(
            self.device
        )
        self.critic_model = ValueNetwork(obs_size).to(self.device)

    def _init_policy(self, env_mode: str) -> None:
        """Initializes the probabilistic actor policy."""
        if env_mode.upper() not in c.POSSIBLE_MODES:
            raise ValueError(f"Unsupported env_mode: {env_mode}")

        self.policy = ProbabilisticActor(
            module=self.actor_model,
            spec=self.env.action_spec,
            in_keys=["logits"],
            out_keys=["action"],
            # Use aliased distribution class
            distribution_class=TorchDistributionCategorical,
            return_log_prob=True,
        ).to(self.device)

    def _init_torchrl_modules(
        self,
        gamma: float,
        gae_lambda: float,
        clip_epsilon: float,
        entropy_coef: float,
        critic_coef: float,
    ) -> None:
        """Initializes GAE estimator and PPO loss module."""
        self.gae_estimator = GAE(
            gamma=gamma,
            lmbda=gae_lambda,
            value_network=self.critic_model,
            average_gae=True,
        )
        self.loss_module = ClipPPOLoss(
            actor=self.policy,
            critic=self.critic_model,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            value_coef=critic_coef,
            normalize_advantage=True,
            loss_critic_type="l2",
        ).to(self.device)

    def _init_optimizers(self, learning_rate: float) -> None:
        """Initializes Adam optimizers for actor and critic."""
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic_model.parameters(), lr=learning_rate
        )

    def _init_collector(self) -> None:
        """Initializes the synchronous data collector."""
        self.collector = SyncDataCollector(
            create_env_fn=self.env,
            policy=self.policy,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            device=self.device,
            storing_device=self.device,
            max_frames_per_traj=-1,
        )

    def _init_replay_buffer(self) -> None:
        """Initializes the replay buffer."""
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.frames_per_batch, device=self.device
            ),
            sampler=SamplerWithoutReplacement(),
            # Ensure batch_size is at least 1 and less than or equal to total frames stored
            batch_size=max(1, min(self.frames_per_batch, self.frames_per_batch // 4)),
        )

    # --- Training Loop ---

    def train(
        self, checkpoint_interval: int = 50, resume_from: Optional[str] = None
    ) -> None:
        """Main training loop over episodes."""
        start_episode = self._maybe_load_checkpoint(resume_from)
        total_start_time = time.time()
        collector_iterator = iter(self.collector)

        for episode in range(start_episode, self.num_episodes):
            episode_start_time = time.time()
            # Reset metrics, ensuring total_frames is preserved
            current_total_frames = (
                self.metrics["total_frames"] if hasattr(self, "metrics") else 0
            )
            self.metrics = self._reset_metrics()
            self.metrics["total_frames"] = current_total_frames
            self.metrics["episode"] = episode

            # Run data collection and updates for the episode
            self._run_episode_updates(episode, collector_iterator)

            episode_time = time.time() - episode_start_time
            self._log_episode_summary(episode, episode_time)
            if episode % checkpoint_interval == 0 or episode == self.num_episodes - 1:
                self._save_checkpoint(episode, self.metrics["total_frames"])

        self.collector.shutdown()
        total_time = time.time() - total_start_time
        self.console.print(
            f"\n[bold blue]Training finished in {total_time:.2f} seconds.[/bold blue]"
        )

    def _run_episode_updates(
        self, episode: int, collector_iterator: Iterator[TensorDict]
    ) -> None:
        """Runs the data collection and PPO update cycle for one episode."""
        pbar = tqdm(
            range(self.num_updates_per_episode),
            desc=f"Episode {episode}/{self.num_episodes-1}",
        )

        for update_idx in pbar:
            try:
                # 1. Collect data for one update cycle
                data_batch = next(collector_iterator)
                current_frames = data_batch.numel()
                self.metrics["total_frames"] += current_frames

                if current_frames == 0:
                    rprint(
                        f"[yellow]Warning: Collected empty data batch. Skipping update.[/yellow]"
                    )
                    continue

                # 2. Calculate GAE on the collected batch *before* storing
                processed_data_batch = self._calculate_gae_on_batch(data_batch)
                if processed_data_batch is None:  # GAE calculation failed
                    rprint(
                        f"[red]Error: GAE calculation failed for batch {update_idx+1}. Skipping update.[/red]"
                    )
                    continue  # Skip to next update cycle

                # 3. Add the *processed* batch (with GAE results) to the buffer
                self.replay_buffer.extend(processed_data_batch)

                # 4. Run PPO epochs using data now in the buffer
                if len(self.replay_buffer) > 0:
                    loss_metrics = self._run_ppo_epochs()  # Samples from buffer
                    self._update_metrics(
                        loss_metrics, processed_data_batch
                    )  # Log based on processed batch
                    pbar.set_postfix(
                        {
                            "avg_reward": (
                                f"{np.mean(self.metrics['episode_rewards']):.2f}"
                                if self.metrics["episode_rewards"]
                                else "N/A"
                            ),
                            "pol_loss": f"{loss_metrics['policy_loss']:.3f}",
                            "val_loss": f"{loss_metrics['value_loss']:.3f}",
                            "frames": self.metrics["total_frames"],
                        }
                    )
                else:
                    rprint(
                        f"[yellow]Warning: Buffer empty after extend. Skipping PPO epochs.[/yellow]"
                    )

                # 5. Clear buffer for next on-policy batch
                self.replay_buffer.empty()

            except StopIteration:
                rprint("[yellow]Collector iterator finished early.[/yellow]")
                break
            except Exception as e:
                rprint(
                    f"[bold red]Error during episode update {update_idx+1}: {e}[/bold red]"
                )
                import traceback

                traceback.print_exc()
                break

    def _calculate_gae_on_batch(self, data_batch: TensorDict) -> Optional[TensorDict]:
        """Calculates GAE on a TensorDict batch, returning the TD with results or None on error."""
        try:
            # Ensure reward shape is correct for GAE calculation
            reward_key = ("next", "reward")
            if reward_key in data_batch:
                if data_batch[reward_key].ndim == len(data_batch.batch_size):
                    data_batch[reward_key] = data_batch[reward_key].unsqueeze(-1)
            else:
                rprint(f"[red]Error: Key {reward_key} missing for GAE in batch.[/red]")
                return None

            # Calculate GAE - modifies data_batch in-place
            with torch.no_grad():
                self.gae_estimator(data_batch)

            # Verify GAE results exist before returning
            if (
                "advantage" not in data_batch.keys()
                or "value_target" not in data_batch.keys()
            ):
                rprint(
                    "[red]Error: GAE did not produce 'advantage' or 'value_target'.[/red]"
                )
                return None

            return data_batch

        except KeyError as e:
            rprint(f"[bold red]KeyError during GAE calculation: {e}.[/bold red]")
            rprint("Batch Keys:", data_batch.keys(include_nested=True))
            return None
        except RuntimeError as e:
            # Handle potential shape mismatch error during GAE itself
            if "must share a unique shape" in str(e):
                rprint(
                    f"[bold red]RuntimeError during GAE (Shape Mismatch): {e}[/bold red]"
                )
                rprint("--- Shapes at time of GAE Shape Error ---")
                keys_to_chk = [
                    ("next", "reward"),
                    ("next", "done"),
                    ("state_value",),
                    ("next", "state_value"),
                ]
                for k in keys_to_chk:
                    try:
                        rprint(f"Shape of {k}: {data_batch[k].shape}")
                    except KeyError:
                        rprint(f"Key {k} not found.")
                rprint("---------------------------------------")
            else:
                rprint(f"[bold red]RuntimeError during GAE calculation: {e}[/bold red]")
            import traceback

            traceback.print_exc()
            return None
        except Exception as e:
            rprint(f"[bold red]Unexpected Error during GAE calculation: {e}[/bold red]")
            import traceback

            traceback.print_exc()
            return None

    def _run_ppo_epochs(self) -> Dict[str, float]:
        """Runs PPO update epochs using mini-batches from the replay buffer."""
        policy_losses, value_losses, entropy_losses = [], [], []


        for epoch in range(self.ppo_epochs): # Use variable name 'epoch'
            try:
                # The sampler inside the buffer handles iterating through mini-batches
                batch_count = 0
                for mini_batch in self.replay_buffer:
                    batch_count += 1
                    mini_batch = mini_batch.to(self.device)
                    # Check necessary keys exist (should, from _calculate_gae_on_batch)
                    if "advantage" not in mini_batch.keys() or "value_target" not in mini_batch.keys():
                        rprint(f"[red]Epoch {epoch+1}: Missing GAE keys in sampled mini-batch. Keys: {mini_batch.keys()}[/red]")
                        continue # Skip this problematic mini-batch

                    loss_dict = self.loss_module(mini_batch)
                    total_loss = loss_dict["loss_objective"] + loss_dict["loss_critic"] + loss_dict["loss_entropy"]

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                    policy_losses.append(loss_dict["loss_objective"].item())
                    value_losses.append(loss_dict["loss_critic"].item())
                    entropy_losses.append(loss_dict["loss_entropy"].item())

                # If the inner loop didn't run at all, it means no mini-batches were sampled
                if batch_count == 0 and epoch == 0: # Check only on first epoch for clarity
                     rprint(f"[yellow]Warning: No mini-batches sampled during PPO epoch {epoch+1}. Buffer size might be too small or sampler issue.[/yellow]")
                     # Check len again here if needed: rprint(f"Buffer len: {len(self.replay_buffer)}")


            except Exception as e:
                rprint(f"[bold red]Error during PPO epoch {epoch+1}: {e}[/bold red]")
                import traceback; traceback.print_exc()
                continue # Try next epoch

        avg_policy_loss = np.mean(policy_losses) if policy_losses else 0.0
        avg_value_loss = np.mean(value_losses) if value_losses else 0.0
        avg_entropy_loss = np.mean(entropy_losses) if entropy_losses else 0.0
        return { "policy_loss": avg_policy_loss, "value_loss": avg_value_loss, "entropy_loss": avg_entropy_loss }

    # --- Metrics and Logging ---

    def _reset_metrics(self) -> Dict[str, Any]:
        """Resets the metrics dictionary for a new episode."""
        # Persist total frames across resets
        current_total_frames = (
            self.metrics["total_frames"]
            if hasattr(self, "metrics") and "total_frames" in self.metrics
            else 0
        )
        return {
            "episode": 0,
            "total_frames": current_total_frames,
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
        }

    def _update_metrics(
        self, loss_metrics: Dict[str, float], data_batch: TensorDict
    ) -> None:
        """Updates metrics lists with losses and extracts episode rewards/lengths."""
        self.metrics["policy_loss"].append(loss_metrics["policy_loss"])
        self.metrics["value_loss"].append(loss_metrics["value_loss"])
        self.metrics["entropy_loss"].append(loss_metrics["entropy_loss"])

        next_td = data_batch.get("next", None)
        if (
            next_td is not None
            and "reward" in next_td.keys()
            and "done" in next_td.keys()
        ):
            # Ensure reward has trailing dim for consistency if needed elsewhere, though not needed for sum
            rewards = next_td["reward"].squeeze(-1)  # Use squeezed version for sum
            dones = next_td["done"].squeeze(-1)  # Use squeezed version

            if rewards.shape[0] != self.batch_size or dones.shape[0] != self.batch_size:
                rprint(
                    f"[yellow]Warning: Reward/Done shape mismatch. Skipping metric update.[/yellow]"
                )
                return

            for env_idx in range(self.batch_size):
                env_dones_flat = dones[env_idx].reshape(-1)
                done_steps = torch.where(env_dones_flat)[0]
                if len(done_steps) > 0:
                    ep_len = min(done_steps[0].item() + 1, rewards.shape[1])
                    ep_reward = rewards[env_idx, :ep_len].sum().item()
                    self.metrics["episode_rewards"].append(ep_reward)
                    self.metrics["episode_lengths"].append(ep_len)

    def _log_episode_summary(self, episode: int, episode_time: float) -> None:
        """Calculates average metrics and logs the episode summary."""
        avg_reward = (
            np.mean(self.metrics["episode_rewards"])
            if self.metrics["episode_rewards"]
            else 0.0
        )
        avg_length = (
            np.mean(self.metrics["episode_lengths"])
            if self.metrics["episode_lengths"]
            else 0.0
        )
        avg_pol_loss = (
            np.mean(self.metrics["policy_loss"]) if self.metrics["policy_loss"] else 0.0
        )
        avg_val_loss = (
            np.mean(self.metrics["value_loss"]) if self.metrics["value_loss"] else 0.0
        )
        avg_ent_loss = (
            np.mean(self.metrics["entropy_loss"])
            if self.metrics["entropy_loss"]
            else 0.0
        )

        self._log_metrics_to_csv(
            episode=episode,
            avg_reward=avg_reward,
            avg_length=avg_length,
            avg_policy_loss=avg_pol_loss,
            avg_value_loss=avg_val_loss,
            avg_entropy_loss=avg_ent_loss,
            time_taken=episode_time,
            total_frames=self.metrics["total_frames"],
        )
        self._print_episode_summary_table(
            episode,
            avg_reward,
            avg_length,
            avg_pol_loss,
            avg_val_loss,
            avg_ent_loss,
            episode_time,
        )
        self._visualize_final_maps(episode)  # Call visualization here

    def _log_metrics_to_csv(self, **kwargs) -> None:
        """Logs key-value metrics to a CSV file."""
        filename = os.path.join(self.log_dir, "training_metrics.csv")
        file_exists = os.path.isfile(filename)
        fieldnames = [
            "episode",
            "avg_reward",
            "avg_length",
            "avg_policy_loss",
            "avg_value_loss",
            "avg_entropy_loss",
            "time_taken",
            "total_frames",
        ]
        filtered_kwargs = {k: kwargs.get(k, None) for k in fieldnames}

        with open(filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(filename) == 0:
                writer.writeheader()
            writer.writerow(filtered_kwargs)

    def _print_episode_summary_table(
        self,
        episode: int,
        avg_reward: float,
        avg_length: float,
        avg_pol_loss: float,
        avg_val_loss: float,
        avg_ent_loss: float,
        episode_time: float,
    ) -> None:
        """Prints the episode summary using a rich table."""
        self.console.print(f"\n[bold green]Episode {episode} Summary[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Value")
        table.add_row("Avg Reward", f"{avg_reward:.3f}")
        table.add_row("Avg Length", f"{avg_length:.1f}")
        table.add_row("Avg Policy Loss", f"{avg_pol_loss:.4f}")
        table.add_row("Avg Value Loss", f"{avg_val_loss:.4f}")
        table.add_row("Avg Entropy Loss", f"{avg_ent_loss:.4f}")
        table.add_row("Time (s)", f"{episode_time:.2f}")
        table.add_row("Total Frames", f"{self.metrics['total_frames']}")
        self.console.print(table)

    # --- Checkpointing ---

    def _save_checkpoint(self, episode: int, total_frames: int) -> None:
        """Saves model and optimizer states to a checkpoint file."""
        checkpoint_data = {
            "episode": episode,
            "total_frames": total_frames,
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic_model.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "config": self.config,
        }
        filename = os.path.join(
            self.checkpoint_dir, f"checkpoint_ep{episode}_frames{total_frames}.pt"
        )
        try:
            torch.save(checkpoint_data, filename)
            self.console.print(f"[cyan]Checkpoint saved: {filename}[/cyan]")
        except Exception as e:
            rprint(f"[bold red]Error saving checkpoint {filename}: {e}[/bold red]")

    def _maybe_load_checkpoint(self, resume_from: Optional[str]) -> int:
        """Loads a checkpoint if path is provided, returns starting episode."""
        if resume_from:
            if not os.path.exists(resume_from):
                rprint(
                    f"[bold red]Error: Checkpoint not found: {resume_from}. Starting fresh.[/bold red]"
                )
                return 0
            try:
                return self._load_checkpoint_state(resume_from)
            except Exception as e:
                rprint(
                    f"[bold red]Error loading checkpoint {resume_from}: {e}. Starting fresh.[/bold red]"
                )
                import traceback

                traceback.print_exc()
                return 0
        return 0

    def _load_checkpoint_state(self, checkpoint_path: str) -> int:
        """Loads trainer state from a checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.critic_model.load_state_dict(checkpoint["critic_state_dict"])
        self.policy.to(self.device)
        self.critic_model.to(self.device)

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self._move_optimizer_state_to_device()

        episode = checkpoint.get("episode", -1) + 1
        if not hasattr(self, "metrics") or self.metrics is None:
            self.metrics = self._reset_metrics()
        self.metrics["total_frames"] = checkpoint.get("total_frames", 0)

        self.console.print(
            f"[green]Loaded checkpoint '{checkpoint_path}'. Resuming from episode {episode}.[/green]"
        )
        return episode

    def _move_optimizer_state_to_device(self) -> None:
        """Moves optimizer states (tensors) to the trainer's device."""
        for optimizer in [self.actor_optimizer, self.critic_optimizer]:
            if not optimizer.state:
                continue
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        try:
                            state[k] = v.to(self.device)
                        except Exception as e:
                            rprint(
                                f"[yellow]Warning: Could not move opt state {k} to {self.device}: {e}[/yellow]"
                            )

    # --- Visualization ---

    def _visualize_final_maps(self, episode: int) -> None:
        """Attempts to retrieve and visualize final maps from the environment."""
        # Tries direct access first, as get_attr failed previously
        try:
            final_maps = None
            # Attempt 1: Direct access
            if (
                hasattr(self.collector.env, "maps")
                and self.collector.env.maps is not None
            ):
                final_maps = self.collector.env.maps

            # Attempt 2: Use get_attr (Fallback, less likely to work based on past errors)
            elif hasattr(self.collector.env, "get_attr"):
                try:
                    final_maps_list = self.collector.env.get_attr("maps")
                    if final_maps_list and final_maps_list[0] is not None:
                        final_maps = final_maps_list[0]
                except AttributeError:
                    pass  # Ignore if get_attr fails

            # Visualize if maps were successfully retrieved
            if final_maps is not None:
                self.visualize_maps(
                    final_maps.clone(), f"Final Maps (Episode {episode} End)"
                )
            else:
                rprint(
                    "[yellow]Could not retrieve final maps for visualization.[/yellow]"
                )

        except Exception as e:
            rprint(f"[yellow]Error during map visualization process: {e}[/yellow]")

    # Use the ORIGINAL visualize_maps implementation as requested
    def visualize_maps(self, maps: torch.Tensor, title: str) -> None:
        """Visualize maps using rich. (Original Implementation)"""
        if maps.device != torch.device("cpu"):
            maps = maps.cpu()

        # Handle potential dimension issues robustly before accessing shapes
        if maps.ndim == 3 and maps.shape[0] >= 1:
            pass
        elif maps.ndim == 2:
            maps = maps.unsqueeze(0)
        else:
            rprint(
                f"[red]Error visualizing maps: Invalid dimensions {maps.shape}.[/red]"
            )
            return

        color_map = {
            0: "[black]0[/black]",
            1: "[white]1[/white]",
            2: "[red]2[/red]",
            3: "[red]3[/red]",
            4: "[red]4[/red]",
            5: "[red]5[/red]",
            6: "[green]6[/green]",
        }
        self.console.print(f"\n{title}")
        num_maps_to_show = min(3, maps.shape[0])

        for i in range(num_maps_to_show):
            # Add checks for valid map dimensions inside the loop
            if (
                i >= maps.shape[0]
                or maps.ndim != 3
                or maps.shape[1] <= 0
                or maps.shape[2] <= 0
            ):
                rprint(
                    f"[yellow]Skipping visualization for map index {i}: Invalid data.[/yellow]"
                )
                continue

            table = Table(title=f"Map {i+1}", show_lines=False, box=None)
            map_height, map_width = maps.shape[1], maps.shape[2]

            for j in range(map_height):
                row = []
                for k in range(map_width):
                    # Bounds check already done effectively by loop ranges
                    cell_value = maps[i, j, k].item()
                    row.append(color_map.get(cell_value, f"[cyan]{cell_value}[/cyan]"))
                table.add_row(*row)

            self.console.print(table)
            if i < num_maps_to_show - 1:
                self.console.print("")


# --- Example Usage ---
if __name__ == "__main__":
    # Using a relative path assumes 'latest_checkpoint.pth' is in the CWD when running the script
    critic_checkpoint_path = "latest_checkpoint.pth"

    # Verify the critic checkpoint path exists
    if not os.path.exists(critic_checkpoint_path):
        rprint(
            f"[bold red]Warning:[/bold red] Critic checkpoint not found at '{os.path.abspath(critic_checkpoint_path)}'. Environment will not use reward shaping critic."
        )
        critic_checkpoint_path = None  # Set to None if not found
    else:
        rprint(
            f"[green]Found environment critic checkpoint at: {os.path.abspath(critic_checkpoint_path)}[/green]"
        )

    trainer = PPOTrainer(
        env_mode="TURTLE",
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        critic_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        num_episodes=50000,
        steps_per_episode=2048,  # Frames collected per episode
        update_interval_steps=32,  # Steps per env per collection -> frames_per_batch = 64*32 = 2048
        device_str="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir="checkpoints",
        log_dir="logs",
        map_size=(12, 12),
        env_critic_path=critic_checkpoint_path,  # Use the potentially None path
    )
    trainer.train(checkpoint_interval=50)
