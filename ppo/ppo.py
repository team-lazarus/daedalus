# ppo.py
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple

from config import TrainConfig
from model import ActorCritic


def compute_advantages_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes GAE advantages and value targets."""
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    num_steps = len(rewards)
    # Ensure values has shape [num_steps + 1] including the value of the state after the last action
    # values should contain V(s_0), V(s_1), ..., V(s_N)
    # rewards should contain r_0, r_1, ..., r_{N-1}
    # dones should contain d_0, d_1, ..., d_{N-1} (0 if not terminal, 1 if terminal)

    if values.shape[0] != rewards.shape[0] + 1:
        raise ValueError(
            f"Values tensor shape mismatch. Expected {rewards.shape[0]+1}, got {values.shape[0]}"
        )

    for step in reversed(range(num_steps)):
        # If done[step] is 1, the episode terminated *after* taking action a_step and receiving r_step from state s_step.
        # The value V(s_{step+1}) is 0 if the state is terminal.
        # delta = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        next_value = values[step + 1]
        mask = 1.0 - dones[step]  # Mask is 0 if episode ended at this step, 1 otherwise
        delta = rewards[step] + gamma * next_value * mask - values[step]
        advantages[step] = last_gae_lam = (
            delta + gamma * gae_lambda * mask * last_gae_lam
        )

    # Value targets are advantages + values V(s_t)
    value_targets = advantages + values[:-1]  # Exclude the last value V(s_N)
    return advantages, value_targets


def ppo_update(
    agent: "PPOAgent",
    batch: Dict[str, torch.Tensor],
    config: TrainConfig,
    device: torch.device,
):
    """Performs PPO optimization steps."""
    # Extract data from batch - ensure they are on the correct device
    maps = batch["maps"].to(device)  # Shape [B, 1, H, W]
    heroes = batch["heroes"].to(device)  # Shape [B, H_dim]
    actions = batch["actions"].to(device)  # Shape [B]
    log_probs_old = batch["log_probs"].to(device)  # Shape [B]
    advantages = batch["advantages"].to(device)  # Shape [B]
    value_targets = batch["value_targets"].to(device)  # Shape [B]

    # Normalize advantages (optional but recommended)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    agent.actor_critic.train()  # Set model to training mode

    dataset = TensorDataset(
        maps, heroes, actions, log_probs_old, advantages, value_targets
    )
    # PPO typically iterates over the same batch multiple times
    loader = DataLoader(
        dataset, batch_size=config.batch_size // config.ppo_epochs, shuffle=True
    )  # Smaller batches for inner epochs

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0

    for _ in range(config.ppo_epochs):
        for map_b, hero_b, act_b, logp_old_b, adv_b, v_targ_b in loader:

            # Get current policy predictions
            logits_new, values_new = agent.actor_critic(map_b, hero_b)
            dist_new = agent.actor_critic.get_action_distribution(
                logits_new, temperature=1.0
            )  # Use temp=1 for loss calcs
            log_probs_new = dist_new.log_prob(act_b)
            entropy = dist_new.entropy().mean()  # Average entropy over batch

            # Policy Ratio
            ratio = torch.exp(log_probs_new - logp_old_b)

            # Clipped Surrogate Objective (Policy Loss)
            surr1 = ratio * adv_b
            surr2 = (
                torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
                * adv_b
            )
            policy_loss = -torch.min(
                surr1, surr2
            ).mean()  # Negative because we want to maximize

            # Value Loss (MSE)
            values_new = values_new.squeeze(-1)  # Ensure correct shape [batch_size]
            value_loss = F.mse_loss(values_new, v_targ_b)

            # Total Loss
            loss = (
                policy_loss
                + config.value_loss_coef * value_loss
                - config.entropy_coef * entropy
            )

            # Optimization step
            agent.optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(agent.actor_critic.parameters(), max_norm=0.5)
            agent.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

    num_updates = len(loader) * config.ppo_epochs
    avg_policy_loss = total_policy_loss / num_updates
    avg_value_loss = total_value_loss / num_updates
    avg_entropy = total_entropy / num_updates

    return avg_policy_loss, avg_value_loss, avg_entropy
