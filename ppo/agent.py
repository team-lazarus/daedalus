# agent.py
import torch
from torch import optim
import numpy as np
from typing import Tuple, Dict, Any, Optional

from config import TrainConfig
from model import ActorCritic

class PPOAgent:
    """Agent implementing PPO logic."""

    def __init__(self, config: TrainConfig, device: torch.device):
        """Initializes the agent, model, and optimizer."""
        self.config = config
        self.device = device
        self.actor_critic = ActorCritic(config).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.lr)
        self.temperature = config.temperature

    def select_action(self, map_tensor: torch.Tensor, hero_tensor: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Selects an action based on the current policy and state."""
        self.actor_critic.eval() # Set model to evaluation mode for action selection
        with torch.no_grad():
            action_logits, state_value = self.actor_critic(map_tensor, hero_tensor)

        # Apply temperature and get distribution
        # Ensure logits have batch dim: [1, action_size]
        if action_logits.ndim == 1:
             action_logits = action_logits.unsqueeze(0)

        # Handle zero temperature (deterministic)
        if self.temperature <= 1e-6:
            action = torch.argmax(action_logits, dim=-1).squeeze().item()
             # Log prob calculation needs care for deterministic. Often set to 0 or 1.
             # For simplicity in PPO loss, we might still use the distribution based on original logits
             # But take the argmax action. Let's calculate log_prob based on original logits.
            dist = self.actor_critic.get_action_distribution(action_logits, temperature=1.0) # Use original logits for logprob calc
            log_prob = dist.log_prob(torch.tensor(action, device=self.device))

        else:
             dist = self.actor_critic.get_action_distribution(action_logits, self.temperature)
             action = dist.sample()
             log_prob = dist.log_prob(action)
             action = action.squeeze().item() # Convert tensor to python int

        # Ensure state_value is squeezed correctly, e.g., from [1, 1] to scalar tensor
        state_value = state_value.squeeze()
        if state_value.ndim > 0: # If batch dim was present and > 1 initially
             state_value = state_value.squeeze(-1) # Remove last dim if it's 1

        return action, log_prob, state_value


    def set_temperature(self, temp: float) -> None:
        """Updates the action selection temperature."""
        self.temperature = max(0.0, min(1.0, temp)) # Clamp between 0 and 1

    def save_state(self) -> Dict[str, Any]:
        """Returns state dicts for checkpointing."""
        return {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Loads state dicts from checkpoint."""
        self.actor_critic.load_state_dict(state['actor_critic_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        print("Agent state loaded from checkpoint.")