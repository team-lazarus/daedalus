# model.py
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple

# --- Your Provided Encoder ---
class PolicyNetworkEncoder(nn.Module):
    def __init__(
        self,
        channels: List[int] = [1, 4, 16, 64, 256],
        input_size: Tuple[int] = (12, 12),
        output_size: int = 1024,
        hero_tensor_size: int = 5
    ):
        super().__init__()
        self.input_size = self.input_size_x, self.input_size_y = input_size
        # Corrected calculation for CNN output size after downsampling
        downsampled_x = self.input_size_x // 2 // 2 # Two downsampling layers with stride 2
        downsampled_y = self.input_size_y // 2 // 2
        self.cnn_output = downsampled_x * downsampled_y * channels[-1] # Use last channel count
        self.output_size = output_size
        self.hero_tensor_size = hero_tensor_size

        self.conv1 = nn.Conv2d(channels[0], channels[1], 3, padding="same")
        self.conv2 = nn.Conv2d(channels[1], channels[2], 3, padding="same")
        # Downsampling layer 1 (example: stride 2)
        self.down1 = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(channels[2], channels[3], 3, padding="same")
        self.conv4 = nn.Conv2d(channels[3], channels[4], 3, padding="same")
         # Downsampling layer 2 (example: stride 2)
        self.down2 = nn.Conv2d(channels[4], channels[4], kernel_size=3, stride=2, padding=1)

        self.linear = nn.Linear(self.cnn_output + self.hero_tensor_size, self.output_size)

    def forward(self, x: torch.Tensor, hero_tensor: torch.Tensor) -> torch.Tensor:
        """Encodes map and hero tensor."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.down1(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.down2(x))

        x = torch.flatten(x, start_dim=1)
        # Ensure hero_tensor is correctly shaped if needed (e.g., B, H)
        combined = torch.cat([x, hero_tensor], dim=1)
        x = F.relu(self.linear(combined))
        return x

# --- Actor Head (Policy) ---
class PolicyHead(nn.Module):
    def __init__(
        self,
        input_size: int = 1024,
        hidden_sizes: List[int] = [512, 256],
        output_size: int = 7,  # Action space size
    ):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs action logits."""
        # Output logits, softmax applied later for sampling/loss calculation
        return self.network(x)

# --- Critic Head (Value) ---
class ValueHead(nn.Module):
    def __init__(
        self,
        input_size: int = 1024,
        hidden_sizes: List[int] = [512, 256],
    ):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1)) # Output a single value
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs state value prediction."""
        return self.network(x)


# --- Combined Actor-Critic Model ---
class ActorCritic(nn.Module):
    def __init__(
        self,
        config: 'TrainConfig' # Use forward reference for type hint
    ):
        super().__init__()
        action_size = config.get_action_size()

        self.encoder = PolicyNetworkEncoder(
            channels=config.channels,
            input_size=config.map_size,
            output_size=config.encoder_output_size,
            hero_tensor_size=config.hero_tensor_size
        )

        self.actor = PolicyHead(
            input_size=config.encoder_output_size,
            hidden_sizes=config.decoder_hidden_sizes,
            output_size=action_size
        )

        self.critic = ValueHead(
            input_size=config.encoder_output_size,
            hidden_sizes=config.decoder_hidden_sizes,
        )

    def forward(self, map_tensor: torch.Tensor, hero_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns action logits and state value."""
        encoded_state = self.encoder(map_tensor, hero_tensor)
        action_logits = self.actor(encoded_state)
        state_value = self.critic(encoded_state)
        return action_logits, state_value

    def get_action_distribution(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.distributions.Categorical:
        """Creates action distribution with temperature."""
        if temperature <= 1e-6: # Avoid division by zero, treat as deterministic
             # Find the index of the max logit
            action_indices = torch.argmax(logits, dim=-1)
            # Create a one-hot distribution peaked at the max action
            probs = F.one_hot(action_indices, num_classes=logits.shape[-1]).float()
            # Create a categorical distribution from these deterministic probabilities
            # Need to add a small epsilon for numerical stability if using Categorical directly
            # Or handle the deterministic case separately in action selection.
            # For simplicity here, we'll proceed assuming temp > 0 or handle argmax outside.
            # A safer way for pure argmax: handle it in the action selection logic.
            # Let's adjust to return logits directly and apply temp in agent.
            pass # Logits are returned, temp applied in agent

        # Apply temperature scaling to logits before softmax
        scaled_logits = logits / max(temperature, 1e-6) # Ensure temp > 0
        return torch.distributions.Categorical(logits=scaled_logits)