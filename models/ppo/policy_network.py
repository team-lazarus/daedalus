import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class PolicyNetworkEncoder(nn.Module):
    def __init__(
        self,
        channels: List[int] = [1, 4, 16, 64, 256],
        input_size: Tuple[int] = (12, 12),
        output_size: int = 1024,
    ):
        super().__init__()
        self.input_size = self.input_size_x, self.input_size_y = input_size
        self.cnn_output = ((self.input_size_x * self.input_size_y) // 36)*channels[4]
        self.output_size = output_size

        self.conv1 = nn.Conv2d(channels[0], channels[1], 3, padding="same")
        self.conv2 = nn.Conv2d(channels[1], channels[2], 3, padding="same")
        self.conv3 = nn.Conv2d(channels[2], channels[3], 3, padding="same")
        self.conv4 = nn.Conv2d(channels[3], channels[4], 3, padding="same")

        self.down1 = nn.Conv2d(channels[2], channels[2], 3, 2)
        self.down2 = nn.Conv2d(channels[4], channels[4], 3, 2)

        self.linear = nn.Linear(self.cnn_output, self.output_size, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.down1(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.down2(x))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear(x))
        return x


class PolicyNetworkDecoder(nn.Module):
    def __init__(
        self,
        input_size: int = 1024,
        hidden_sizes: List[int] = [512, 256],
        output_size: int = 7,  # 12x12 flattened output
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Create sequential layers with decreasing sizes
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Final layer uses softmax for action probabilities
        x = F.softmax(self.fc3(x), dim=1)
        return x


class PPOPolicyNetwork(nn.Module):
    def __init__(
        self,
        channels: List[int] = [1, 4, 16, 64, 256],
        input_size: Tuple[int] = (12, 12),
        encoder_output_size: int = 1024,
        decoder_hidden_sizes: List[int] = [512, 256],
        output_size: int = 7,  # 12x12 flattened output
    ):
        super().__init__()
        self.encoder = PolicyNetworkEncoder(
            channels=channels,
            input_size=input_size,
            output_size=encoder_output_size
        )
        
        self.decoder = PolicyNetworkDecoder(
            input_size=encoder_output_size,
            hidden_sizes=decoder_hidden_sizes,
            output_size=output_size
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    console = Console()
    
    # Create a table for model architecture
    table = Table(title="Policy Network Architecture Test")
    table.add_column("Component", style="cyan")
    table.add_column("Input Shape", style="green")
    table.add_column("Output Shape", style="magenta")
    
    # Test encoder
    encoder = PolicyNetworkEncoder()
    in_tensor = torch.rand((16, 1, 12, 12))
    out_tensor = encoder(in_tensor)
    table.add_row("Encoder", str(in_tensor.shape), str(out_tensor.shape))
    
    # Test decoder
    decoder = PolicyNetworkDecoder()
    decoder_out = decoder(out_tensor)
    table.add_row("Decoder", str(out_tensor.shape), str(decoder_out.shape))
    
    # Test full policy network
    policy_net = PPOPolicyNetwork()
    policy_out = policy_net(in_tensor)
    table.add_row("Full Policy Network", str(in_tensor.shape), str(policy_out.shape))
    
    # Print the table
    console.print(Panel.fit(table, title="Neural Network Test Results", border_style="blue"))
    
    # Print model parameter statistics
    total_params = sum(p.numel() for p in policy_net.parameters())
    trainable_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
    
    console.print(Panel(
        f"[bold]Model Statistics[/bold]\n"
        f"Total parameters: [yellow]{total_params:,}[/yellow]\n"
        f"Trainable parameters: [green]{trainable_params:,}[/green]\n"
        f"Memory footprint (approx): [cyan]{total_params * 4 / (1024 * 1024):.2f} MB[/cyan]",
        title="Model Info",
        border_style="green"
    ))