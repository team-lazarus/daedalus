# config.py
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class TrainConfig:
    # Environment settings
    map_size: Tuple[int, int] = (12, 12)
    mode: str = "narrow"  # "narrow", "turtle", "wide"
    random_walk_steps: int = 36
    hero_tensor_size: int = 5  # health, item1, item2, direction, rooms_left

    # Model settings
    channels: List[int] = field(default_factory=lambda: [1, 4, 16, 64, 256])
    encoder_output_size: int = 1024
    decoder_hidden_sizes: List[int] = field(default_factory=lambda: [512, 256])
    # Note: Action size is determined dynamically based on mode

    # PPO settings
    lr: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # Lambda for Generalized Advantage Estimation
    clip_epsilon: float = 0.2  # PPO clipping parameter
    ppo_epochs: int = 10  # Number of optimization epochs per batch
    batch_size: int = 64  # Number of steps collected before update
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    temperature: float = 1.0  # Noise level (0 to 1)

    # Training settings
    num_epochs: int = 1000
    save_interval: int = 5  # Save checkpoint and print map every N epochs
    log_interval: int = 1  # Log metrics every N batches/updates
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    checkpoint_path: Optional[str] = None  # Path to load checkpoint from
    output_dir: str = "ppo_training_output"  # Directory to save results

    def get_action_size(self) -> int:
        """Calculates action space size based on mode."""
        if self.mode == "narrow":
            return 7
        elif self.mode == "turtle":
            return 6 + 4  # 6 map mod + 4 moves
        elif self.mode == "wide":
            n = self.map_size[0]  # Assuming square map
            return 6 * n * n
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def load_config_from_yaml(filepath: str) -> TrainConfig:
    """Loads training configuration from a YAML file."""
    try:
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)
        # You might want more robust validation here
        return TrainConfig(**config_dict)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {filepath}. Using default config.")
        return TrainConfig()
    except Exception as e:
        print(f"Error loading config from {filepath}: {e}. Using default config.")
        return TrainConfig()


def dump_config_to_yaml(config: TrainConfig, filepath: str) -> None:
    """Saves the current configuration to a YAML file."""
    import os

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        # Convert dataclass to dict for dumping
        import dataclasses

        yaml.dump(dataclasses.asdict(config), f, default_flow_style=False)
