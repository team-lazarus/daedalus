import csv
import os
import random
import time
from dataclasses import dataclass, field, fields
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from daedalus.critics.level_criticv4 import level_critic as actual_critic

INITIAL_CHECKPOINT_PATH = "neural_critic_checkpoints/critic_MLP_20250422_113333/latest_checkpoint.pth"  # make this "" when running from start


class CriticApproximatorMLP(nn.Module):
    """
    MLP Critic Approximator.

    Predicts a score for a given map layout represented as a flattened tensor.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [256, 128],
        output_size: int = 1,
        dropout_prob: float = 0.3,
    ):
        """
        Initializes the MLP layers.

        Args:
            input_size: The size of the flattened input map tensor.
            hidden_sizes: A list of integers defining the size of each hidden layer.
            output_size: The size of the output layer (typically 1 for a score).
            dropout_prob: The dropout probability to apply after activation functions.
        """
        super().__init__()
        self.input_size = input_size
        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the network.

        Args:
            x: The input tensor of shape (N, 1, H, W).

        Returns:
            The output tensor representing the predicted score(s).

        Raises:
            ValueError: If the input tensor shape or flattened size is incorrect.
        """
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected input shape (N, 1, H, W), got {x.shape}")
        x = torch.flatten(x, start_dim=1)
        if x.shape[1] != self.input_size:
            raise ValueError(
                f"Flattened input size ({x.shape[1]}) does not match "
                f"expected MLP input size ({self.input_size})"
            )
        x = self.network(x)
        return x


@dataclass
class CriticConfig:
    """Configuration class for Neural Critic training using MLP."""

    map_size: Tuple[int, int] = (12, 12)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 50
    steps_per_epoch: int = 1000
    batch_size: int = 128
    learning_rate: float = 1e-4
    mlp_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    mlp_dropout_prob: float = 0.3
    map_gen_min_steps: int = 0
    map_gen_max_steps: int = 256
    tile_empty_prob: float = 0.90
    tile_door_prob: float = 0.05
    tile_empty_value: int = 1
    tile_door_value: int = 6
    tile_enemy_values: List[int] = field(default_factory=lambda: [3, 4, 5])
    validation_freq: int = 500
    validation_batches: int = 50
    save_checkpoint_freq_epochs: int = 5
    checkpoint_dir: str = "neural_critic_checkpoints"
    run_name: Optional[str] = None
    log_filename: str = "training_metrics.csv"
    hero_tensor_size: int = 5
    print_validation_maps: bool = True
    num_validation_maps_to_print: int = 5

    tile_enemy_prob: float = field(init=False)

    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.tile_enemy_prob = max(
            0.0, 1.0 - self.tile_empty_prob - self.tile_door_prob
        )
        if not self.tile_enemy_values and self.tile_enemy_prob > 0:
            print(
                "Warning: Non-zero enemy probability but no enemy "
                "tile values specified."
            )
            self.tile_enemy_prob = 0

        if self.run_name is None:
            self.run_name = f"critic_MLP_{time.strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.run_name)


def configure_critic_from_yaml(yaml_path: str) -> CriticConfig:
    """
    Loads MLP critic configuration from a YAML file, filtering unknown keys.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        A CriticConfig instance.
    """
    try:
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        valid_keys = {f.name for f in fields(CriticConfig) if f.init}
        filtered_config = {k: v for k, v in yaml_config.items() if k in valid_keys}

        if "print_validation_maps" in filtered_config:
            filtered_config["print_validation_maps"] = bool(
                filtered_config["print_validation_maps"]
            )
        return CriticConfig(**filtered_config)
    except FileNotFoundError:
        print(
            f"Warning: YAML config file not found at {yaml_path}. "
            "Using default config."
        )
        return CriticConfig()
    except Exception as e:
        print(f"Error loading YAML config: {e}. Using default config.")
        return CriticConfig()


def generate_random_map(config: CriticConfig) -> torch.Tensor:
    """
    Generates a random map tensor based on the configuration settings.

    Args:
        config: The CriticConfig instance containing map generation parameters.

    Returns:
        A map tensor of shape specified in config.map_size.
    """
    map_tensor = torch.zeros(config.map_size, dtype=torch.int64)
    size_x, size_y = config.map_size
    if size_x <= 0 or size_y <= 0:
        return map_tensor

    start_pos = (random.randint(0, size_x - 1), random.randint(0, size_y - 1))
    current_pos = start_pos
    num_steps = random.randint(config.map_gen_min_steps, config.map_gen_max_steps)

    for _ in range(num_steps):
        rand_val = random.random()
        if rand_val < config.tile_empty_prob:
            tile_value = config.tile_empty_value
        elif rand_val < config.tile_empty_prob + config.tile_door_prob:
            tile_value = config.tile_door_value
        elif config.tile_enemy_prob > 0 and config.tile_enemy_values:
            tile_value = random.choice(config.tile_enemy_values)
        else:
            tile_value = config.tile_empty_value

        map_tensor[current_pos] = tile_value

        dx, dy = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        next_x = max(0, min(size_x - 1, current_pos[0] + dx))
        next_y = max(0, min(size_y - 1, current_pos[1] + dy))
        current_pos = (next_x, next_y)

    return map_tensor


def generate_batch(
    batch_size: int,
    config: CriticConfig,
    critic_func: Callable,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of maps and their 'true' scores from the original critic.

    Args:
        batch_size: The number of maps to generate in the batch.
        config: The CriticConfig instance.
        critic_func: The original critic function to score the maps.
        device: The torch device to place the output tensors on.

    Returns:
        A tuple containing:
        - maps_batch_nn: Tensor of maps ready for the neural network (N, 1, H, W).
        - true_scores: Tensor of corresponding scores from critic_func (N, 1).
    """
    maps_list = [generate_random_map(config) for _ in range(batch_size)]
    maps_batch_cpu = torch.stack(maps_list)

    dummy_hero_batch = torch.zeros(
        (batch_size, config.hero_tensor_size),
        dtype=torch.int64,
        device=device,
    )
    true_scores = critic_func(maps_batch_cpu.to(device), dummy_hero_batch)

    maps_batch_nn = maps_batch_cpu.unsqueeze(1).float().to(device)
    true_scores = true_scores.float().to(device).unsqueeze(-1)
    return maps_batch_nn, true_scores


class NeuralCriticTrainer:
    """Orchestrates the training of the MLP neural critic approximator."""

    def __init__(self, config: CriticConfig, critic_func: Callable):
        """
        Initializes the trainer, model, optimizer, and logging.

        Args:
            config: The CriticConfig instance.
            critic_func: The original critic function used for generating target scores.
        """
        self.config = config
        self.critic_func = critic_func
        self.device = torch.device(config.device)
        self.console = Console()

        self._set_seeds(config.seed)

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.log_filepath = os.path.join(config.checkpoint_dir, config.log_filename)
        self.log_fieldnames = [
            "step",
            "epoch",
            "timestamp",
            "train_mse",
            "validation_mse",
        ]
        self._init_log_file()

        self.console.print("Initializing [bold blue]MLP[/] critic model...")
        map_h, map_w = config.map_size
        self.model = CriticApproximatorMLP(
            input_size=map_h * map_w,
            hidden_sizes=config.mlp_hidden_sizes,
            output_size=1,
            dropout_prob=config.mlp_dropout_prob,
        ).to(self.device)
        self.console.print(
            f"  MLP Hidden Sizes: {config.mlp_hidden_sizes}, "
            f"Dropout: {config.mlp_dropout_prob}"
        )

        self._load_initial_checkpoint(INITIAL_CHECKPOINT_PATH)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        self.start_epoch = 0
        self.total_steps_trained = 0

    def _set_seeds(self, seed: int):
        """Sets random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)

    def _init_log_file(self):
        """Creates the log file and writes the header if it doesn't exist."""
        if not os.path.exists(self.log_filepath):
            try:
                with open(self.log_filepath, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.log_fieldnames)
            except IOError as e:
                self.console.print(
                    f"[bold red]Error:[/bold red] Could not create log file "
                    f"{self.log_filepath}: {e}"
                )

    def _load_initial_checkpoint(self, path: str):
        """Loads initial model weights from a specified checkpoint path."""
        if os.path.exists(path):
            try:
                # Load checkpoint allowing arbitrary objects (like CriticConfig)
                # Only do this if the checkpoint source is trusted.
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=False
                )
                if "model_state_dict" in checkpoint:
                    # Ensure model state matches before loading
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    self.console.print(
                        f"Loaded initial model weights from [green]{path}[/green]."
                    )
                else:
                    self.console.print(
                        f"[yellow]Warning:[/yellow] Checkpoint at {path} exists but lacks 'model_state_dict'. Using randomly initialized weights."
                    )
            except Exception as e:
                self.console.print(
                    f"[red]Error loading initial checkpoint from {path}: {e}. "
                    "Starting with randomly initialized weights.[/red]"
                )
        else:
            self.console.print(
                f"Initial checkpoint [yellow]{path}[/yellow] not found. "
                "Starting with randomly initialized weights."
            )

    def _log_metrics(
        self, step: int, epoch: int, train_mse: float, validation_mse: float
    ):
        """Appends a row of metrics to the CSV log file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_data = [
            step,
            epoch + 1,
            timestamp,
            f"{train_mse:.6f}",
            f"{validation_mse:.6f}",
        ]
        try:
            with open(self.log_filepath, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(log_data)
        except IOError as e:
            self.console.print(
                f"[bold red]Warning:[/bold red] Could not write to log file "
                f"{self.log_filepath}: {e}"
            )

    def _save_checkpoint(self, epoch: int, is_latest: bool = False):
        """Saves model and optimizer state to the run's checkpoint directory."""
        filename = (
            "latest_checkpoint.pth"
            if is_latest
            else f"checkpoint_epoch_{epoch + 1}.pth"
        )
        path = os.path.join(self.config.checkpoint_dir, filename)
        temp_path = path + ".tmp"
        checkpoint = {
            "epoch": epoch,
            "total_steps_trained": self.total_steps_trained,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, path)
        except Exception as e:
            self.console.print(f"[red]Error saving checkpoint to {path}: {e}[/red]")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _load_checkpoint(self):
        """Loads the latest checkpoint from the run's directory to resume training."""
        latest_checkpoint_path = os.path.join(
            self.config.checkpoint_dir, "latest_checkpoint.pth"
        )
        if os.path.exists(latest_checkpoint_path):
            try:
                # Load checkpoint allowing arbitrary objects for resuming
                checkpoint = torch.load(
                    latest_checkpoint_path, map_location=self.device, weights_only=False
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.start_epoch = checkpoint.get("epoch", -1) + 1
                self.total_steps_trained = checkpoint.get("total_steps_trained", 0)
                self.console.print(
                    f"Resuming training. Checkpoint loaded from [green]{latest_checkpoint_path}[/green]. Resuming from epoch {self.start_epoch}."
                )
            except Exception as e:
                self.console.print(
                    f"[red]Error loading checkpoint: {e}. Starting from scratch for this run.[/red]"
                )
                self.start_epoch = 0
                self.total_steps_trained = 0
        else:
            self.console.print(
                "No run-specific checkpoint found. Starting training from scratch."
            )
            self.start_epoch = 0
            self.total_steps_trained = 0

    def _print_validation_example(
        self, map_tensor: torch.Tensor, true_score: float, pred_score: float, index: int
    ):
        """Prints a single validation map example with scores using Rich."""
        if map_tensor.ndim != 2:
            self.console.print(
                f"[red]Error: Invalid map dimensions for printing ({map_tensor.shape})[/red]"
            )
            return

        map_np = map_tensor.cpu().numpy().astype(int)
        map_size_x, map_size_y = map_np.shape

        colors = {
            0: "dim grey50",
            1: "white",
            6: "bright_green",
            2: "bright_red",
            3: "red",
            4: "dark_red",
            5: "red3",
        }
        default_color = "magenta"
        char_width = 2

        table = Table(
            title=f"Validation Map Example #{index + 1}",
            show_header=False,
            show_edge=True,
            box=None,
            padding=0,
            expand=False,
        )
        for _ in range(map_size_y):
            table.add_column(justify="center", width=char_width)

        for r in range(map_size_x):
            row_cells = [
                f"[{colors.get(tile, default_color)}]{tile:>{char_width-1}} [/]"
                for tile in map_np[r]
            ]
            table.add_row(*row_cells)

        self.console.print(table)
        score_text = Text.assemble(
            "  Scores: ",
            ("True = ", "white"),
            (f"{true_score:.4f}", "bright_green"),
            (" | Pred = ", "white"),
            (f"{pred_score:.4f}", "bright_magenta"),
            (" | Diff = ", "white"),
            (
                f"{pred_score - true_score:+.4f}",
                "bright_red" if abs(pred_score - true_score) > 0.1 else "dim",
            ),
        )
        self.console.print(score_text)
        self.console.print("-" * (map_size_y * (char_width + 1) + 1))

    def _validate(self) -> float:
        """
        Performs validation on a set of batches and returns the average MSE loss.
        Optionally prints example maps.
        """
        self.model.eval()
        total_val_loss = 0.0
        num_batches = self.config.validation_batches
        if num_batches <= 0:
            self.model.train()
            return 0.0

        last_batch_maps_cpu = None
        last_batch_true_scores = None
        last_batch_pred_scores = None

        with torch.no_grad():
            for i in range(num_batches):
                val_maps_nn, val_true_scores = generate_batch(
                    self.config.batch_size,
                    self.config,
                    self.critic_func,
                    self.device,
                )
                val_pred_scores = self.model(val_maps_nn)
                val_loss = self.criterion(val_pred_scores, val_true_scores)
                total_val_loss += val_loss.item()

                if i == num_batches - 1:
                    last_batch_maps_cpu = val_maps_nn.squeeze(1).cpu()
                    last_batch_true_scores = val_true_scores.squeeze(-1).cpu()
                    last_batch_pred_scores = val_pred_scores.squeeze(-1).cpu()

        avg_val_loss = total_val_loss / num_batches
        self.model.train()

        if (
            self.config.print_validation_maps
            and last_batch_maps_cpu is not None
            and last_batch_true_scores is not None
            and last_batch_pred_scores is not None
        ):
            num_examples = min(
                self.config.num_validation_maps_to_print,
                last_batch_maps_cpu.shape[0],
            )
            if num_examples > 0:
                self.console.print(
                    Panel(
                        f"Validation Examples (Random {num_examples} from last batch)",
                        style="blue",
                        expand=False,
                    )
                )
                indices = random.sample(
                    range(last_batch_maps_cpu.shape[0]), num_examples
                )
                for i, idx in enumerate(indices):
                    map_to_print = last_batch_maps_cpu[idx]
                    true_score = last_batch_true_scores[idx].item()
                    pred_score = last_batch_pred_scores[idx].item()
                    self._print_validation_example(
                        map_tensor=map_to_print,
                        true_score=true_score,
                        pred_score=pred_score,
                        index=i,
                    )

        return avg_val_loss

    def train(self):
        """Runs the main training loop for the MLP critic."""
        cfg = self.config
        self.console.print(
            Panel.fit(
                f"Starting Neural Critic Training: run='{cfg.run_name}', model='MLP'",
                title="Setup",
                border_style="blue",
            )
        )
        self.console.print(
            f"Device: [cyan]{self.device}[/cyan], Epochs: {cfg.num_epochs}, Steps/Epoch: {cfg.steps_per_epoch}, Batch Size: {cfg.batch_size}"
        )
        self.console.print(f"Log file: [dim]{self.log_filepath}[/dim]")
        self.console.print(
            f"Validation Freq: {cfg.validation_freq} steps, Print Maps: {cfg.print_validation_maps}, Num Maps: {cfg.num_validation_maps_to_print}"
        )
        self.console.print(f"Checkpoint Freq: {cfg.save_checkpoint_freq_epochs} epochs")
        self.console.print(
            f"Map Gen Steps: {cfg.map_gen_min_steps}-{cfg.map_gen_max_steps}, Empty: {cfg.tile_empty_prob:.2f}, Door: {cfg.tile_door_prob:.2f}, Enemy: {cfg.tile_enemy_prob:.2f}"
        )
        self.console.print(f"Checkpoints Dir: [green]{cfg.checkpoint_dir}[/green]")
        self.console.print(
            f"Initial Checkpoint Path: [cyan]{INITIAL_CHECKPOINT_PATH}[/cyan]"
        )

        self._load_checkpoint()
        if self.start_epoch >= cfg.num_epochs:
            self.console.print(
                "[yellow]Checkpoint indicates training already completed. Exiting.[/yellow]"
            )
            return

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            TextColumn(
                "[bold]Metrics:[/]{task.fields[metrics]}", justify="left", style="white"
            ),
            console=self.console,
            transient=False,
        )
        total_training_steps = cfg.num_epochs * cfg.steps_per_epoch
        initial_metrics = " Train MSE: ---- | Val MSE: ----"

        self.model.train()
        with progress:
            task = progress.add_task(
                "[cyan]Training Critic (MLP)",
                total=total_training_steps,
                completed=self.total_steps_trained,
                metrics=initial_metrics,
            )
            last_logged_train_mse = float("nan")
            last_logged_val_mse = float("nan")

            while self.total_steps_trained < total_training_steps:
                current_epoch = self.total_steps_trained // cfg.steps_per_epoch

                maps_nn, true_scores = generate_batch(
                    cfg.batch_size, cfg, self.critic_func, self.device
                )
                predicted_scores = self.model(maps_nn)
                loss = self.criterion(predicted_scores, true_scores)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.total_steps_trained += 1
                train_mse = loss.item()
                last_logged_train_mse = train_mse

                val_mse_str = (
                    f"{last_logged_val_mse:.4f}"
                    if not np.isnan(last_logged_val_mse)
                    else "----    "
                )
                metrics_str = (
                    f" Train MSE:[red]{train_mse:>8.4f}[/]| Val MSE:{val_mse_str:>8}"
                )

                if self.total_steps_trained % cfg.validation_freq == 0:
                    val_mse = self._validate()
                    last_logged_val_mse = val_mse
                    val_mse_str = f"{val_mse:.4f}"
                    metrics_str = f" Train MSE:[red]{train_mse:>8.4f}[/]| Val MSE:{val_mse_str:>8}"
                    self._log_metrics(
                        step=self.total_steps_trained,
                        epoch=current_epoch,
                        train_mse=last_logged_train_mse,
                        validation_mse=val_mse,
                    )

                progress.update(task, advance=1, metrics=metrics_str)

                is_epoch_end = self.total_steps_trained % cfg.steps_per_epoch == 0
                if (
                    is_epoch_end and self.total_steps_trained > 0
                ):  # Avoid saving at step 0 if epoch size matches steps
                    completed_epoch_index = current_epoch
                    if (
                        completed_epoch_index + 1
                    ) % cfg.save_checkpoint_freq_epochs == 0:
                        self._save_checkpoint(completed_epoch_index)
                    self._save_checkpoint(completed_epoch_index, is_latest=True)

        self.console.print(
            Panel(
                f"Training finished after {cfg.num_epochs} epochs ({self.total_steps_trained:,} total steps). Model: MLP",
                title="Complete",
                border_style="green",
            )
        )
        final_epoch_index = cfg.num_epochs - 1
        if (final_epoch_index + 1) % cfg.save_checkpoint_freq_epochs != 0:
            self._save_checkpoint(final_epoch_index)
        self._save_checkpoint(final_epoch_index, is_latest=True)


if __name__ == "__main__":
    CONFIG_PATH = "neural_critic_config.yaml"
    config: CriticConfig
    if os.path.exists(CONFIG_PATH):
        print(f"Loading configuration from {CONFIG_PATH}")
        config = configure_critic_from_yaml(CONFIG_PATH)
    else:
        print(
            f"Configuration file '{CONFIG_PATH}' not found. Using default CriticConfig."
        )
        config = CriticConfig()

    trainer = NeuralCriticTrainer(config, actual_critic)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print(
            "\n[yellow]Training interrupted by user. Saving final checkpoint...[/yellow]"
        )
        last_completed_epoch = (
            trainer.total_steps_trained // config.steps_per_epoch
        ) - 1
        if last_completed_epoch < trainer.start_epoch:
            last_completed_epoch = trainer.start_epoch - 1
        if last_completed_epoch >= 0:
            trainer._save_checkpoint(last_completed_epoch, is_latest=True)
            print(
                f"Interrupted state checkpoint saved (epoch {last_completed_epoch + 1} started)."
            )
        else:
            print("No checkpoint saved as training interrupted very early.")
    except Exception as e:
        trainer.console.print_exception(show_locals=False)
        print(
            "\n[red]An error occurred during training. Attempting to save final checkpoint...[/red]"
        )
        last_completed_epoch = (
            trainer.total_steps_trained // config.steps_per_epoch
        ) - 1
        if last_completed_epoch < trainer.start_epoch:
            last_completed_epoch = trainer.start_epoch - 1
        if last_completed_epoch >= 0:
            trainer._save_checkpoint(last_completed_epoch, is_latest=True)
            print(
                f"Error state checkpoint saved (epoch {last_completed_epoch + 1} started)."
            )
        else:
            print("No checkpoint saved due to early error.")

    console = Console()
    print("\n--- Optional: Testing Trained MLP Critic Approximator ---")
    try:
        test_config = trainer.config
        test_device = trainer.device
        console.print("Attempting to test the trained [bold blue]MLP[/] model...")

        test_model = CriticApproximatorMLP(
            input_size=test_config.map_size[0] * test_config.map_size[1],
            hidden_sizes=test_config.mlp_hidden_sizes,
            dropout_prob=test_config.mlp_dropout_prob,
        ).to(test_device)

        latest_checkpoint_path = os.path.join(
            test_config.checkpoint_dir, "latest_checkpoint.pth"
        )

        if os.path.exists(latest_checkpoint_path):
            # Use weights_only=False here too if the saved checkpoint includes the config
            checkpoint = torch.load(
                latest_checkpoint_path, map_location=test_device, weights_only=False
            )
            test_model.load_state_dict(checkpoint["model_state_dict"])
            test_model.eval()
            console.print(
                f"Loaded model state from [green]{latest_checkpoint_path}[/green] for testing."
            )

            test_maps_nn, test_true_scores = generate_batch(
                5, test_config, actual_critic, test_device
            )
            with torch.no_grad():
                test_pred_scores = test_model(test_maps_nn)

            console.print("\nSample Predictions vs True Scores:")
            table = Table(title="MLP Critic Approximation Test")
            table.add_column("Map Index", style="cyan")
            table.add_column("Predicted Score", style="magenta", justify="right")
            table.add_column("True Score (from critic)", style="green", justify="right")
            table.add_column("Difference", style="red", justify="right")

            for i in range(test_pred_scores.shape[0]):
                pred = test_pred_scores[i].item()
                true = test_true_scores[i].item()
                diff = pred - true
                table.add_row(str(i), f"{pred:.4f}", f"{true:.4f}", f"{diff:+.4f}")
            console.print(table)
        else:
            console.print(
                f"[yellow]Could not find latest checkpoint at {latest_checkpoint_path} for testing.[/yellow]"
            )
    except FileNotFoundError:
        console.print(
            "[yellow]Trained model checkpoint not available for testing.[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Testing failed: {e}[/red]")
        console.print_exception(show_locals=False)
