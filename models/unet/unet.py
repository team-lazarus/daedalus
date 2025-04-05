import torch
import torch.nn as nn
import lightning as L
from typing import List, Tuple, Optional, Union, Callable
from rich.console import Console
from rich.logging import RichHandler
import logging
import torch.nn.functional as F

# Set up rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("unet")

from .unet_parts import DoubleConv, Down, Up, OutConv


class LightningUNet(L.LightningModule):
    """Lightning implementation of U-Net with optional conditioning and PCGRL critic.

    Attributes:
        n_channels: Number of input channels.
        n_classes: Number of output classes.
        bilinear: Whether to use bilinear upsampling.
        enc_dims: Channel dimensions for encoder path.
        dec_dims: Channel dimensions for decoder path.
        hero_size: Size of conditioning tensor (0 to disable).
        lr: Learning rate for the optimizer.
        critic: PCGRL critic for reward-based learning.
        entropy_coef: Coefficient for entropy regularization.
        baseline_decay: Decay rate for reward baseline (moving average).
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        *,
        critic: Callable,
        bilinear: bool = False,
        enc_dims: List[int] = [8, 16, 32, 64, 128],
        dec_dims: Optional[List[int]] = None,
        hero_size: int = 0,
        lr: float = 1e-3,
        entropy_coef: float = 0.01,  # For entropy regularization
        baseline_decay: float = 0.9,  # For reward baseline
    ) -> None:
        """Initialize the LightningUNet model with PCGRL critic integration.

        Args:
            n_channels: Number of input channels.
            n_classes: Number of output classes.
            bilinear: Use bilinear upsampling instead of transposed convolutions.
            enc_dims: Channel dimensions for encoder (min 5 elements).
            dec_dims: Channel dimensions for decoder (min 4 elements).
                      If None, calculated from enc_dims.
            hero_size: Conditioning tensor size (0 to disable).
            lr: Learning rate for the optimizer.
            critic: PCGRL critic that returns reward values for generated levels.
            entropy_coef: Coefficient for entropy regularization (higher = more exploration).
            baseline_decay: Decay rate for the reward baseline (higher = slower adaptation).

        Raises:
            ValueError: If dimensions are invalid or hero_size is negative.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["critic"])
        self.critic = critic

        # Reward baseline for variance reduction
        self.register_buffer("reward_baseline", torch.tensor(0.0))

        # Average reward tracking for logging
        self.avg_reward = 0.0
        self.reward_count = 0

        # Validate inputs
        try:
            if hero_size < 0:
                raise ValueError("hero_size cannot be negative")
            if len(enc_dims) < 5:
                raise ValueError("enc_dims must have at least 5 elements")
            if critic is None:
                raise ValueError("critic must be provided for PCGRL training")
        except ValueError as e:
            console.print(f"[bold red]Configuration Error:[/bold red] {str(e)}")
            raise

        # Set up decoder dimensions if not provided
        factor = 2 if bilinear else 1
        if dec_dims is None:
            dec_dims = [enc_dims[3], enc_dims[2], enc_dims[1], enc_dims[0]]
            self.hparams.dec_dims = dec_dims
            log.info(f"Auto-generated decoder dimensions: {dec_dims}")
        elif len(dec_dims) < 4:
            error_msg = "dec_dims must have at least 4 elements"
            console.print(f"[bold red]Configuration Error:[/bold red] {error_msg}")
            raise ValueError(error_msg)

        # Build encoder (downsampling path)
        log.info("Building U-Net architecture...")
        try:
            self.inc = DoubleConv(n_channels, enc_dims[0])
            self.down1 = Down(enc_dims[0], enc_dims[1])
            self.down2 = Down(enc_dims[1], enc_dims[2])
            self.down3 = Down(enc_dims[2], enc_dims[3])
            self.down4 = Down(enc_dims[3], enc_dims[4] // factor)

            # Build decoder (upsampling path) with hero tensor accommodation
            bottleneck_channels = enc_dims[4] // factor + hero_size
            self.up1 = Up(bottleneck_channels, dec_dims[0] // factor, bilinear)
            self.up2 = Up(dec_dims[0], dec_dims[1] // factor, bilinear)
            self.up3 = Up(dec_dims[1], dec_dims[2] // factor, bilinear)
            self.up4 = Up(dec_dims[2], dec_dims[3], bilinear)

            # Output layer
            self.outc = OutConv(dec_dims[3], n_classes)

            # Log architecture summary
            log.info("✅ U-Net architecture built successfully")
            log.info(f"  • Input channels: {n_channels}")
            log.info(f"  • Output classes: {n_classes}")
            log.info(
                f"  • Using {'bilinear' if bilinear else 'transposed conv'} upsampling"
            )
            if hero_size > 0:
                log.info(f"  • Hero conditioning enabled ({hero_size} features)")
            log.info(
                f"  • PCGRL critic enabled (entropy={entropy_coef:.3f}, baseline decay={baseline_decay:.2f})"
            )
        except Exception as e:
            console.print_exception()
            raise RuntimeError(f"Failed to build U-Net architecture: {str(e)}")

    def forward(
        self, x: torch.Tensor, hero: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional conditioning.

        Args:
            x: Input tensor [B, n_channels, H, W].
            hero: Optional conditioning tensor [B, hero_size].
                 Required if hero_size > 0.

        Returns:
            Output tensor [B, n_classes, H, W].

        Raises:
            ValueError: If hero tensor dimensions don't match expected.
        """
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Bottleneck

        # Apply hero tensor conditioning if enabled
        if self.hparams.hero_size > 0:
            B, _, H, W = x5.shape

            if hero is None:
                # Use zeros if no hero tensor provided (with warning)
                log.warning("❗ Hero tensor expected but not provided, using zeros")
                hero_features = torch.zeros(
                    B, self.hparams.hero_size, H, W, device=x5.device, dtype=x5.dtype
                )
            else:
                try:
                    # Validate hero tensor dimensions
                    if hero.shape[0] != B or hero.shape[1] != self.hparams.hero_size:
                        error_msg = (
                            f"Expected hero shape [B={B}, F={self.hparams.hero_size}], "
                            f"got {hero.shape}"
                        )
                        console.print(
                            f"[bold red]Hero Tensor Error:[/bold red] {error_msg}"
                        )
                        raise ValueError(error_msg)

                    # Reshape and expand hero tensor to match spatial dims
                    hero_features = hero.view(B, self.hparams.hero_size, 1, 1).expand(
                        B, self.hparams.hero_size, H, W
                    )
                    log.debug(f"Applied hero tensor conditioning (shape: {hero.shape})")
                except Exception as e:
                    console.print_exception()
                    raise

            # Concatenate along channel dimension
            x5 = torch.cat([x5, hero_features], dim=1)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

    def _get_level_maps(self, logits):
        """Convert model logits to discrete level maps for the critic.

        Args:
            logits: Model output logits [B, n_classes, H, W]

        Returns:
            List of level maps (each map is a 2D list of integers)
        """
        # Convert to class indices
        predictions = torch.argmax(logits, dim=1)  # [B, H, W]

        # Convert to CPU numpy arrays then to Python lists
        batch_maps = []
        for pred in predictions:
            level_map = pred.detach().cpu().numpy().tolist()
            batch_maps.append(level_map)

        return batch_maps

    def _sample_level_maps(self, logits):
        """Sample level maps from probability distributions for exploration.

        Args:
            logits: Model output logits [B, n_classes, H, W]

        Returns:
            Tuple of (sampled_maps, log_probs):
                - sampled_maps: List of level maps
                - log_probs: Log probabilities of the sampled maps [B]
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # [B, n_classes, H, W]

        # Sample from the categorical distribution
        batch_size, n_classes, height, width = logits.shape
        categorical = torch.distributions.Categorical(probs=probs.permute(0, 2, 3, 1))
        samples = categorical.sample()  # [B, H, W]
        log_probs = categorical.log_prob(samples)  # [B, H, W]

        # Sum log probs over spatial dimensions for each map
        map_log_probs = log_probs.sum(dim=[1, 2])  # [B]

        # Convert samples to level maps
        batch_maps = []
        for sample in samples:
            level_map = sample.detach().cpu().numpy().tolist()
            batch_maps.append(level_map)

        return batch_maps, map_log_probs

    def _compute_entropy(self, logits):
        """Compute entropy of the output distribution for regularization.

        Args:
            logits: Model output logits [B, n_classes, H, W]

        Returns:
            Average entropy per batch element
        """
        probs = F.softmax(logits, dim=1)  # [B, n_classes, H, W]
        log_probs = F.log_softmax(logits, dim=1)  # [B, n_classes, H, W]

        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=1)  # [B, H, W]

        # Average over spatial dimensions and batch
        return entropy.mean()

    def _process_batch(
        self, batch: Union[Tuple, List]
    ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """Process batch data and compute policy gradient loss with the critic.

        Handles different batch formats (with or without hero tensor).

        Args:
            batch: Either (x,) or (x, hero)

        Returns:
            Tuple of (policy_loss, entropy, rewards)
        """
        try:
            # Unpack batch based on format
            if len(batch) == 2:
                x, hero = batch
                log.debug(
                    f"Processing batch with hero tensor (shape: {hero.shape if hero is not None else None})"
                )
            elif len(batch) == 1:
                x = batch[0]
                hero = None
                log.debug("Processing batch without hero tensor")
            else:
                error_msg = f"Expected batch with 1 or 2 elements, got {len(batch)}"
                console.print(f"[bold red]Batch Error:[/bold red] {error_msg}")
                raise ValueError(error_msg)

            # Skip hero tensor if not configured
            if self.hparams.hero_size == 0 and hero is not None:
                log.debug("Hero tensor provided but not configured - ignoring")
                hero = None

            # Forward pass to get logits
            logits = self(x, hero)

            # Sample level maps and get their log probabilities
            level_maps, log_probs = self._sample_level_maps(logits)

            # Get rewards from critic for each sampled map
            rewards = []
            for level_map in level_maps:
                try:
                    reward = self.critic(level_map)
                    rewards.append(reward)
                except Exception as e:
                    log.error(f"❌ Critic evaluation failed: {str(e)}")
                    # Use a default negative reward if critic fails
                    rewards.append(-1.0)

            # Convert rewards to tensor
            rewards_tensor = torch.tensor(rewards, device=log_probs.device)

            # Update reward statistics
            batch_size = len(rewards)
            self.avg_reward = (self.avg_reward * self.reward_count + sum(rewards)) / (
                self.reward_count + batch_size
            )
            self.reward_count += batch_size

            # Update reward baseline with moving average
            with torch.no_grad():
                self.reward_baseline = (
                    self.hparams.baseline_decay * self.reward_baseline
                    + (1 - self.hparams.baseline_decay) * rewards_tensor.mean()
                )

            # Compute advantages (rewards - baseline)
            advantages = rewards_tensor - self.reward_baseline

            # Policy gradient loss: -log_prob * advantage
            # Negative because we want to maximize reward
            policy_loss = -log_probs * advantages
            policy_loss = policy_loss.mean()

            # Compute entropy for regularization
            entropy = self._compute_entropy(logits)

            # Combined loss with entropy regularization
            # We subtract entropy to encourage exploration
            loss = policy_loss - self.hparams.entropy_coef * entropy

            return loss, entropy, rewards

        except Exception as e:
            console.print_exception()
            log.error(f"❌ Batch processing failed: {str(e)}")
            raise

    def training_step(self, batch, batch_idx):
        """Execute training step with PCGRL critic."""
        loss, entropy, rewards = self._process_batch(batch)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_entropy", entropy, prog_bar=True)
        self.log("train_reward_mean", sum(rewards) / len(rewards), prog_bar=True)
        self.log("train_reward_min", min(rewards))
        self.log("train_reward_max", max(rewards))
        self.log("train_reward_baseline", self.reward_baseline)

        # Periodically log detailed training information
        if batch_idx % 50 == 0:
            log.info(
                f"Training batch {batch_idx}: loss={loss:.4f}, entropy={entropy:.4f}, "
                + f"reward_mean={sum(rewards) / len(rewards):.4f}, "
                + f"reward_range=[{min(rewards):.4f}, {max(rewards):.4f}]"
            )

        return loss

    def validation_step(self, batch, batch_idx):
        """Execute validation step with PCGRL critic."""
        loss, entropy, rewards = self._process_batch(batch)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_entropy", entropy)
        self.log("val_reward_mean", sum(rewards) / len(rewards), prog_bar=True)
        self.log("val_reward_min", min(rewards))
        self.log("val_reward_max", max(rewards))

        # Log first few validation batches for monitoring
        if batch_idx < 3:
            log.info(
                f"Validation batch {batch_idx}: loss={loss:.4f}, entropy={entropy:.4f}, "
                + f"reward_mean={sum(rewards) / len(rewards):.4f}"
            )

        return loss

    def test_step(self, batch, batch_idx):
        """Execute test step with PCGRL critic."""
        loss, entropy, rewards = self._process_batch(batch)

        # Log metrics
        self.log("test_loss", loss)
        self.log("test_entropy", entropy)
        self.log("test_reward_mean", sum(rewards) / len(rewards))
        self.log("test_reward_min", min(rewards))
        self.log("test_reward_max", max(rewards))

        # Only log the first test batch to avoid cluttering logs
        if batch_idx == 0:
            console.print(
                f"[bold blue]Test started[/bold blue] - First batch: "
                + f"loss={loss:.4f}, entropy={entropy:.4f}, "
                + f"reward_mean={sum(rewards) / len(rewards):.4f}"
            )

        return loss

    def generate_level(self, x, hero=None, temperature=1.0, deterministic=False):
        """Generate a level from input conditioning.

        Args:
            x: Input tensor [1, n_channels, H, W]
            hero: Optional hero tensor [1, hero_size]
            temperature: Sampling temperature (higher = more random)
            deterministic: If True, use argmax instead of sampling

        Returns:
            Generated level map as a 2D list of integers
        """
        with torch.no_grad():
            # Get logits from model
            logits = self(x, hero)

            if deterministic:
                # Use argmax (greedy)
                predictions = torch.argmax(logits, dim=1)  # [1, H, W]
                level_map = predictions[0].cpu().numpy().tolist()
            else:
                # Apply temperature scaling
                logits = logits / temperature

                # Sample from the distribution
                probs = F.softmax(logits, dim=1)  # [1, n_classes, H, W]
                categorical = torch.distributions.Categorical(
                    probs=probs[0].permute(1, 2, 0)
                )
                sample = categorical.sample()  # [H, W]
                level_map = sample.cpu().numpy().tolist()

            # Evaluate with critic
            try:
                reward = self.critic(level_map)
                log.info(f"Generated level with reward: {reward:.4f}")
            except Exception as e:
                log.error(f"❌ Failed to evaluate generated level: {str(e)}")

            return level_map

    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        log.info(f"Configured Adam optimizer with learning rate: {self.hparams.lr}")
        return optimizer

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Log overall statistics
        console.print(f"[green]Epoch {self.current_epoch} completed[/green]")
        console.print(f"  • Average reward: {self.avg_reward:.4f}")
        console.print(f"  • Reward baseline: {self.reward_baseline.item():.4f}")

        # Reset reward tracking for next epoch
        self.avg_reward = 0.0
        self.reward_count = 0

    def enable_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency.

        Applies checkpointing to encoder layers for memory savings.
        Note: Up layers aren't checkpointed due to multiple inputs.
        """
        try:
            console.print("[bold blue]Enabling gradient checkpointing...[/bold blue]")
            memory_before = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )

            # Apply checkpointing to encoder blocks
            layers_to_checkpoint = ["inc", "down1", "down2", "down3", "down4", "outc"]
            for layer_name in layers_to_checkpoint:
                layer = getattr(self, layer_name)
                setattr(
                    self,
                    layer_name,
                    lambda x, layer=layer: torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    ),
                )
                log.info(f"  • Applied checkpointing to {layer_name}")

            memory_after = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            if torch.cuda.is_available():
                memory_saved = (
                    memory_before - memory_after if memory_before > memory_after else 0
                )
                console.print(
                    f"[green]✅ Checkpointing enabled for {len(layers_to_checkpoint)} layers[/green]"
                )
                if memory_saved > 0:
                    console.print(
                        f"[green]   Memory saved: {memory_saved / 1024**2:.2f} MB[/green]"
                    )
            else:
                console.print(
                    "[yellow]⚠️ CUDA not available, memory savings not measured[/yellow]"
                )

        except Exception as e:
            console.print(
                f"[bold red]Failed to enable checkpointing:[/bold red] {str(e)}"
            )
            console.print_exception()
            log.error("❌ Checkpointing failed")
            raise
