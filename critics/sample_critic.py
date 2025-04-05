def simple_critic(map_batch: torch.Tensor, hero_batch: torch.Tensor) -> torch.Tensor:
    """
    Simple critic: reward based on unique non-empty tiles + small bonus for health.
    """
    n_batch, w, h = map_batch.shape
    rewards = torch.zeros(n_batch, device=map_batch.device)
    for i in range(n_batch):
        # Count unique tile types > 0 (ignore empty/unchanged)
        unique_tiles, counts = torch.unique(map_batch[i], return_counts=True)
        non_empty_unique_count = torch.sum(unique_tiles > 0).item()

        # Example: Reward diversity of tiles + small health contribution
        reward = float(non_empty_unique_count)

        # Add contribution from hero state (e.g., health)
        health = hero_batch[i, 0].item()  # Index 0 assumed to be health
        reward += health * 0.05  # Small bonus based on health

        rewards[i] = reward
    return rewards
