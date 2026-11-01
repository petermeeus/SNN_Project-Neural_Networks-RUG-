"""Set the seed for PyTorch random number generation."""

import torch


def set_seed(seed: int) -> None:
    """Seed PyTorch before model initialization and data-loader iteration."""
    torch.manual_seed(seed)
