"""Choose an available device for training and evaluation."""

import torch


def get_device() -> torch.device:
    """Prefer an NVIDIA GPU, then Apple GPU, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
