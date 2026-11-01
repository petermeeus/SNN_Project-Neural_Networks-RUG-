"""Spike-count accuracy from Tutorial 5."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device | str = "cpu") -> float:
    """Return accuracy over all test examples, including a partial last batch."""
    model.to(device)
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            spikes, _ = model(data)
            predicted = spikes.sum(dim=0).argmax(dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    if total == 0:
        raise ValueError("The test loader has no examples.")
    return correct / total
