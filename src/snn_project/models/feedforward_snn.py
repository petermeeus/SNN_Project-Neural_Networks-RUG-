"""Feedforward spiking network from Tutorial 5."""

from pathlib import Path

import snntorch as snn
import torch
from torch import nn


class FeedForwardSNN(nn.Module):
    """Two linear layers with leaky integrate-and-fire neurons."""

    def __init__(self, num_inputs: int = 28 * 28, num_hidden: int = 1000, num_outputs: int = 10, num_steps: int = 25, beta: float = 0.95,) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.flatten(start_dim=1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)


def save_model(model: FeedForwardSNN, path: str | Path) -> None:
    """Save the network's settings and learned parameters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": {
                "num_inputs": model.fc1.in_features,
                "num_hidden": model.fc1.out_features,
                "num_outputs": model.fc2.out_features,
                "num_steps": model.num_steps,
                "beta": model.lif1.beta.item(),
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_model(path: str | Path, device: torch.device | str = "cpu") -> FeedForwardSNN:
    """Recreate a saved network on the requested device."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = FeedForwardSNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
