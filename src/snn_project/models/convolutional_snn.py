"""Convolutional spiking network based on Tutorial 6."""

from pathlib import Path

import snntorch as snn
import torch
import torch.nn.functional as F
from snntorch import surrogate
from torch import nn


class ConvolutionalSNN(nn.Module):
    """Two convolutional layers and an output layer for 28x28 images."""

    def __init__(self, in_channels: int = 1, conv1_channels: int = 12, conv2_channels: int = 64, num_outputs: int = 10, num_steps: int = 50, beta: float = 0.5, surrogate_slope: float = 25,) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.surrogate_slope = surrogate_slope
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)

        self.conv1 = nn.Conv2d(in_channels, conv1_channels, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(conv2_channels * 4 * 4, num_outputs)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        mem3_rec = []

        for _ in range(self.num_steps):
            cur1 = F.max_pool2d(self.conv1(x), 2)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = F.max_pool2d(self.conv2(spk1), 2)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc1(spk2.flatten(start_dim=1))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec)


def save_model(model: ConvolutionalSNN, path: str | Path) -> None:
    """Save the network's settings and learned parameters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": {
                "in_channels": model.conv1.in_channels,
                "conv1_channels": model.conv1.out_channels,
                "conv2_channels": model.conv2.out_channels,
                "num_outputs": model.fc1.out_features,
                "num_steps": model.num_steps,
                "beta": model.lif1.beta.item(),
                "surrogate_slope": model.surrogate_slope,
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_model(
    path: str | Path, device: torch.device | str = "cpu"
) -> ConvolutionalSNN:
    """Recreate a saved convolutional network on the requested device."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ConvolutionalSNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
