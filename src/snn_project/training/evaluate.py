"""Spike-count evaluation and tutorial-style training plots."""

from pathlib import Path

import imageio_ffmpeg
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device | str | None = None,
    history: dict[str, list] | None = None,
    figure_dir: str | Path | None = None,
) -> float:
    """Evaluate every test example and optionally save plots and an animation."""

    model.to(device)
    model.eval()
    total = 0
    correct = 0
    example_spikes = None
    example_target = None

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            spikes, _ = model(data)
            if figure_dir is not None and example_spikes is None:
                example_spikes = spikes[:, 0].detach().cpu()
                example_target = targets[0].item()
            predicted = spikes.sum(dim=0).argmax(dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    if total == 0:
        raise ValueError("The test loader has no examples.")
    if figure_dir is not None:
        figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        if history is not None:
            plot_history(history, figure_dir)
        save_spike_count(example_spikes, example_target, figure_dir)
    return correct / total


def plot_history(history: dict[str, list], figure_dir: Path) -> None:
    """Save the notebook's loss and training accuracy plots."""
    fig, ax = plt.subplots(facecolor="w", figsize=(10, 5))
    ax.plot(
        range(1, len(history["loss_hist"]) + 1),
        history["loss_hist"],
        label="Train loss",
    )
    if history["test_loss_hist"]:
        ax.plot(
            history["test_steps"],
            history["test_loss_hist"],
            "o-",
            label="Test loss (batch)",
        )
    ax.set(title="Loss curves", xlabel="Training batch", ylabel="Spike-count loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(facecolor="w", figsize=(10, 5))
    ax.plot(range(1, len(history["acc_hist"]) + 1), history["acc_hist"])
    ax.set(title="Training accuracy", xlabel="Training batch", ylabel="Accuracy")
    fig.tight_layout()
    fig.savefig(figure_dir / "training_accuracy.png", dpi=150)
    plt.close(fig)


def save_spike_count(spikes: torch.Tensor, target: int, figure_dir: Path) -> None:
    """Save the notebook's spike-count animation for one test sample."""
    fig, ax = plt.subplots(facecolor="w", figsize=(12, 7))
    labels = [str(index) for index in range(spikes.shape[1])]
    ax.set_title(f"Test sample target: {target}")
    animation = splt.spike_count(
        spikes, fig, ax, labels=labels, animate=True, interpolate=1
    )
    with plt.rc_context({"animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe()}):
        animation.save(figure_dir / "spike_count.mp4", writer="ffmpeg")
    plt.close(fig)
