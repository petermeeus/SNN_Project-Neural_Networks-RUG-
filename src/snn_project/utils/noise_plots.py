"""Figures for the noise experiment; all panels use the recorded evaluation spikes."""

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_accuracy(rows, path, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([100 * result["noise"] for result in rows], [100 * result["accuracy"] for result in rows], marker="o")
    ax.set(xlabel="Probability of event/spike removal (%)", ylabel="Test accuracy (%)", title=title, ylim=(0, 100))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_rasters(example, path, title):
    """Plot every flattened neuron; input dots mean >=1 event in a 1 ms bin."""
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 9))
    for ax, name in zip(axes, ("input", "layer1", "layer2", "output")):
        values = example[name].reshape(len(example[name]), -1)
        times, neurons = np.nonzero(values)
        ax.scatter(times, neurons, s=0.5, color="black", rasterized=True)
        ax.set(ylabel=f"{name}\nNeuron index", ylim=(-0.5, values.shape[1] - 0.5))
    axes[-1].set(
        xlabel="Time step (1 ms bins; includes batch padding)",
        xlim=(-0.5, len(values) - 0.5),
    )
    fig.suptitle(
        title + "\nInput: polarity × pixel; hidden: channel × pixel; output: digit"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def channel_grid(frames):
    """Tile every channel in a square-ish grid without averaging activations."""
    time, channels, height, width = frames.shape
    columns = int(np.ceil(np.sqrt(channels)))
    rows = int(np.ceil(channels / columns))
    grid = np.zeros((time, rows * (height + 1), columns * (width + 1)))
    for channel in range(channels):
        row, column = divmod(channel, columns)
        grid[
            :,
            row * (height + 1) : row * (height + 1) + height,
            column * (width + 1) : column * (width + 1) + width,
        ] = frames[:, channel]
    return grid


def save_animation(example, path, title, fps=25):
    """Input channels, both hidden layers and cumulative output counts, synchronized."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    images = []
    grids = [channel_grid(example[name]) for name in ("input", "layer1", "layer2")]
    for ax, grid, name in zip(
        axes[:3],
        grids,
        ("Input (both polarities)", "Layer 1 (all channels)", "Layer 2 (all channels)"),
    ):
        images.append(
            ax.imshow(grid[0], cmap="gray_r", vmin=0, vmax=max(1, grid.max()))
        )
        ax.set_title(name)
        ax.axis("off")
    counts = example["output"].cumsum(axis=0)
    bars = axes[3].bar(np.arange(counts.shape[1]), counts[0], color="steelblue")
    bars[example["label"]].set_color("darkorange")
    axes[3].set(
        title="Accumulated output spikes",
        xlabel="Digit (orange = target)",
        ylabel="Spike count",
        ylim=(0, max(1, counts.max()) * 1.05),
        xticks=np.arange(counts.shape[1]),
    )
    heading = fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.87))

    def update(step):
        for image, grid in zip(images, grids):
            image.set_data(grid[step])
        for bar, count in zip(bars, counts[step]):
            bar.set_height(count)
        heading.set_text(
            f"{title}\nTime step {step} (1 ms/bin; includes batch padding; slowed playback)"
        )
        return [*images, *bars, heading]

    animation = FuncAnimation(
        fig, update, frames=len(counts), interval=1000 / fps, blit=False
    )
    try:
        with plt.rc_context({"animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe()}):
            animation.save(path, writer="ffmpeg", fps=fps)
    finally:
        plt.close(fig)
