"""Show the two polarity channels of the frames supplied to the model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_event_frames(frames, path: str | Path, max_frames: int = 6) -> None:
    """Plot 1 ms N-MNIST frames, shaped [time, polarity, height, width].

    Includes the spatial location, event counts per polarity and time order.
    Does not include exact timestamps and event order within the bin.
    """

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[1] != 2 or len(frames) == 0 or max_frames < 1:
        raise ValueError(
            "Expected nonempty [time, 2, height, width] frames and max_frames >= 1."
        )
    
    indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames)), dtype=int)
    fig, axes = plt.subplots(2, len(indices), squeeze=False, figsize=(2 * len(indices), 4))

    for column, index in enumerate(indices):
        for polarity in (0, 1):
            ax = axes[polarity, column]
            ax.imshow(
                frames[index, polarity],
                cmap="gray_r",
                vmin=0,
                vmax=max(1, frames.max()),
            )
            ax.set_title(f"Bin {index}, polarity {polarity}")
            ax.axis("off")

    fig.suptitle("Event counts in 1 ms bins; both polarity channels reach the model")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
