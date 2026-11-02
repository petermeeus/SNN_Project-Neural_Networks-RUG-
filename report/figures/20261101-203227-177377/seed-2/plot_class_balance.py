"""Plot the exact saved training subset used by this seed-2 run."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tonic
import torch
from matplotlib.ticker import StrMethodFormatter


def main():
    output_dir = Path(__file__).resolve().parent
    root = next(p for p in output_dir.parents if (p / "pyproject.toml").exists())
    run_dir = output_dir.parent
    splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert any(run["seed"] == 2 for run in results["test_runs"])
    assert (root / "data/NMNIST/Train").is_dir()
    dataset = tonic.datasets.NMNIST(save_to=str(root / "data"), train=True)
    train_indices = np.asarray(splits["train"], dtype=np.int64)
    validation_indices = np.asarray(splits["validation"], dtype=np.int64)
    assert len(np.unique(train_indices)) == len(train_indices)
    assert not np.intersect1d(train_indices, validation_indices).size
    assert len(train_indices) + len(validation_indices) == len(dataset)
    generated = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(results["split_seed"])
    ).numpy()
    assert np.array_equal(train_indices, generated[len(validation_indices):])
    targets = np.asarray(dataset.targets, dtype=np.int64)[train_indices]
    counts = np.bincount(targets, minlength=10)
    total = len(targets)
    assert len(counts) == 10 and counts.sum() == total

    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    bars = ax.bar(np.arange(10), counts, color="#3569A6", width=0.72)
    ax.bar_label(
        bars,
        labels=[f"{count:,}\n({count / total:.2%})" for count in counts],
        padding=5,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    ax.axhline(total / 10, color="#444444", linestyle="--", linewidth=1,
               label=f"Equal balance: {total // 10:,} per class")
    ax.set(
        xlabel="Digit class",
        ylabel="Training examples",
        xticks=np.arange(10),
        ylim=(0, counts.max() * 1.18),
        title=f"N-MNIST training class balance — seed 2\n"
              f"{total:,} examples · saved training split (split seed {results['split_seed']})",
    )
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E3E5E8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.savefig(output_dir / "training_class_balance.png", dpi=200)
    plt.close(fig)

    summary = {
        "dataset": "N-MNIST",
        "model_seed": 2,
        "split_seed": results["split_seed"],
        "source": str((run_dir / "splits.json").relative_to(root)),
        "labels_source": "tonic.datasets.NMNIST.targets from local data/NMNIST/Train",
        "training_examples": total,
        "counts": {str(i): int(count) for i, count in enumerate(counts)},
        "percentages": {str(i): float(count / total * 100) for i, count in enumerate(counts)},
        "saved_split_matches_generation": True,
    }
    (output_dir / "training_class_balance.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
