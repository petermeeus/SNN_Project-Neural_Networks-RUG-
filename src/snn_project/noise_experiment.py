""" Evaluate our saved CSNN under input, first-layer and second-layer spike loss.

    Run: uv run python -m snn_project.noise_experiment
"""

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import tonic
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from snn_project.models.convolutional_snn import load_model
from snn_project.utils.device import get_device
from snn_project.utils.noise_plots import plot_accuracy, save_animation, save_rasters


class InputNoise(Dataset):
    """Binomially deleting events over a supplied frame"""

    def __init__(self, dataset, probability=0.0):
        self.dataset, self.probability = dataset, probability

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        frames, label = self.dataset[index]
        if self.probability:
            rng = np.random.default_rng(index)
            frames = rng.binomial(np.asarray(frames).astype(np.int64), 1 - self.probability)

        return torch.as_tensor(frames, dtype=torch.float32), int(label)


@contextmanager
def spike_noise(model, experiment, probability, trace=None):
    """Suppress outgoing spikes; preserve the neuron's own membrane/reset state.

    When requested, record every neuron for example 0 in this batch, after noise.
    Hooks are removed on exit, including when a forward pass fails.
    """
    handles = []

    def hook_for(name):
        def hook(module, inputs, output):
            spikes, membrane = output

            if name == experiment and probability:
                keep = torch.rand_like(spikes)
                spikes = spikes * (keep >= probability)

            if trace is not None:
                trace[name].append(spikes[0].detach().cpu())

            return spikes, membrane
        return hook

    try:
        for name, layer in (("layer1", model.lif1), ("layer2", model.lif2), ("output", model.lif3)):
            if name == experiment or trace is not None:
                handles.append(layer.register_forward_hook(hook_for(name)))
        yield

    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def evaluate_noise(model, test_loader, device, experiment, probability, record=False):
    """Return accuracy and optionally the exact first example used in evaluation."""
    model.eval()
    torch.manual_seed(0)
    correct = total = 0
    example = None
    for batch_index, (frames, labels) in enumerate(tqdm(test_loader, desc=f"{experiment}: {probability:.0%} spike loss", leave=False)):
        frames, labels = frames.to(device), labels.to(device)
        trace = ({name: [] for name in ("layer1", "layer2", "output")} if record and batch_index == 0 else None)

        with spike_noise(model, experiment, probability, trace):
            spikes, _ = model(frames)
            
        predictions = spikes.sum(0).argmax(1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
        if trace is not None:
            example = {name: torch.stack(values).numpy() for name, values in trace.items()}
            example["input"] = frames[:, 0].cpu().numpy()
            example["label"] = labels[0].item()
            example["prediction"] = predictions[0].item()

    if total == 0:
        raise ValueError("The evaluation dataset is empty.")
    return correct / total, example


def main() -> None:
    batch_size = 128
    num_workers = 2
    noise_levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99)
    experiments = ("input", "layer1", "layer2")

    root_path = Path(__file__).resolve().parents[2]
    data_path = root_path / "data"
    cache_path = root_path / "cache" / "nmnist" / "denoise10000_frames1000_complete"
    checkpoint_path = root_path / "checkpoints" / "convolutional_snn_20261101-203227-177377_seed0.pt"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    figure_dir = root_path / "report" / "figures" / f"noise-{timestamp}"
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}", flush=True)
    model = load_model(checkpoint_path, device)

    event_transform = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=1000, include_incomplete=True)
        ])
    event_test_data = tonic.datasets.NMNIST(save_to=str(data_path), transform=event_transform, train=False)
    test_frames = tonic.DiskCachedDataset(event_test_data, cache_path=str(cache_path / "test"))
    subset_length_test = len(test_frames)  # Something like 128 is a good shorter run
    test_subset = Subset(test_frames, range(subset_length_test))

    settings = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "samples": len(test_subset),
        "batch_size": batch_size,
        "noise_levels": noise_levels,
        "input_noise": "event deletion after denoising",
        "internal_noise": "outgoing spike deletion, no rescaling, membrane unchanged",
        "padding": "same sample order/batches and durations at every noise level; padded steps included",
    }
    (figure_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    for experiment in experiments:
        experiment_dir = figure_dir / experiment
        experiment_dir.mkdir()
        results = []

        for probability in noise_levels:
            noisy_test_data = InputNoise(test_subset, probability if experiment == "input" else 0)
            test_loader = DataLoader(noisy_test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda", collate_fn=tonic.collation.PadTensors(batch_first=False))
            accuracy, example = evaluate_noise(model, test_loader, device, experiment, probability, record=True)

            prefix = experiment_dir / f"noise-{round(probability * 100):02d}"
            title = f"{experiment}: {probability:.0%} spike loss | target {example['label']}, prediction {example['prediction']}"
            save_rasters(example, prefix.with_suffix(".png"), title)
            save_animation(example, prefix.with_suffix(".mp4"), title)

            results.append({"noise": probability, "accuracy": accuracy})
            (experiment_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            plot_accuracy(results, experiment_dir / "accuracy.png", experiment)
            print(f"{experiment}, {probability:.0%} loss: {accuracy:.2%}", flush=True)

        print(f"Saved {experiment}: {experiment_dir}", flush=True)


if __name__ == "__main__":
    main()
