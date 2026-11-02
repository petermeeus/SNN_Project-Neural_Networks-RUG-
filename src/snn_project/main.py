"""Train and evaluate the feedforward SNN on MNIST."""

import json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import tonic
import tonic.transforms as tonictransforms
import torch
import torchvision
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

#from snn_project.models.feedforward_snn import FeedForwardSNN, save_model
from snn_project.models.convolutional_snn import ConvolutionalSNN, save_model
from snn_project.training.evaluate import evaluate
from snn_project.training.train import train
from snn_project.utils.device import get_device
from snn_project.utils.event_frames import save_event_frames
from snn_project.utils.seed import set_seed


def main() -> None:
    batch_size = 128
    num_epochs = 5
    seeds = (0, 1, 2)
    split_seed = 0
    val_candidates = ((0.001, 0.5), (0.01, 0.5), (0.001, 0.9))

    root_path = Path(__file__).resolve().parents[2]
    data_path = root_path / "data"
    cache_path = root_path / "cache" / "nmnist" / "denoise10000_frames1000_complete"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    figure_dir = root_path / "report" / "figures" / timestamp
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}", flush=True)
    
    event_transform = transforms.Compose([
            tonictransforms.Denoise(filter_time=10000),
            tonictransforms.ToFrame(
                sensor_size=tonic.datasets.NMNIST.sensor_size,
                time_window=1000,
                include_incomplete=True)
            ])
    train_transform = tonic.transforms.Compose([torch.from_numpy, torchvision.transforms.RandomRotation([-10,10])])

    event_train_data = tonic.datasets.NMNIST(save_to=str(data_path), transform=event_transform, train=True)
    event_test_data = tonic.datasets.NMNIST(save_to=str(data_path), transform=event_transform, train=False)

    indices = torch.randperm(len(event_train_data), generator=torch.Generator().manual_seed(split_seed)).tolist()
    validation_size = max(1, len(indices) // 10)
    validation_indices, train_indices = (indices[:validation_size], indices[validation_size:])
    
    training_frames = DiskCachedDataset(event_train_data, transform=train_transform, cache_path=str(cache_path / "train"))
    validation_frames = DiskCachedDataset(event_train_data, transform=torch.from_numpy, cache_path=str(cache_path / "train"))
    test_frames = DiskCachedDataset(event_test_data, transform=torch.from_numpy, cache_path=str(cache_path / "test"))

    def make_loader(dataset, shuffle=False, seed=0):
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=torch.Generator().manual_seed(seed),
                collate_fn=tonic.collation.PadTensors(batch_first=False),
                num_workers=2,
                pin_memory=device.type == "cuda",
            )

    validation_loader = make_loader(Subset(validation_frames, validation_indices))
    test_loader = make_loader(test_frames)
    (figure_dir / "splits.json").write_text(
            json.dumps(
                    {
                        "train": train_indices,
                        "validation": validation_indices,
                        "test": list(range(len(event_test_data))),
                    }
                ),
                encoding="utf-8",
            )

    best_accuracy = -1
    validation_results = []

    for learning_rate, beta in val_candidates:
        set_seed(seeds[0])
        model = ConvolutionalSNN(beta=beta).to(device)
        
        print(f"Training on {len(train_indices)} images for {num_epochs} epoch(s)...", flush=True)
        history = train(
            model,
            make_loader(Subset(training_frames, train_indices), True, seeds[0]),
            validation_loader,
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )

        print(f"Evaluating on {len(validation_indices)} validation images...", flush=True)
        accuracy = evaluate(model, validation_loader, device=device)
        validation_results.append(
            {
                "learning_rate": learning_rate,
                "beta": beta,
                "accuracy": accuracy,
            }
        )
        print(f"Validation: lr={learning_rate}, beta={beta}: {accuracy:.2%}", flush=True)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameters = (learning_rate, beta)
            best_model, best_history = model.cpu(), history
    
    learning_rate, beta = best_parameters
    print(f"Selected hyperparameters: lr={learning_rate}, beta={beta}", flush=True)

    results = []
    for seed in seeds:
        # Reuse the selected first-seed model; train fresh models for other seeds.
        if seed == seeds[0]:
            model, history = best_model.to(device), best_history
        else:
            set_seed(seed)
            model = ConvolutionalSNN(beta=beta).to(device)
            history = train(
                model,
                make_loader(Subset(training_frames, train_indices), True, seed),
                validation_loader,
                device=device,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
            )

        checkpoint = (root_path / "checkpoints" / f"convolutional_snn_{timestamp}_seed{seed}.pt")
        save_model(model, checkpoint)
        metrics = evaluate(
            model,
            test_loader,
            device=device,
            history=history,
            figure_dir=figure_dir / f"seed-{seed}",
            return_metrics=True,
        )

        results.append(
            {
                "seed": seed,
                **metrics,
                "training_seconds": history["training_seconds"],
                "training_output_spikes_per_sample": history[
                    "training_output_spikes_per_sample"
                ],
            }
        )

        print(f"Seed {seed}: {results[-1]}", flush=True)
        model.cpu()

    accuracies = [result["accuracy"] for result in results]
    summary = {
        "learning_rate": learning_rate,
        "beta": beta,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "split_seed": split_seed,
        "validation_runs": validation_results,
        "test_runs": results,
        "test_accuracy_mean": mean(accuracies),
        "test_accuracy_std": stdev(accuracies),
        "training_seconds_mean": mean(r["training_seconds"] for r in results),
        "output_spikes_per_sample_mean": mean(
            r["output_spikes_per_sample"] for r in results
        ),
    }

    (figure_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_event_frames(validation_frames[train_indices[0]][0], figure_dir / "event_frames.png")
    print(f"Test accuracy: {mean(accuracies):.2%} +/- {stdev(accuracies):.2%} (sample SD)")
    print(
            f"Mean training time: {summary['training_seconds_mean']:.1f}s; "
            f"output spikes/example: {summary['output_spikes_per_sample_mean']:.1f}"
    )
    print(f"Saved plots and spike animation to {figure_dir}")


if __name__ == "__main__":
    main()
