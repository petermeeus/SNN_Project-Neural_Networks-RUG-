"""Train and evaluate the feedforward SNN on MNIST."""

from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from snn_project.models.feedforward_snn import FeedForwardSNN, save_model
from snn_project.training.evaluate import evaluate
from snn_project.training.train import train
from snn_project.utils.device import get_device
from snn_project.utils.seed import set_seed


def main() -> None:
    set_seed(0)
    batch_size = 128
    num_epochs = 1
    data_path = Path(__file__).resolve().parents[2] / "data" / "raw"

    device = get_device()
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

    transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ])
    
    train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = FeedForwardSNN().to(device)
    print(f"Training on {len(train_data)} images for {num_epochs} epoch(s)...", flush=True)
    
    history = train(model, train_loader, device=device, num_epochs=num_epochs)
    print(f"Final training batch loss: {history['loss_hist'][-1]:.3f}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    checkpoint_path = data_path.parents[1] / "checkpoints" / f"feedforward_snn_{timestamp}.pt"
    save_model(model, checkpoint_path)
    print(f"Saved model: {checkpoint_path}", flush=True)

    print(f"Evaluating on {len(test_data)} images...", flush=True)
    accuracy = evaluate(model, test_loader, device=device)
    print(f"Test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
