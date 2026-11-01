"""Train and evaluate the feedforward SNN on MNIST."""

from datetime import datetime
from pathlib import Path

import tonic
import tonic.transforms as tonictransforms
from tonic import DiskCachedDataset
from tonic import MemoryCachedDataset

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms

#from snn_project.models.feedforward_snn import FeedForwardSNN, save_model
from snn_project.models.convolutional_snn import ConvolutionalSNN, save_model
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

    sensor_size = tonic.datasets.NMNIST.sensor_size

    event_transform = transforms.Compose([
            tonictransforms.Denoise(filter_time=10000),
            tonictransforms.ToFrame(sensor_size=sensor_size,
            time_window=1000)
        ])

    event_transform_cache = tonic.transforms.Compose([
            torch.from_numpy,
            torchvision.transforms.RandomRotation([-10,10])
        ])

    '''
    transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ])
    '''

    generator = torch.Generator().manual_seed(0)

    event_train_data = tonic.datasets.NMNIST(save_to='./data', transform=event_transform, train=True)
    event_test_data = tonic.datasets.NMNIST(save_to='./data', transform=event_transform, train=False)
    subset_length_train = int(len(event_train_data) * 1)
    #subset_length_test = int(len(event_test_data) * 1)
    indices_train = torch.randperm(len(event_train_data), generator=generator)[:subset_length_train]
    #indices_test = torch.randperm(len(event_test_data), generator=generator)[:subset_length_test]
    event_train_data = Subset(event_train_data, indices_train.tolist())
    #event_test_data = Subset(event_test_data, indices_test.tolist())

    #cached_train_data = MemoryCachedDataset(event_train_data)
    #cached_test_data = MemoryCachedDataset(event_test_data)
    cached_train_data = DiskCachedDataset(event_train_data, transform=event_transform_cache, cache_path='./cache/nmnist/train')
    cached_test_data = DiskCachedDataset(event_test_data, transform=event_transform_cache, cache_path='./cache/nmnist/test')

    
    train_loader = DataLoader(cached_train_data, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True, num_workers=2, pin_memory=device.type=="cuda")
    test_loader = DataLoader(cached_test_data, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), num_workers=2, pin_memory=device.type=="cuda")    
    
    #train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    #test_data = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    #model = FeedForwardSNN().to(device)
    model = ConvolutionalSNN().to(device)
    print(f"Training on {len(cached_train_data)} images for {num_epochs} epoch(s)...", flush=True)
    
    history = train(model, train_loader, test_loader, device=device, num_epochs=num_epochs)
    print(f"Final training batch loss: {history['loss_hist'][-1]:.3f}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    checkpoint_path = data_path.parents[1] / "checkpoints" / f"feedforward_snn_{timestamp}.pt"
    save_model(model, checkpoint_path)
    print(f"Saved model: {checkpoint_path}", flush=True)

    print(f"Evaluating on {len(cached_test_data)} images...", flush=True)
    figure_dir = data_path.parents[1] / "report" / "figures" / timestamp
    accuracy = evaluate(model, test_loader, device=device, history=history, figure_dir=figure_dir)
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Saved plots and spike animation to {figure_dir}")


if __name__ == "__main__":
    main()
