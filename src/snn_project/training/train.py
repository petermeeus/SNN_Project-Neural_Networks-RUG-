"""Training loop from Tutorial 5."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from snn_project.utils.device import get_device


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
    device: torch.device | str | None = None,
    num_epochs: int = 1,
    learning_rate: float = 5e-4,
) -> dict[str, list[float]]:
    """ Train with summed membrane cross-entropy and optional test batches. """
    
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    history: dict[str, list[float]] = {"loss_hist": [], "test_loss_hist": []}

    for epoch in range(num_epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for data, targets in progress:
            data, targets = data.to(device), targets.to(device)
            model.train()
            _, mem_rec = model(data)
            loss_val = sum(loss_fn(mem, targets) for mem in mem_rec)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            batch_loss = loss_val.item()
            history["loss_hist"].append(batch_loss)
            progress.set_postfix(loss=f"{batch_loss:.3f}")

            if test_loader is not None:
                model.eval()
                with torch.no_grad():
                    test_data, test_targets = next(iter(test_loader))
                    test_data = test_data.to(device)
                    test_targets = test_targets.to(device)
                    _, test_mem = model(test_data)
                    test_loss = sum(loss_fn(mem, test_targets) for mem in test_mem)
                    history["test_loss_hist"].append(test_loss.item())

    return history
