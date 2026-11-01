"""Training loop from Tutorial 5."""

import torch
from snntorch import functional as SF
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
    device: torch.device | str | None = None,
    num_epochs: int = 1,
    learning_rate: float = 2e-2,
    test_interval: int = 10,
) -> dict[str, list]:
    """Train and record test-batch loss at the first, interval, and final batches."""
    
    if test_interval < 1:
        raise ValueError("test_interval must be at least 1")
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    history: dict[str, list] = {
        "loss_hist": [],
        "acc_hist": [],
        "test_loss_hist": [],
        "test_steps": [],
    }
    test_iterator = None

    for epoch in range(num_epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for batch_index, (data, targets) in enumerate(progress):
            data, targets = data.to(device), targets.to(device)
            model.train()
            spk_rec, _ = model(data)
            loss_val = loss_fn(spk_rec, targets)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            batch_loss = loss_val.item()
            batch_acc = SF.accuracy_rate(spk_rec, targets)
            history["loss_hist"].append(batch_loss)
            history["acc_hist"].append(batch_acc)
            progress.set_postfix(loss=f"{batch_loss:.3f}, {batch_acc:.1%}")

            step = len(history["loss_hist"])
            is_last_batch = batch_index + 1 == len(train_loader)
            if test_loader is not None and (
                step == 1 or step % test_interval == 0 or is_last_batch
            ):
                model.eval()
                with torch.no_grad():
                    if test_iterator is None:
                        test_iterator = iter(test_loader)
                    try:
                        test_data, test_targets = next(test_iterator)
                    except StopIteration:
                        test_iterator = iter(test_loader)
                        test_data, test_targets = next(test_iterator)
                    test_data = test_data.to(device)
                    test_targets = test_targets.to(device)
                    test_spikes, _ = model(test_data)
                    test_loss = loss_fn(test_spikes, test_targets)
                    history["test_loss_hist"].append(test_loss.item())
                    history["test_steps"].append(step)

    return history
