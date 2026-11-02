"""Training loop from Tutorial 5."""

from time import perf_counter

import torch
from snntorch import functional as SF
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader | None = None,
    device: torch.device | str | None = None,
    num_epochs: int = 1,
    learning_rate: float = 2e-2,
    validation_interval: int = 10,
) -> dict:
    """Train and record validation batches. 
    Wall time is recorded for data loading and validation monitoring.
    Output activity counts spikes per presentation, including for the padded timesteps.
    """
    
    if validation_interval < 1 or num_epochs < 1:
        raise ValueError("validation_interval and num_epochs must be at least 1")
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    history = {
        "loss_hist": [],
        "acc_hist": [],
        "validation_loss_hist": [],
        "validation_steps": [],
    }
    validation_iterator = None
    total_spikes = total_examples = 0
    active_device = next(model.parameters()).device

    def synchronize():
        if active_device.type == "cuda":
            torch.cuda.synchronize(active_device)
        elif active_device.type == "mps":
            torch.mps.synchronize()
    synchronize()
    start = perf_counter()

    for epoch in range(num_epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for batch_index, (data, targets) in enumerate(progress):
            data, targets = data.to(device, dtype=torch.float32), targets.to(device)
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
            total_spikes += spk_rec.detach().sum().item()
            total_examples += targets.numel()
            progress.set_postfix(loss=f"{batch_loss:.3f}, {batch_acc:.1%}")

            step = len(history["loss_hist"])
            is_last_batch = batch_index + 1 == len(train_loader)
            if validation_loader is not None and (step == 1 or step % validation_interval == 0 or is_last_batch):
                model.eval()
                with torch.no_grad():
                    if validation_iterator is None:
                        validation_iterator = iter(validation_loader)
                    try:
                        validation_data, validation_targets = next(validation_iterator)
                    except StopIteration:
                        validation_iterator = iter(validation_loader)
                        validation_data, validation_targets = next(validation_iterator)
                    validation_data = validation_data.to(device, dtype=torch.float32)
                    validation_targets = validation_targets.to(device)
                    validation_spikes, _ = model(validation_data)
                    validation_loss = loss_fn(validation_spikes, validation_targets)
                    history["validation_loss_hist"].append(validation_loss.item())
                    history["validation_steps"].append(step)
    
    if not total_examples:
        raise ValueError("The training loader has no examples.")
    synchronize()
    history["training_seconds"] = perf_counter() - start
    history["training_output_spikes_per_sample"] = total_spikes / total_examples
    return history
