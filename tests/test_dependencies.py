"""Check that the core SNN dependencies support surrogate-gradient learning."""

import snntorch as snn
import torch
from snntorch import surrogate


def test_lif_surrogate_gradient() -> None:
    """A spike must propagate a finite, nonzero gradient to its input."""
    neuron = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
    current = torch.tensor([[1.5]], requires_grad=True)
    spikes, membrane = neuron(current, torch.zeros_like(current))

    assert spikes.shape == current.shape
    assert membrane.shape == current.shape
    assert spikes.item() == 1.0

    spikes.sum().backward()

    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert current.grad.abs().sum().item() > 0
