# Training Spiking Neural Networks - NN@RUG

## Setup

Make sure [uv](https://docs.astral.sh/uv/) is installed, then run these commands from the root directory:

```powershell
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format .
uv run jupyter lab
```

Activate the venv manually from PowerShell: `.\.venv\Scripts\Activate.ps1`.

## Project Structure

```text
.
|-- src/snn_project/          # Importable project code
|   |-- models/
|   |   `-- feedforward_snn.py # Two-layer spiking network
|   |-- training/
|   |   |-- train.py         # Training loop
|   |   `-- evaluate.py      # Test-set spike-count accuracy
|   `-- utils/
|       |-- device.py        # Choose GPU or CPU
|       `-- seed.py          # Random seed and reproducibility helpers
|-- tests/
|   `-- test_dependencies.py # Checks if prebuilt LIF neuron and packages are working
|-- data/
|   |-- raw/                 # Original datasets
|   |-- processed/           # Preprocessed data
|   `-- README.md            # Dataset sources and preprocessing notes
|-- checkpoints/             # Saved models
|-- pyproject.toml           # Project metadata, dependencies, and build backend
|-- uv.lock                  # Generated dependency lockfile
|-- .python-version          # Project Python version
|-- .gitignore               # Excludes local environments, caches, and datasets
`-- .venv/                   # Local environment managed by uv; not tracked
```

## N-MNIST training

To train on N-MNIST and evaluate the convolutional SNN:

```powershell
uv run python -m snn_project.main
```

The settings are at the top of `main()`. A fixed 90/10 split of the training set selects from three `(learning_rate, beta)` choices using seed 0. The selected model is tested alongside fresh models trained with seeds 1 and 2, all on the same test set. Only training data receives random rotations. One epoch is default starting point, final results use more epochs.

`report/figures/<timestamp>/results.json` records per-seed accuracy, training time, output spikes per example, and mean test accuracy with sample standard deviation across seeds. `splits.json` records the exact indices. Timing includes loading, cache creation and validation checks, so the first run may be slower. Spike counts include padded timesteps and only measure output activity. Each seed saves a checkpoint under `checkpoints/`, plus training plots and a spike animation.

`utils/event_frames.py` shows selected 1 ms frames with both polarity channels. Frames retain spatial location, polarity counts and coarse time order; exact timestamps and ordering within each bin are lost. Denoising removes isolated events. Event counts describe brightness changes, not absolute image intensity.

The script selects CUDA when available, then MPS, otherwise CPU. On Windows, uv installs CUDA 13.0 builds of PyTorch and torchvision.

The file contains model settings and parameters; optimizer state is not saved. To load a saved model:

```python
from snn_project.models.convolutional_snn import load_model

model = load_model("checkpoints/<filename>.pt", device="cpu")
```

## Noise experiment

Run `uv run python -m snn_project.noise_experiment`. Settings are defined directly
inside `main()`: checkpoint path, batch size, workers and noise levels. Set
`subset_length_test` to 128 for a quick preview, or leave it as the full test set.

The saved model runs once per noise level for input, layer-1, then layer-2 noise.
The default performs 15 evaluation passes and renders 15 animations. Results are
single accuracies, with no repeated seeds or error bars.

`noise_experiment.py` contains the experiment; `utils/noise_plots.py` contains the
figures. Each experiment saves `accuracy.png`, scores in `results.json`, and a
four-layer raster PNG and synchronized MP4 for every noise level under
`report/figures/noise-<timestamp>/`. The same first test example is recorded using
the actual evaluation batch and noise mask. The animation shows both input
polarities, every hidden feature map, and accumulating digit spike counts.
Rasters show every flattened neuron; an input dot means at least one event in
that pixel/polarity/bin (multiplicity is not shown).

Input noise independently removes events **after denoising** by binomial thinning
of frame counts. It keeps the original 1 ms bins, lengths and batch padding.
Hidden-layer noise removes outgoing spikes independently per neuron/time step
without rescaling; the neuron's membrane and reset mechanism are preserved.
One fixed random state makes the experiment reproducible. Padding is included
as in the existing evaluator. Keep batch size/order fixed for comparisons, and
report a new clean baseline when using a smaller subset.

## Dependencies

- `torch`: tensors, automatic differentiation, neural network layers.
- `snntorch`: spiking neurons and surrogate gradients, built on PyTorch.
- `numpy`, `scipy`: array operations and scientific/numerical tools.
- `matplotlib`: plots of losses, spikes, and results.
- `torchvision`: image datasets such as MNIST and image transforms.
- `scikit-learn`: evaluation metrics such as confusion matrices and F1.
- `tqdm`: progress bars for training loops.
- `tonic`: event-based datasets such as N-MNIST and preprocessing transforms.
- `pytest` (development): runs tests.
- `ruff` (development): linting and formatting, replacing Flake8 and Black for this project.
- `jupyterlab`, `ipykernel` (development): a notebook editor and the Python kernel that runs `.ipynb` cells. Select this project's `.venv` in your editor.
- `hatchling` (build backend): lets uv install `src/snn_project` as an importable package.
