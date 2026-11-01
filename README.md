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

## Tutorial 5 model

To train on MNIST and evaluate the trained model:

```powershell
uv run python -m snn_project.main
```

The script downloads MNIST into `data/raw/` if needed, trains for one epoch, and reports test-set accuracy. It selects CUDA when available, otherwise CPU. The seed, batch size, and epoch count are set at the top of `main()`. After training, the script saves the model under `checkpoints/` with a unique timestamp. Checkpoint files are ignored by Git. 

The file contains model settings and parameters; optimizer state is not saved. To load a saved model:

```python
from snn_project.models.feedforward_snn import load_model

model = load_model("checkpoints/<filename>.pt", device="cpu")
```

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
