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
|   |   `-- snn.py           # Network architectures and spiking neuron layers
|   |-- training/
|   |   |-- train.py         # Training loops and checkpoint saving
|   |   `-- evaluate.py      # Evaluation routines and metrics
|   `-- utils/
|       `-- seed.py          # Random seed and reproducibility helpers
|-- tests/
|   `-- test_dependencies.py # Checks if prebuilt LIF neuron and packages are working
|-- data/
|   |-- raw/                 # Original datasets
|   |-- processed/           # Preprocessed data
|   `-- README.md            # Dataset sources and preprocessing notes
|-- pyproject.toml           # Project metadata, dependencies, and build backend
|-- uv.lock                  # Generated dependency lockfile
|-- .python-version          # Project Python version
|-- .gitignore               # Excludes local environments, caches, and datasets
`-- .venv/                   # Local environment managed by uv; not tracked
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
