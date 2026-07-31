# CAMUS Deep Learning Segmentation

A learning-led, reproducible project for multi-structure segmentation of 2D
echocardiography from the CAMUS dataset.

The project is being developed incrementally so that every data, PyTorch, CUDA,
modeling, training, and evaluation decision can be explained clearly in a
technical interview.

## Current status

- Reproducible Python 3.12 environment managed by `uv`.
- PyTorch 2.12.1 with the CUDA 13.2 runtime verified on an NVIDIA RTX 4090.
- CAMUS directory layout and patient metadata documented interactively.
- NIfTI image-mask inspection in progress.
- Model implementation and training have not started yet.

## Repository layout

```text
dl-segmentation-camus/
├── data/
│   ├── raw/camus/              # Local CAMUS dataset; never committed
│   └── splits/                 # Patient-level train/validation/test splits
├── docs/
│   ├── adr/                    # Architecture decision records
│   └── glossary.md
├── notebooks/
│   ├── 01_understand_camus_dataset.ipynb
│   └── reference/              # Dataset-provided reference material
├── CAMUS_LICENSE.md
├── pyproject.toml
└── uv.lock
```

## Environment setup

Install [`uv`](https://docs.astral.sh/uv/), clone the repository, and reproduce
the locked environment:

```bash
uv sync --locked
```

The project uses a CUDA-enabled PyTorch wheel. A system CUDA Toolkit and `nvcc`
are not required for the current prebuilt PyTorch workflow, but a compatible
NVIDIA driver is required to use the GPU.

For VS Code notebooks, select the interpreter at `.venv/bin/python`.

## Dataset placement

The CAMUS data is not redistributed by this repository. After accepting the
dataset terms, place the patient directories under:

```text
data/raw/camus/patient0001/
...
data/raw/camus/patient0500/
```

Raw data, virtual environments, generated outputs, and temporary files are
excluded from Git.

## Dataset citation

Any use of the CAMUS database must cite:

> S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al. “Deep Learning for
> Segmentation using an Open Large-Scale Dataset in 2D Echocardiography.” IEEE
> Transactions on Medical Imaging, 38(9), 2198–2210, 2019.
> https://doi.org/10.1109/TMI.2019.2900516

See `CAMUS_LICENSE.md` for the local copy of the dataset terms.
