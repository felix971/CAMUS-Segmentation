# CAMUS 2D Echocardiography Segmentation

Hand-built PyTorch U-Net and nnU-Net v2 baselines for segmenting the left
ventricular cavity, myocardium, and left atrium in CAMUS echocardiography.
The emphasis is an explainable pipeline and traceable experiments, not a claim
of clinical readiness or a new state-of-the-art architecture.

## Start here

| What you need | Entry point |
|---|---|
| Project layout and file responsibilities | [Repository guide](docs/repository-guide.md) |
| Training, resuming, plotting, and evaluation | [Training guide](docs/training.md) |
| Historical baselines versus new experiments | [Experiment register](docs/experiments.md) |
| Detailed methods, original configurations, and verified results | [Completed baselines](docs/baselines.md) |
| Learning about data, tensors, and the hand-built U-Net | [Notebook guide](notebooks/README.md) |
| Terminology and runtime environment | [Glossary](docs/glossary.md) · [CUDA decision](docs/adr/0001-cuda-runtime-strategy.md) |

## Verified historical results

Same patient-level split: **400 training / 50 validation / 50 test patients**.
Each patient provides 2CH/4CH views at ED/ES, yielding 1,600 / 200 / 200 frames.

| Test Dice | Hand-built U-Net, 10 epochs | nnU-Net v2 2D, fold 0 |
|---|---:|---:|
| Left ventricular cavity | 0.9089 | 0.9446 |
| Myocardium | 0.8199 | 0.8920 |
| Left atrium | 0.8598 | 0.9239 |
| Mean foreground | **0.8629** | **0.9202** |

Dice is calculated per image and foreground class, then averaged. Background
is excluded. These are **historical baseline results**, not results from the new
50-epoch continuation. Full provenance and limitations are in
[the experiment register](docs/experiments.md).

This is a **pipeline-level comparison**, not an architecture ablation: training
budgets, optimizer, augmentation and preprocessing differ. The hand-built model
is evaluated on resized `512 × 416` masks; nnU-Net on the native image grid.
The nominal budgets are 2,000 versus 250,000 optimizer updates. One split and
one run do not establish seed stability or external generalization.

![Historical U-Net validation example](docs/assets/validation_prediction.png)

## Completed continuation experiment

The epoch-10 model and AdamW state were restored and training completed through
**epoch 50 / 10,000 cumulative updates**. The validation-selected best is
**epoch 23**, not the final checkpoint. Independently recomputed mean foreground
validation Dice increased from **0.8635 to 0.8870** (+2.35 percentage points)
on 50 validation patients / 200 images. Later training loss continued falling
while validation loss rose; further budget alone is not supported by this curve.

On 2026-09-21, the fixed epoch-23 checkpoint was evaluated on the existing
**50 test patients / 200 images**, using the same resized-grid protocol as the
original hand-built baseline. Test mean foreground Dice increased from
**0.8629 to 0.8777** (+1.48 percentage points). No training or checkpoint
reselection was performed during this test evaluation.

| Test Dice | Original epoch 10 | Validation-selected epoch 23 |
|---|---:|---:|
| Left ventricular cavity | 0.9089 | 0.9188 |
| Myocardium | 0.8199 | 0.8388 |
| Left atrium | 0.8598 | 0.8755 |
| Mean foreground | 0.8629 | **0.8777** |

The test cohort had previously been used for baseline evaluation and descriptive
failure analysis; it is not a newly collected or never-inspected cohort.
The nnU-Net comparison still uses different training recipes and evaluation grids.

[Recorded curves and interpretation](docs/experiments.md) ·
[Metrics + provenance JSON](docs/results/unet_resume10_to50.json) ·
[Epoch history CSV](docs/results/unet_resume10_to50_history.csv) ·
[Epoch-23 test report](docs/results/unet_resume10_to50_test.json)

New runs belong in `outputs/runs/<run-name>/`, never in the old checkpoint directory.

The old checkpoint lacks RNG state. Seed 42 initializes the continuation
stage only; this is not an exact reconstruction of an uninterrupted run from
epoch 1. Historical missing curves are not fabricated. Training uses only the
training and validation sets; test evaluation remains a separate explicit step.

See [the training guide](docs/training.md) for the exact command, safe restart
procedure, and the meanings of `baseline.pt`, `best.pt`, and `last.pt`.
Check the run's `summary.json` and `history.csv` for actual completion and scores;
a configuration targeting epoch 50 is not evidence that epoch 50 finished.

## Environment and data

Python 3.12; locked dependencies are in `uv.lock`:

```bash
uv sync --locked
.venv/bin/python -m camus_segmentation.train --help
.venv/bin/python -m unittest discover -s tests -v
```

PyTorch's prebuilt CUDA runtime requires a compatible NVIDIA driver, not a
separate system CUDA Toolkit. The numerical training recipe does not add AMP,
augmentation, or a scheduler during the budget extension.

Download CAMUS from its [official dataset page](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html)
after accepting the terms. Place patient folders in `data/raw/camus/`.
The repository does **not** redistribute CAMUS data or trained checkpoints.
Existing nnU-Net artifacts under `data/nnunet/` should be reused; do not rerun
an entire notebook merely because its kernel was restarted.

## Repository layout

```text
camus_segmentation/    # Reusable model, training, evaluation and plotting
notebooks/            # Preserved learning and nnU-Net workflows
examples/             # Small standalone learning examples
tests/                # Lightweight CPU regression tests
docs/                 # Navigation, commands, methods and experiment register
data/splits/          # Versioned patient-level split definitions
data/raw/camus/       # Local licensed data (ignored by Git)
data/nnunet/          # Existing nnU-Net workspace (ignored)
outputs/baselines/    # Frozen historical weights (ignored)
outputs/runs/         # One independent directory per experiment segment (ignored)
outputs/checkpoints/  # Legacy epoch-10 path; preserved for old notebooks (ignored)
```

## Dataset citation and terms

> S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al. “Deep Learning for
> Segmentation using an Open Large-Scale Dataset in 2D Echocardiography.”
> IEEE Transactions on Medical Imaging, 38(9), 2198–2210, 2019.
> https://doi.org/10.1109/TMI.2019.2900516

See [CAMUS_LICENSE.md](CAMUS_LICENSE.md). Educational/non-commercial research
only, subject to the CAMUS terms. This is not a clinical system.
