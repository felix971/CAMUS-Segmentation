# Repository Guide: Separating Learning, Formal Experiments, and Historical Results

## Where to Start

- **Understand the project and verified results**: See the root [README](../README.md).
- **Run experiments with the hand-built U-Net**: See [Training and Resuming Training](training.md). The formal entry point is `python -m camus_segmentation.train`.
- **Learn about the data and network**: See [notebook 01](../notebooks/01_understand_camus_dataset.ipynb). It contains teaching code for building the network step by step and performing a single update; it is not the entry point for the formal 50-epoch training run.
- **Review the completed nnU-Net workflow**: See [notebook 02](../notebooks/02_train_camus_with_nnunetv2.ipynb). If preprocessing, training, and prediction artifacts already exist, do not rerun everything from the beginning just to view later results.
- **Check historical baselines**: See the [Experiment Register](experiments.md). The original 10-epoch results and newer resumed-training results must be labeled separately.
- **Verify the code**: ` .venv/bin/python -m unittest discover -s tests -v `.

## Directory Responsibilities

| Directory | What belongs here | What does not belong here |
|---|---|---|
| `camus_segmentation/` | Reusable code for data, models, training, evaluation, and plotting | Notebook drafts and training logs |
| `tests/` | Automated small-sample CPU tests using explicitly synthetic fixtures | Formal medical research results |
| `notebooks/` | Educational exploration and existing nnU-Net run records | The sole formal training entry point for the hand-built model |
| `notebooks/reference/` | Reference notebooks, such as the Simpson's method EF example | Clinical conclusions validated by this project |
| `examples/` | Small, standalone conceptual examples | Automated tests or formal experiments |
| `data/raw/camus/` | Local raw data subject to the applicable terms | Git-tracked files |
| `data/splits/` | Version-controlled patient splits | Splits changed ad hoc during training |
| `data/nnunet/` | Existing nnU-Net data links, preprocessing artifacts, weights, and predictions | Results of the current resumed training of the hand-built model |
| `outputs/baselines/` | Frozen copies of historical baselines and provenance notes | Checkpoints overwritten by new training |
| `outputs/runs/<run-name>/` | Configuration, logs, CSV files, weights, and curves for one experiment segment | Outputs from multiple unlabeled runs mixed together |
| `outputs/checkpoints/` | A historical path retained for older notebooks | The default output location for new experiments |
| `docs/` | Method descriptions, operating guides, and experiment records | Numbers without sources |
| `docs/assets/` | Figures selected for the README | Images automatically overwritten by every experiment |

## Why the Directory Tree Was Not Completely Reorganized

The existing notebooks and nnU-Net workspace depend on relative paths, symbolic links, and persistent artifacts. Moving `data/`, renaming packages, or clearing notebook outputs just to make things look tidy could disrupt completed training work.

This cleanup preserved those paths and all cells, code, and outputs in the original notebooks, while separating responsibilities through clear entry points, navigation, and individual run directories. The former root-level `test.py` was only a NumPy shape example, so it was moved to `examples/numpy_shapes.py` rather than being incorrectly counted as part of the test suite.

## Git and Large Files

`.gitignore` excludes `.venv/`, raw data, the nnU-Net workspace, and `outputs/`. Do not use `git add -f` to upload these directories just to commit results. Selected metrics in JSON/CSV format and selected images may be committed, but first check their provenance, the CAMUS terms, and whether they contain patient data.

Before the initial cleanup, the README and nnU-Net notebook already contained uncommitted work. The initial cleanup policy, established before publication, was to preserve that existing work rather than “clean it up” with `git reset --hard`, cleared outputs, or rebuilt notebooks, and not to commit or push automatically during that cleanup. This describes the historical cleanup policy, not the repository's current publication status.
