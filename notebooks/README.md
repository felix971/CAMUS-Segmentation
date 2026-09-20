# Notebook Usage Guide

This directory preserves existing learning records and outputs from computationally expensive runs. The formal training entry point for the hand-built U-Net is in the Python package; rerunning the notebooks is not required.

| File | Purpose | When to run it |
|---|---|---|
| `01_understand_camus_dataset.ipynb` | Data, masks, tensors, convolutions, step-by-step U-Net construction, and a single gradient update | When learning concepts or inspecting the data |
| `02_train_camus_with_nnunetv2.ipynb` | The complete single-fold 2D nnU-Net v2 workflow and existing outputs | When a specific step is needed; inspect existing artifacts first |
| `reference/script_camus_ef.ipynb` | Reference code for EF / Simpson's biplane method | For learning and reference; it does not mean that this project has completed clinical validation of EF |

- `Step n` in Markdown identifies a logical step; `In[n]` is only an execution count and changes after a restart.
- Restarting the kernel loses variables, but does not delete `data/nnunet/` or checkpoints. Do not repeat preprocessing or training just because a variable is undefined.
- The hand-built U-Net column in Step 10 of notebook 02 contains **historical reference values from the original 10-epoch run**; it does not automatically read the latest model from resumed training. Do not treat it as the result of the new 50-epoch experiment.
- Existing notebook outputs are retained for now, and the notebooks are not rerun in bulk during cleanup.
- The notebooks contain machine-specific path settings. When switching computers, check and update `PROJECT_ROOT` first; do not treat absolute paths in historical outputs as universal installation paths.
- The saved ultrasound, annotation, and prediction examples come from anonymized CAMUS research cases and are shown only for noncommercial research purposes. See [CAMUS_LICENSE.md](../CAMUS_LICENSE.md) for the dataset citation and applicable terms, including CC BY-NC-SA 4.0. The complete raw dataset and model weights are not distributed with the repository.

Next: [Formal Training Guide](../docs/training.md) · [Repository Guide](../docs/repository-guide.md) · [Experiment Register](../docs/experiments.md)
