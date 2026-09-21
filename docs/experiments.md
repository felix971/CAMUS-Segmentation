# Experiment Register: Keep the Old Baseline Separate from Resumed Training

## 1. Frozen Hand-built U-Net 10-Epoch Baseline

- Original path: `outputs/checkpoints/best_model.pt` (retained for the old notebooks, no longer used as an output for new training).
- Frozen copy: `outputs/baselines/unet_10ep/best.pt` (read-only copy).
- SHA-256: `8c4495b60647e49ede8c315c6c703f5893485938a89e282ac8e9b7f618afe78d`.
- Saved epoch: 10; update count in the AdamW state: 2000.
- Saved validation mean foreground Dice: `0.8634774088859558`.
- Independently verified test mean foreground Dice: `0.8628640993226364`.
- Evaluation grid: `512 × 416`; Dice is computed per image and per foreground class, then averaged; background is excluded.
- The old checkpoint saved model + optimizer + config, but no RNG state; no training curves were found that could fully reconstruct epochs 1–10. Do not fabricate missing records.

## 2. nnU-Net v2 2D Reference

- Workspace: `data/nnunet/`, a single fold (fold 0); do not rerun or move the existing training and test predictions.
- Nominal training budget: 1000 fixed-iteration epochs, with 250 optimizer updates per epoch.
- Independently recomputed test mean foreground Dice: `0.9201659658868531`.
- Evaluation grid: the original image grid; the comparison with the hand-built model is not an architecture ablation in a common spatial domain.
- The checkpoint selection statistic, augmentation, optimizer, preprocessing, and training budget all differ.

## 3. Results of This Resumed Hand-built U-Net Run (Completed)

**A cumulative total of 50 epochs was completed on 2026-09-20. Verification covered 40 rows of actual records (epochs 11–50), 10000 updates in the final checkpoint, and epoch 23 / 4600 updates in the best checkpoint. The best model was independently loaded and reevaluated on 50 validation patients and 200 images, with results matching the training records.**

| Validation metric | Original 10-epoch baseline | Best in this run (epoch 23) |
|---|---:|---:|
| Left ventricular cavity Dice | 0.9138 | 0.9258 |
| Myocardium Dice | 0.8206 | 0.8404 |
| Left atrium Dice | 0.8560 | 0.8949 |
| Mean foreground Dice | 0.8635 | **0.8870** |

Validation mean foreground Dice increased by **2.3530 percentage points**. The training run used only the training and validation splits; its validation results must not be conflated with the separate test evaluation below.

Validation mean Dice at epoch 50 was **0.8770**, lower than at epoch 23; over the same interval, training loss fell from 0.1863 to 0.0616, while validation loss rose from 0.2625 to 0.4169. These trends are consistent with late-stage overfitting and do not support simply adding more epochs.

![Resumed training curves: validation set, not test set](assets/unet_resume10_to50_curves.png)

Version-controlled evidence: [Full metrics and provenance JSON](results/unet_resume10_to50.json) · [Per-epoch CSV](results/unet_resume10_to50_history.csv). Curves and metrics are versioned alongside the code, while weights and raw data remain in Git-ignored directories. The Git commit history identifies the published version; the code identifier in the results JSON records the workspace at training time, not a later publication commit.

### Fixed Settings and Artifacts

- Experiment directory: `outputs/runs/unet_resume10_to50_seed42/`.
- Preserved the original epoch 10 checkpoint and restored the model and AdamW; this segment ran for 40 epochs, with a cumulative target of 50 epochs.
- The model, loss, batch size 8, learning rate `1e-3`, weight decay `1e-4`, input dimensions `512×416`, and absence of augmentation/scheduler/AMP remained unchanged.
- Seed 42 applies only to resumed training because the old RNG state was missing; new checkpoints save the RNG state. No claim is made that the original 10 epochs used this seed.
- `baseline.pt`: original epoch 10; `best.pt`: validation-best epoch 23; `last.pt`: complete epoch 50 state.
- `history.csv` contains only the actually recorded epochs 11–50; the old epoch 10 validation metrics are saved and plotted separately.
- Before training, 47 tests passed and the training logic passed an independent review; the original weights' SHA256 was unchanged, and the executed code hash matched the launch record.
- Per-epoch training + validation timings totaled approximately 16.05 minutes, excluding some saving and startup overhead.

### Subsequent Test Evaluation of the Fixed Epoch-23 Checkpoint

On 2026-09-21, the already selected `best.pt` was evaluated once on the existing test split: **50 patients / 200 images**, batch size 8, input dimensions `512 × 416`. The checkpoint SHA-256 remained `c53cbf583651fca20a14e4fdcc534b499850b997e91c4796169c898cd24bd21d`, matching the earlier validation report. No weights were updated and no checkpoint was reselected from test results.

| Test metric | Original epoch 10 | Selected epoch 23 |
|---|---:|---:|
| Left ventricular cavity Dice | 0.9089 | 0.9188 |
| Myocardium Dice | 0.8199 | 0.8388 |
| Left atrium Dice | 0.8598 | 0.8755 |
| Mean foreground Dice | 0.8629 | **0.8777** |

The new test mean foreground Dice is `0.8776893615722656`, a gain of **1.4825 percentage points** over the independently verified old baseline. Test Dice + CrossEntropy loss is `0.2822867453098297`. The per-image, foreground-only resized-grid protocol is unchanged; this is an observed improvement on this test cohort, not evidence of multi-seed stability or statistical significance.

The original report is `outputs/runs/unet_resume10_to50_seed42/test.json`. The [versionable test report](results/unet_resume10_to50_test.json) preserves its metrics, hashes, counts, and protocol, changing only absolute local paths to repository-relative paths. The report also contains `checkpoint.mean_validation_dice` as selection metadata; the actual test score is `metrics.mean_foreground_dice` with `split.name` set to `test`.

## 4. Limits of Interpretation

This experiment addresses whether continuing training from the existing weights improves validation performance. It is not a strict reproduction of an uninterrupted 50-epoch run from scratch with a fixed seed, because the original RNG state is missing; nor does it establish stability across multiple seeds.

The validation-best checkpoint in this run was epoch 23, not the final epoch. Extending training initially improved validation performance, but later training brought no further improvement; a higher validation score does not guarantee a higher test score. Model and budget selection must not be driven by attempts to improve test scores.

The existing test cases have already undergone descriptive failure analysis. If these cases are used for repeated tuning in the future, that work must be explicitly described as exploratory and independently validated.

## 5. File Provenance

The old metrics come from the locally and independently reviewed `verified_metrics.json` (2026-09-19) and the frozen checkpoint. The new validation metrics come from this run's history, summary, best/last checkpoints, and independent recomputation in validation.json. The subsequent test metrics come from the explicit epoch-23 evaluation in test.json. Versionable copies are linked above in docs/results.

See [training.md](training.md) for commands and file descriptions.
