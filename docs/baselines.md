# Completed Baselines: Methods, Configurations, and Results

These are historical results from the original **10-epoch** hand-built U-Net and the single-fold nnU-Net baseline, not results from the new continuation experiment.

## Results

Both pipelines use the same patient-level split: 400 patients for training, 50
for validation, and 50 held out for testing. Checkpoints are selected using the
validation split and evaluated on the 200 test images without further model
selection.

### Held-out test comparison

| Metric | Hand-built PyTorch U-Net | nnU-Net v2 2D | Pipeline gap |
|---|---:|---:|---:|
| Left ventricular cavity Dice | 0.9089 | **0.9446** | +0.0357 |
| Myocardium Dice | 0.8199 | **0.8920** | +0.0721 |
| Left atrium Dice | 0.8598 | **0.9239** | +0.0641 |
| Mean foreground Dice | 0.8629 | **0.9202** | **+0.0573** |

Dice is calculated separately for every image and foreground class, averaged
over the 200 test images, and then averaged over the three foreground classes.
Background is excluded. The nnU-Net pipeline is 5.73 percentage points higher
in mean foreground Dice under the two pipelines' respective training recipes.

### Hand-built U-Net validation and test metrics

| Metric | Validation | Test |
|---|---:|---:|
| Dice + CrossEntropy loss | 0.309221 | 0.308204 |
| Left ventricular cavity Dice | 0.9138 | 0.9089 |
| Myocardium Dice | 0.8206 | 0.8199 |
| Left atrium Dice | 0.8560 | 0.8598 |
| Mean foreground Dice | 0.8635 | **0.8629** |

### Comparison scope

This is a comparison of two complete training pipelines, not a controlled
architecture ablation. The hand-built U-Net trains for 10 full-data epochs
(2,000 optimizer updates) with AdamW and no augmentation. The standard
`nnUNetTrainer` runs 1,000 fixed-length epochs of 250 sampled-patch iterations
(250,000 optimizer updates) with SGD, polynomial learning-rate decay, built-in
augmentation, foreground oversampling, and deep supervision. The result gap
must therefore not be attributed to architecture alone.

The aggregation rule is shared, but the pixel grids differ: the hand-built
pipeline evaluates masks resized to `512 x 416`, while nnU-Net restores each
prediction to the original image geometry before evaluation.

### Hand-built U-Net qualitative validation example

![Ultrasound image, ground-truth mask, and predicted mask](assets/validation_prediction.png)

The example shows `patient0200`, the two-chamber view at end-diastole. Its mean
foreground Dice is 0.8881. The figure is a qualitative example; the table above
contains the full validation and test results.

## Dataset and task

CAMUS contains 500 patients. This project uses four annotated frames per
patient: two-chamber and four-chamber views at end-diastole and end-systole.

| Split | Patients | Annotated images |
|---|---:|---:|
| Training | 400 | 1,600 |
| Validation | 50 | 200 |
| Test | 50 | 200 |

The split is patient-level, so no patient appears in more than one subset. The
four segmentation labels are:

| Label | Structure |
|---:|---|
| 0 | Background |
| 1 | Left ventricular cavity |
| 2 | Myocardium |
| 3 | Left atrium |

The CAMUS data is not redistributed by this repository. Access instructions are
available from the
[official CAMUS dataset page](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html).
After accepting the dataset terms, place the patient directories under:

```text
data/raw/camus/patient0001/
...
data/raw/camus/patient0500/
```

## Hand-built model

The model is a four-level 2D U-Net implemented directly with PyTorch:

```text
Input: 1 x 512 x 416
Encoder channels: 32 -> 64 -> 128 -> 256
Bottleneck channels: 512
Decoder channels: 256 -> 128 -> 64 -> 32
Output: 4 x 512 x 416 logits
```

Each encoder and decoder block contains two `3 x 3` convolutions, each followed
by ReLU. Four `2 x 2` max-pooling operations reduce spatial resolution, and
transposed convolutions restore it. Skip connections concatenate encoder
features with decoder features at matching resolutions. A final `1 x 1`
convolution produces one logit map per class.

The model has **7,759,620 trainable parameters**.

## Training configurations

### Hand-built PyTorch U-Net

| Setting | Value |
|---|---|
| Input size | `512 x 416` |
| Image resize | Bilinear interpolation |
| Mask resize | Nearest-neighbor interpolation |
| Image normalization | Divide intensity values by 255 |
| Data augmentation | None for this baseline |
| Loss | CrossEntropy + foreground Soft Dice loss |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Batch size | 8 |
| Epochs | 10 |
| Checkpoint selection | Highest validation mean foreground Dice |
| Training GPU | NVIDIA GeForce RTX 4090 |

The best checkpoint is written to
`outputs/checkpoints/best_model.pt`. Generated checkpoints remain local and are
excluded from Git. Each checkpoint also records the training configuration used
to create it.

### nnU-Net v2 2D baseline

| Setting | Value |
|---|---|
| Dataset | `Dataset501_CAMUS` |
| Configuration | `2d`, fold 0 |
| Trainer and plans | `nnUNetTrainer`, `nnUNetPlans` |
| Target spacing | `0.308 x 0.308` |
| Patch size | `512 x 640` |
| Batch size | 10 |
| Normalization | Z-score |
| Architecture | Eight-stage `PlainConvUNet` |
| Optimizer | SGD, momentum 0.99, Nesterov |
| Learning rate | `0.01` with polynomial decay |
| Training length | 1,000 epochs, 250 patch iterations per epoch |
| Checkpoint used for test inference | `checkpoint_best.pth` |
| Training GPU | NVIDIA GeForce RTX 4090 |

nnU-Net fingerprints the dataset, chooses the preprocessing and network plan,
resamples and normalizes the cases, trains fold 0, and restores test predictions
to their original image geometry before export.


## Current limitations

- Results come from one fixed patient split and one training run per pipeline.
- The hand-built baseline has no data augmentation or learning-rate schedule.
- The pipeline comparison uses different architectures, preprocessing,
  optimizers, augmentation, and training budgets; it is not a controlled
  architecture ablation.
- The nnU-Net experiment uses the nnU-Net 2.8.1 default `nnUNetPlans` planner,
  rather than the newer residual-encoder presets.
- Exact deterministic seeds and repeated-run variance are not yet reported.
- Neither pipeline has been evaluated on an external dataset.
- The results do not establish clinical validity or state-of-the-art
  performance.

## Dataset citation and terms

Any use of the CAMUS database must cite:

> S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al. “Deep Learning for
> Segmentation using an Open Large-Scale Dataset in 2D Echocardiography.” IEEE
> Transactions on Medical Imaging, 38(9), 2198–2210, 2019.
> https://doi.org/10.1109/TMI.2019.2900516

See [CAMUS_LICENSE.md](../CAMUS_LICENSE.md) for the local copy of the dataset
terms. The dataset is subject to CC BY-NC-SA 4.0 and additional CAMUS terms.
