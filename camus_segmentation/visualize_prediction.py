"""Visualize one CAMUS validation prediction."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .data import CamusDataset, build_samples
from .metrics import foreground_dice_scores
from .model import UNet2D


SAMPLE_INDEX = 0
CLASS_NAMES = (
    "Background",
    "Left ventricular cavity",
    "Myocardium",
    "Left atrium",
)


@torch.no_grad()
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    camus_root = project_root / "data" / "raw" / "camus"
    validation_split = (
        project_root / "data" / "splits" / "subgroup_validation.txt"
    )
    checkpoint_path = (
        project_root / "outputs" / "checkpoints" / "best_model.pt"
    )

    validation_samples = build_samples(camus_root, validation_split)
    validation_dataset = CamusDataset(validation_samples)
    sample = validation_samples[SAMPLE_INDEX]
    image, target = validation_dataset[SAMPLE_INDEX]

    device = torch.device("cuda")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model = UNet2D().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_batch = image.unsqueeze(0).to(device)
    logits = model(image_batch)
    prediction = logits.argmax(dim=1).cpu()
    dice_scores = foreground_dice_scores(
        prediction,
        target.unsqueeze(0),
    )[0]

    output_path = (
        project_root
        / "outputs"
        / "predictions"
        / f"{sample.patient_id}_{sample.view}_{sample.phase}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(image.squeeze(0).numpy().T, cmap="gray")
    axes[0].set_title("Ultrasound image")

    axes[1].imshow(
        target.numpy().T,
        cmap="viridis",
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )
    axes[1].set_title("Ground-truth mask")

    prediction_plot = axes[2].imshow(
        prediction[0].numpy().T,
        cmap="viridis",
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )
    axes[2].set_title(
        f"Predicted mask\nMean foreground Dice: {dice_scores.mean():.3f}"
    )

    for axis in axes:
        axis.axis("off")

    colorbar = figure.colorbar(
        prediction_plot,
        ax=axes[1:],
        ticks=range(4),
        shrink=0.8,
    )
    colorbar.ax.set_yticklabels(CLASS_NAMES)
    figure.suptitle(f"{sample.patient_id} | {sample.view} | {sample.phase}")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Sample:          ", sample.patient_id, sample.view, sample.phase)
    print("Image batch:     ", tuple(image_batch.shape))
    print("Logits:          ", tuple(logits.shape))
    print("Prediction:      ", tuple(prediction.shape))
    for class_name, dice_score in zip(CLASS_NAMES[1:], dice_scores):
        print(f"  {class_name} Dice: {dice_score.item():.4f}")
    print(f"  Mean foreground Dice: {dice_scores.mean().item():.4f}")
    print("Saved figure:    ", output_path)


if __name__ == "__main__":
    main()
