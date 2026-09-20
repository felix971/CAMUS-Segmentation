"""Visualize one CAMUS validation prediction."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .data import CamusDataset, build_samples
from .evaluate_test import checkpoint_inference_config, validate_output_path
from .metrics import foreground_dice_scores
from .model import UNet2D


SAMPLE_INDEX = 0
CLASS_NAMES = (
    "Background",
    "Left ventricular cavity",
    "Myocardium",
    "Left atrium",
)


def parse_arguments() -> argparse.Namespace:
    """Select the checkpoint, validation sample, device and figure destination."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=project_root / "outputs/checkpoints/best_model.pt",
        help="Checkpoint to visualize.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=project_root / "outputs/figures/validation_prediction.png",
        help="New destination for the prediction figure; existing files are rejected.",
    )
    parser.add_argument(
        "--sample-index", type=int, default=SAMPLE_INDEX,
        help="Zero-based index within the validation samples only.",
    )
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda",
        help="Inference device.",
    )
    arguments = parser.parse_args()
    if arguments.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
    return arguments


@torch.no_grad()
def main() -> None:
    arguments = parse_arguments()
    validate_output_path(arguments.output)
    project_root = Path(__file__).resolve().parents[1]
    camus_root = project_root / "data" / "raw" / "camus"
    validation_split = (
        project_root / "data" / "splits" / "subgroup_validation.txt"
    )
    checkpoint_path = arguments.checkpoint
    print("Checkpoint:      ", checkpoint_path.resolve())
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    image_size, _ = checkpoint_inference_config(checkpoint)
    validation_samples = build_samples(camus_root, validation_split)
    if arguments.sample_index >= len(validation_samples):
        raise ValueError(
            f"--sample-index {arguments.sample_index} is outside the validation split "
            f"({len(validation_samples)} samples)"
        )
    validation_dataset = CamusDataset(validation_samples, image_size=image_size)
    sample = validation_samples[arguments.sample_index]
    image, target = validation_dataset[arguments.sample_index]

    device = torch.device(arguments.device)
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

    output_path = arguments.output
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
    try:
        # Exclusive creation also guards against aliases created during inference.
        with output_path.open("xb") as stream:
            figure.savefig(
                stream, format=output_path.suffix.lstrip(".") or "png", dpi=150,
            )
    finally:
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
