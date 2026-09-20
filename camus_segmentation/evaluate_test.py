"""Evaluate a CAMUS checkpoint on a validation or held-out test split."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import DEFAULT_IMAGE_SIZE, CamusDataset, build_samples
from .evaluation import FOREGROUND_CLASS_NAMES, evaluate_model
from .loss import DiceCrossEntropyLoss
from .model import UNet2D


BATCH_SIZE = 8


def parse_arguments() -> argparse.Namespace:
    """Read inference settings, retaining the legacy checkpoint and test split."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=project_root / "outputs/checkpoints/best_model.pt",
        help="Checkpoint to evaluate.",
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="test",
        help="Patient split to evaluate; use validation while selecting a model.",
    )
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda",
        help="Inference device.",
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="Override checkpoint batch size (legacy fallback: 8).",
    )
    parser.add_argument(
        "--output-json", type=Path,
        help="Write metrics and provenance to a new JSON file; never overwrite.",
    )
    arguments = parser.parse_args()
    if arguments.batch_size is not None and arguments.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    return arguments


def file_sha256(path: Path) -> str:
    """Hash the exact input bytes without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_path(path: Path) -> None:
    """Reject existing outputs, including symlink and hardlink input aliases."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new path: {path}")
    for parent in path.parents:
        if parent.exists() or parent.is_symlink():
            if not parent.is_dir():
                raise NotADirectoryError(f"Output parent is not a directory: {parent}")
            break


def checkpoint_inference_config(checkpoint: dict) -> tuple[tuple[int, int], int]:
    """Validate metadata and restore geometry/batch size with legacy fallbacks."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("checkpoint epoch must be a nonnegative integer")
    config = checkpoint.get("training_config", {})
    if not isinstance(config, dict):
        raise ValueError("checkpoint training_config must be a mapping")
    image_size = config.get("image_size", DEFAULT_IMAGE_SIZE)
    if (
        not isinstance(image_size, (list, tuple))
        or len(image_size) != 2
        or any(type(size) is not int or size < 16 or size % 16 for size in image_size)
    ):
        raise ValueError(
            "training_config image_size must contain two positive integer multiples of 16"
        )
    batch_size = config.get("batch_size", BATCH_SIZE)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("training_config batch_size must be a positive integer")
    return tuple(image_size), batch_size


def main() -> None:
    arguments = parse_arguments()
    if arguments.output_json is not None:
        validate_output_path(arguments.output_json)
    project_root = Path(__file__).resolve().parents[1]
    camus_root = project_root / "data" / "raw" / "camus"
    split_name = "validation" if arguments.split == "validation" else "testing"
    split_path = project_root / "data/splits" / f"subgroup_{split_name}.txt"
    checkpoint_path = arguments.checkpoint
    print("Checkpoint:       ", checkpoint_path.resolve())
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    image_size, batch_size = checkpoint_inference_config(checkpoint)
    validation_dice = checkpoint.get("mean_validation_dice")
    if (
        not isinstance(validation_dice, (int, float))
        or isinstance(validation_dice, bool)
        or not math.isfinite(validation_dice)
    ):
        raise ValueError("checkpoint mean_validation_dice must be a finite number")
    if arguments.batch_size is not None:
        batch_size = arguments.batch_size
    checkpoint_sha256 = (
        file_sha256(checkpoint_path) if arguments.output_json is not None else None
    )
    split_sha256 = file_sha256(split_path) if arguments.output_json is not None else None
    samples = build_samples(camus_root, split_path)
    number_of_patients = len({sample.patient_id for sample in samples})
    dataset = CamusDataset(samples, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device(arguments.device)
    model = UNet2D().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_function = DiceCrossEntropyLoss()

    loss, dice = evaluate_model(model, loader, loss_function, device)

    label = arguments.split.title()
    print("Checkpoint epoch: ", checkpoint["epoch"])
    print(
        "Best validation Dice:",
        f"{checkpoint['mean_validation_dice']:.4f}",
    )
    print(f"{label} patients:    ", number_of_patients)
    print(f"{label} samples:     ", len(dataset))
    print(f"{label} batches:     ", len(loader))
    print(f"{label} loss:          {loss:.6f}")
    for class_name, dice_score in zip(FOREGROUND_CLASS_NAMES, dice):
        print(f"  {class_name} Dice: {dice_score.item():.4f}")
    print(f"  Mean foreground Dice: {dice.mean().item():.4f}")

    if arguments.output_json is not None:
        report = {
            "schema_version": 1,
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_sha256,
                "epoch": checkpoint["epoch"],
                "mean_validation_dice": checkpoint["mean_validation_dice"],
            },
            "split": {
                "name": arguments.split,
                "path": str(split_path.resolve()),
                "sha256": split_sha256,
                "patients": number_of_patients,
                "samples": len(dataset),
            },
            "inference": {
                "device": str(device), "batch_size": batch_size,
                "batches": len(loader), "image_size": list(image_size),
            },
            "metric_protocol": {
                "grid": "resized", "prediction": "argmax",
                "foreground_class_ids": [1, 2, 3], "background_excluded": True,
                "smooth": 1e-6, "empty_class_score": 1.0,
                "reduction": "per-image Dice, mean over samples, then mean over foreground classes",
                "image_interpolation": "bilinear", "image_align_corners": False,
                "mask_interpolation": "nearest", "loss": "DiceCrossEntropyLoss",
                "loss_reduction": "sample-weighted mean of batch losses",
            },
            "metrics": {
                "loss": loss,
                "foreground_dice": dict(zip(FOREGROUND_CLASS_NAMES, dice.tolist())),
                "mean_foreground_dice": dice.mean().item(),
            },
        }
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output_json.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        print("Saved JSON:       ", arguments.output_json)


if __name__ == "__main__":
    main()
