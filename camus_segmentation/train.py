"""Train the CAMUS segmentation model."""

import argparse
import csv
import math
import os
import random
import shutil
import time
from collections.abc import Callable
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CamusDataset, build_samples, read_patient_ids
from .evaluation import FOREGROUND_CLASS_NAMES, evaluate_model
from .experiment import (atomic_copy, atomic_save, capture_rng, log_stdout, restore_rng,
                         runtime_provenance, sha256, write_json)
from .loss import DiceCrossEntropyLoss
from .model import UNet2D


def parse_arguments(argv=None) -> argparse.Namespace:
    """Read training settings from the command line."""

    parser = argparse.ArgumentParser(
        description="Train the CAMUS 2D U-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="New directory for this run; never an existing directory.")
    parser.add_argument("--resume", type=Path, help="Checkpoint to continue from.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="TOTAL target epoch (not additional epochs when resuming).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Images per update: fresh default 8; saved value on resume.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="AdamW learning rate: fresh default 0.001; saved value on resume.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="AdamW weight decay: fresh default 0.0001; saved value on resume.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Fresh/legacy RNG seed (default 42); saved RNG takes precedence.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="One training update and one validation batch; not resumable.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args(argv)


def train_one_epoch(
    model: torch.nn.Module,
    training_loader: DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *, max_batches=None,
) -> float:
    """Return mean batch loss; optional batch limit is only used by smoke tests."""

    model.train()
    total_loss = 0.0
    number_of_batches = len(training_loader)
    if max_batches is not None:
        number_of_batches = min(number_of_batches, max_batches)

    for batch_index, (images, masks) in enumerate(islice(training_loader, number_of_batches), start=1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; no optimizer update performed")
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_index % 25 == 0 or batch_index == number_of_batches:
            print(
                f"  Batch {batch_index:03d}/{number_of_batches}: "
                f"loss={loss.item():.6f}"
            )

    return total_loss / number_of_batches


HISTORY_COLUMNS = (
    "epoch", "optimizer_steps", "learning_rate", "train_loss", "val_loss",
    "val_dice_lv", "val_dice_myo", "val_dice_la", "val_mean_dice", "epoch_seconds",
)


def training_config(arguments, checkpoint=None, image_size=(512, 416)):
    """Use the saved recipe on resume; defaults apply only to fresh runs."""
    required = {"epoch", "training_config", "model_state_dict", "optimizer_state_dict",
                "validation_loss", "validation_dice", "mean_validation_dice"}
    if checkpoint is not None and (not isinstance(checkpoint, dict) or not required <= checkpoint.keys()):
        raise ValueError("Missing required checkpoint fields; cannot resume")
    if checkpoint and checkpoint.get("smoke_test", False):
        raise ValueError("Cannot resume a smoke-test artifact")
    if checkpoint:
        metrics = [checkpoint["validation_loss"], checkpoint["mean_validation_dice"],
                   *checkpoint["validation_dice"], checkpoint.get("best_mean_validation_dice", 0.0)]
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError("Checkpoint metrics must be finite")
    saved_epoch = checkpoint["epoch"] if checkpoint else 0
    if arguments.epochs <= saved_epoch:
        raise ValueError(f"TOTAL target must exceed saved epoch {saved_epoch}")
    config = dict(checkpoint["training_config"]) if checkpoint else {
        "batch_size": 8, "image_size": list(image_size),
        "loss": "DiceCrossEntropyLoss", "optimizer": "AdamW",
        "learning_rate": 1e-3, "weight_decay": 1e-4,
        "betas": [0.9, 0.999], "eps": 1e-8,
    }
    recipe = {
        "model": "UNet2D(in_channels=1,num_classes=4,base_channels=32)",
        "loss": "DiceCrossEntropyLoss", "optimizer": "AdamW",
        "preprocessing": "image/255;bilinear(align_corners=False);mask-nearest;no-augmentation",
    }
    for key, value in recipe.items():
        if config.setdefault(key, value) != value:
            raise ValueError(f"Unsupported saved {key} recipe: {config[key]}")
    for key in ("batch_size", "learning_rate", "weight_decay"):
        override = getattr(arguments, key)
        if override is not None:
            if checkpoint and override != config[key]:
                raise ValueError(f"Resume conflict: {key}={override}, saved {config[key]}")
            config[key] = override
    if checkpoint and "schema_version" in checkpoint:
        if "rng_state" not in checkpoint:
            raise ValueError("New-schema checkpoint is missing rng_state")
        if arguments.seed is not None and arguments.seed != config["seed"]:
            raise ValueError("Resume seed conflict: saved RNG must be restored, not reseeded")
    config["seed"] = config.get("seed", 42) if arguments.seed is None else arguments.seed
    config["epochs"] = arguments.epochs
    return config


def optimizer_step(checkpoint, config):
    """The fixed all-parameter AdamW recipe has one common update counter."""
    saved = checkpoint["optimizer_state_dict"]
    for group in saved["param_groups"]:
        for key, setting in (("lr", "learning_rate"), ("weight_decay", "weight_decay"),
                             ("eps", "eps"), ("betas", "betas")):
            value = list(group[key]) if key == "betas" else group[key]
            if value != config[setting]:
                raise ValueError(f"Saved optimizer {key} disagrees with training_config")
    steps = {float(state["step"]) for state in saved["state"].values()}
    if len(steps) != 1 or not all(math.isfinite(step) and step >= 1 and step.is_integer() for step in steps):
        raise ValueError("Saved optimizer steps must be identical positive integers")
    step = int(steps.pop())
    if checkpoint.get("global_step", step) != step:
        raise ValueError("Saved global_step disagrees with optimizer steps")
    return step


def copy_checkpoint(source, destination, smoke_test):
    if smoke_test:
        marked = torch.load(source, map_location="cpu", weights_only=True)
        marked["smoke_test"] = True
        atomic_save(marked, destination)
    else:
        atomic_copy(source, destination)


def linked_checkpoint(source, reference, filename):
    """Prefer sibling artifacts so a complete run directory can be moved."""
    candidate = source.parent / filename
    if not candidate.is_file():
        candidate = Path(reference["path"])
    if not candidate.is_file() or sha256(candidate) != reference["sha256"]:
        raise ValueError(f"Missing or changed linked checkpoint: {filename}")
    return candidate


def run_training(arguments, training_dataset, validation_dataset, *,
                 model_factory: Callable[[], torch.nn.Module] = UNet2D, data_info: dict | None = None):
    """Run the unchanged recipe; injectable datasets/model enable tiny CPU tests."""
    if arguments.run_dir.exists() or arguments.run_dir.is_symlink():
        raise FileExistsError(f"Run directory must be new: {arguments.run_dir}")
    checkpoint = (torch.load(arguments.resume, map_location="cpu", weights_only=True)
                  if arguments.resume else None)
    config = training_config(arguments, checkpoint, training_dataset.image_size)
    for dataset in (training_dataset, validation_dataset):
        if len(dataset) == 0:
            raise ValueError("Training and validation datasets must be nonempty")
        if list(dataset.image_size) != config["image_size"]:
            raise ValueError("Dataset image_size conflicts with saved recipe")
    data_info = dict(data_info or {"splits": {}, "source": "injected datasets"})
    data_info["counts"] = {"training_samples": len(training_dataset),
                           "validation_samples": len(validation_dataset)}
    if checkpoint and "schema_version" in checkpoint:
        previous_data = checkpoint["data"]
        if previous_data["counts"] != data_info["counts"]:
            raise ValueError("Resume data counts differ from checkpoint")
        for name in ("training", "validation"):
            previous_hash = previous_data["splits"].get(name, {}).get("sha256")
            current_hash = data_info["splits"].get(name, {}).get("sha256")
            if previous_hash != current_hash:
                raise ValueError(f"Resume {name} split hash differs from checkpoint")
    global_step = optimizer_step(checkpoint, config) if checkpoint else 0
    best_epoch, best_dice = 0, -1.0
    baseline, source, best_reference = None, None, None
    best_source, baseline_source = None, None
    if checkpoint:
        source = {"path": str(arguments.resume.resolve()), "sha256": sha256(arguments.resume)}
        best_epoch = checkpoint.get("best_epoch", checkpoint["epoch"])
        best_dice = checkpoint.get("best_mean_validation_dice", checkpoint["mean_validation_dice"])
        best_source = arguments.resume
        if best_epoch != checkpoint["epoch"]:
            best_source = linked_checkpoint(arguments.resume, checkpoint["best_checkpoint"], "best.pt")
        if "schema_version" not in checkpoint:
            baseline_source = arguments.resume
            baseline = {key: checkpoint[key] for key in (
                "epoch", "validation_loss", "validation_dice", "mean_validation_dice")}
        elif checkpoint["baseline"]:
            baseline = dict(checkpoint["baseline"])
            baseline_source = linked_checkpoint(arguments.resume, baseline, "baseline.pt")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(arguments.device)
    model = model_factory().to(device)
    loss_function = DiceCrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], betas=tuple(config["betas"]),
        eps=config["eps"], weight_decay=config["weight_decay"],
    )
    start_epoch = 1
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        # Adam moments follow parameter devices; the non-capturable step counter
        # stays on CPU as required by PyTorch's standard AdamW implementation.
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    generators = {"train": torch.Generator().manual_seed(config["seed"]),
                  "validation": torch.Generator().manual_seed(config["seed"] + 1)}
    training_loader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True,
                                 generator=generators["train"])
    validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False,
                                   generator=generators["validation"])
    if checkpoint and "rng_state" in checkpoint:
        restore_rng(checkpoint["rng_state"], generators)
    run_dir = arguments.run_dir
    run_dir.mkdir(parents=True)
    if checkpoint:
        copy_checkpoint(best_source, run_dir / "best.pt", arguments.smoke_test)
        best_reference = {"path": str((run_dir / "best.pt").resolve()),
                          "sha256": sha256(run_dir / "best.pt")}
        if baseline:
            copy_checkpoint(baseline_source, run_dir / "baseline.pt", arguments.smoke_test)
            baseline.update(path=str((run_dir / "baseline.pt").resolve()),
                            sha256=sha256(run_dir / "baseline.pt"))
            write_json(run_dir / "baseline.json", baseline)
    source_history = None
    if checkpoint and "schema_version" in checkpoint:
        parent_history = arguments.resume.parent / "history.csv"
        if parent_history.is_file():
            shutil.copyfile(parent_history, run_dir / "source_history.csv")
            source_history = {"path": str(parent_history.resolve()), "sha256": sha256(parent_history),
                              "through_checkpoint_epoch": checkpoint["epoch"]}
    initialization = ("restored_checkpoint_rng" if checkpoint and "rng_state" in checkpoint
                      else "legacy_seed_no_historical_rng" if checkpoint else "fresh_seed")
    metadata = {
        "training_config": config, "smoke_test": arguments.smoke_test,
        "source": source, "source_history": source_history, "baseline": baseline,
        "rng_initialization": initialization,
        "seed_meaning": "Seed initializes fresh/legacy streams only; saved RNG is restored on later resumes. "
                        "Legacy history/RNG are unavailable: this is not an exact replay of the original run.",
        "history_segment": {"start_epoch": start_epoch, "target_epoch": arguments.epochs},
        "data": data_info,
        **runtime_provenance(Path(__file__).resolve().parents[1], device),
    }
    write_json(run_dir / "config.json", metadata)
    elapsed_seconds = checkpoint.get("elapsed_seconds", 0.0) if checkpoint else 0.0
    with log_stdout(run_dir / "train.log"), (run_dir / "history.csv").open("x", newline="") as history:
        print(f"Device: {device}; smoke_test={arguments.smoke_test}; RNG: {initialization}")
        print(f"Training samples: {len(training_dataset)}; validation samples: {len(validation_dataset)}")
        print(f"Run directory: {run_dir}; starting optimizer step: {global_step}")
        writer = csv.writer(history)
        writer.writerow(HISTORY_COLUMNS)
        stop_epoch = start_epoch if arguments.smoke_test else arguments.epochs
        state = None
        for epoch in range(start_epoch, stop_epoch + 1):
            started = time.monotonic()
            print(f"Epoch {epoch}/{arguments.epochs}")
            train_loss = train_one_epoch(model, training_loader, loss_function, optimizer, device,
                                         max_batches=1 if arguments.smoke_test else None)
            global_step += 1 if arguments.smoke_test else len(training_loader)
            validation_batches = islice(validation_loader, 1) if arguments.smoke_test else validation_loader
            val_loss, dice = evaluate_model(model, validation_batches, loss_function, device)
            if not math.isfinite(val_loss) or not torch.isfinite(dice).all():
                raise FloatingPointError("Nonfinite validation loss or Dice; checkpoint not published")
            mean_dice = dice.mean().item()
            print(f"Mean training loss: {train_loss:.6f}")
            print(f"Validation loss: {val_loss:.6f}")
            for class_name, score in zip(FOREGROUND_CLASS_NAMES, dice):
                print(f"  {class_name} Dice: {score.item():.4f}")
            print(f"  Mean foreground Dice: {mean_dice:.4f}")
            elapsed = time.monotonic() - started
            elapsed_seconds += elapsed
            improved = mean_dice > best_dice
            if improved:
                best_epoch, best_dice = epoch, mean_dice
            state = {
                "schema_version": 1, "smoke_test": arguments.smoke_test,
                "rng_state": capture_rng(generators),
                "source": source, "baseline": baseline, "data": data_info,
                "elapsed_seconds": elapsed_seconds, "epoch_seconds": elapsed,
                "best_checkpoint": None if improved else best_reference,
                "best_epoch": best_epoch, "best_mean_validation_dice": best_dice,
                "epoch": epoch, "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_loss": val_loss, "validation_dice": dice.tolist(),
                "mean_validation_dice": mean_dice, "training_config": config,
            }
            # Record measured metrics before committing a checkpoint. A failed
            # save may leave one extra row; source_history records the checkpoint
            # epoch cutoff so later runs never pretend that row was resumed.
            writer.writerow([epoch, global_step, optimizer.param_groups[0]["lr"],
                             train_loss, val_loss, *dice.tolist(), mean_dice, elapsed])
            history.flush()
            os.fsync(history.fileno())
            # Commit last first: if publishing best fails, last itself contains
            # the new best weights/optimizer/RNG (best_epoch == epoch).
            atomic_save(state, run_dir / "last.pt")
            if improved:
                atomic_save(state, run_dir / "best.pt")
                best_reference = {"path": str((run_dir / "best.pt").resolve()),
                                  "sha256": sha256(run_dir / "best.pt")}
        assert state is not None  # The total target was validated before allocation.
        summary = {key: state[key] for key in (
            "epoch", "global_step", "best_epoch", "best_mean_validation_dice", "elapsed_seconds", "smoke_test")}
        summary.update(target_epoch=arguments.epochs, completed=True,
                       best_checkpoint=str((run_dir / "best.pt").resolve()))
        write_json(run_dir / "summary.json", summary)
        print(f"Completed epoch {state['epoch']}; best epoch {best_epoch}, Dice {best_dice:.6f}")
    return state


def preflight_data(camus_root, split_root):
    """Validate patient IDs/files; testing IDs are read only for overlap checks."""
    ids, splits = {}, {}
    for name in ("training", "validation", "testing"):
        path = split_root / f"subgroup_{name}.txt"
        if name == "testing" and not path.exists():
            continue
        patients = read_patient_ids(path)
        if not patients:
            raise ValueError(f"{name} split is empty")
        if len(patients) != len(set(patients)):
            raise ValueError(f"{name} split contains duplicate patient IDs")
        for other, other_ids in ids.items():
            if set(patients) & other_ids:
                raise ValueError(f"Patient overlap between {name} and {other}")
        ids[name] = set(patients)
        splits[name] = {"path": str(path.resolve()), "sha256": sha256(path),
                        "patients": len(patients), "ids_only": name == "testing"}
    samples = {}
    for name in ("training", "validation"):
        samples[name] = build_samples(camus_root, split_root / f"subgroup_{name}.txt")
        for sample in samples[name]:
            for path in (sample.image_path, sample.mask_path):
                if not path.is_file():
                    raise ValueError(f"Missing {name} data file: {path}")
        splits[name]["samples"] = len(samples[name])
    return samples["training"], samples["validation"], {"splits": splits}


def main() -> None:
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    camus_root = project_root / "data" / "raw" / "camus"
    split_root = project_root / "data" / "splits"
    checkpoint = (torch.load(arguments.resume, map_location="cpu", weights_only=True)
                  if arguments.resume else None)
    config = training_config(arguments, checkpoint)
    del checkpoint  # Do not retain a second CPU checkpoint during the run.
    training_samples, validation_samples, data_info = preflight_data(camus_root, split_root)
    training = CamusDataset(training_samples, image_size=tuple(config["image_size"]))
    validation = CamusDataset(validation_samples, image_size=tuple(config["image_size"]))
    run_training(arguments, training, validation, data_info=data_info)


if __name__ == "__main__":
    main()
