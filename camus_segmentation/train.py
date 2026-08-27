"""Train the CAMUS segmentation model."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CamusDataset, build_samples
from .evaluation import FOREGROUND_CLASS_NAMES, evaluate_model
from .loss import DiceCrossEntropyLoss
from .model import UNet2D


def parse_arguments() -> argparse.Namespace:
    """Read training settings from the command line."""

    parser = argparse.ArgumentParser(
        description="Train the CAMUS 2D U-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of complete passes through the training set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images used for one parameter update.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW decoupled weight decay.",
    )
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    training_loader: DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train on every batch once and return the mean batch loss."""

    model.train()
    total_loss = 0.0
    number_of_batches = len(training_loader)

    for batch_index, (images, masks) in enumerate(training_loader, start=1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_index % 25 == 0 or batch_index == number_of_batches:
            print(
                f"  Batch {batch_index:03d}/{number_of_batches}: "
                f"loss={loss.item():.6f}"
            )

    return total_loss / number_of_batches


def main() -> None:
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    camus_root = project_root / "data" / "raw" / "camus"
    training_split = project_root / "data" / "splits" / "subgroup_training.txt"
    validation_split = (
        project_root / "data" / "splits" / "subgroup_validation.txt"
    )
    checkpoint_path = (
        project_root / "outputs" / "checkpoints" / "best_model.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    training_samples = build_samples(camus_root, training_split)
    training_dataset = CamusDataset(training_samples)
    training_loader = DataLoader(
        training_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
    )
    validation_samples = build_samples(camus_root, validation_split)
    validation_dataset = CamusDataset(validation_samples)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda")
    model = UNet2D().to(device)
    loss_function = DiceCrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=arguments.weight_decay,
    )
    training_config = {
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "image_size": list(training_dataset.image_size),
        "loss": "DiceCrossEntropyLoss",
        "optimizer": "AdamW",
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
    }

    torch.cuda.reset_peak_memory_stats(device)

    print("Device:          ", torch.cuda.get_device_name(device))
    print("Training samples:", len(training_dataset))
    print("Training batches:", len(training_loader))
    print("Validation samples:", len(validation_dataset))
    print("Validation batches:", len(validation_loader))
    print("Epochs:          ", arguments.epochs)
    print("Batch size:      ", arguments.batch_size)
    print("Learning rate:   ", arguments.learning_rate)
    print("Weight decay:    ", arguments.weight_decay)
    print("Checkpoint:      ", checkpoint_path)

    best_mean_validation_dice = -1.0

    for epoch in range(1, arguments.epochs + 1):
        print(f"Epoch {epoch}/{arguments.epochs}")
        mean_training_loss = train_one_epoch(
            model,
            training_loader,
            loss_function,
            optimizer,
            device,
        )
        print(f"Mean training loss: {mean_training_loss:.6f}")

        validation_loss, validation_dice = evaluate_model(
            model,
            validation_loader,
            loss_function,
            device,
        )
        mean_validation_dice = validation_dice.mean().item()
        print(f"Validation loss:    {validation_loss:.6f}")
        for class_name, dice_score in zip(
            FOREGROUND_CLASS_NAMES,
            validation_dice,
        ):
            print(f"  {class_name} Dice: {dice_score.item():.4f}")
        print(f"  Mean foreground Dice: {mean_validation_dice:.4f}")

        if mean_validation_dice > best_mean_validation_dice:
            best_mean_validation_dice = mean_validation_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_dice": validation_dice.tolist(),
                    "mean_validation_dice": mean_validation_dice,
                    "training_config": training_config,
                },
                checkpoint_path,
            )
            print(f"Saved new best checkpoint: {checkpoint_path}")

    peak_memory_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"Peak CUDA allocation: {peak_memory_gib:.2f} GiB")


if __name__ == "__main__":
    main()
