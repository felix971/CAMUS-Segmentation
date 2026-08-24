"""Evaluate the best CAMUS checkpoint on the held-out test split."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CamusDataset, build_samples
from .evaluation import FOREGROUND_CLASS_NAMES, evaluate_model
from .loss import DiceCrossEntropyLoss
from .model import UNet2D


BATCH_SIZE = 8


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    camus_root = project_root / "data" / "raw" / "camus"
    test_split = project_root / "data" / "splits" / "subgroup_testing.txt"
    checkpoint_path = (
        project_root / "outputs" / "checkpoints" / "best_model.pt"
    )

    test_samples = build_samples(camus_root, test_split)
    number_of_test_patients = len(
        {sample.patient_id for sample in test_samples}
    )
    test_dataset = CamusDataset(test_samples)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    device = torch.device("cuda")
    model = UNet2D().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_function = DiceCrossEntropyLoss()

    test_loss, test_dice = evaluate_model(
        model,
        test_loader,
        loss_function,
        device,
    )

    print("Checkpoint epoch: ", checkpoint["epoch"])
    print(
        "Best validation Dice:",
        f"{checkpoint['mean_validation_dice']:.4f}",
    )
    print("Test patients:    ", number_of_test_patients)
    print("Test samples:     ", len(test_dataset))
    print("Test batches:     ", len(test_loader))
    print(f"Test loss:          {test_loss:.6f}")
    for class_name, dice_score in zip(FOREGROUND_CLASS_NAMES, test_dice):
        print(f"  {class_name} Dice: {dice_score.item():.4f}")
    print(f"  Mean foreground Dice: {test_dice.mean().item():.4f}")


if __name__ == "__main__":
    main()
