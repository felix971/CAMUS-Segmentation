"""Run one end-to-end CAMUS training step."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CamusDataset, build_samples
from .loss import DiceCrossEntropyLoss
from .model import UNet2D


BATCH_SIZE = 8
LEARNING_RATE = 1e-3


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    camus_root = project_root / "data" / "raw" / "camus"
    training_split = project_root / "data" / "splits" / "subgroup_training.txt"

    training_samples = build_samples(camus_root, training_split)
    training_dataset = CamusDataset(training_samples)
    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    device = torch.device("cuda")
    model = UNet2D().to(device)
    loss_function = DiceCrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    model.train()
    images, masks = next(iter(training_loader))
    images = images.to(device)
    masks = masks.to(device)

    head_weights = model.segmentation_head.weight
    head_weights_before = head_weights.detach().clone()

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad()
    logits = model(images)
    loss = loss_function(logits, masks)
    loss.backward()

    head_gradient_norm = head_weights.grad.norm().item()
    optimizer.step()
    head_update_norm = (
        head_weights.detach() - head_weights_before
    ).norm().item()
    peak_memory_gib = torch.cuda.max_memory_allocated(device) / 1024**3

    print("Device:             ", torch.cuda.get_device_name(device))
    print("Training samples:   ", len(training_dataset))
    print("Image batch:        ", images.shape, images.device)
    print("Mask batch:         ", masks.shape, masks.device)
    print("Logits:             ", logits.shape, logits.device)
    print(f"Loss:                 {loss.item():.6f}")
    print(f"Head gradient norm:   {head_gradient_norm:.6e}")
    print(f"Head update norm:     {head_update_norm:.6e}")
    print(f"Peak CUDA allocation: {peak_memory_gib:.2f} GiB")


if __name__ == "__main__":
    main()
