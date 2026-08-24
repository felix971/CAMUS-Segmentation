"""Shared evaluation loop for CAMUS segmentation."""

import torch
from torch.utils.data import DataLoader

from .metrics import foreground_dice_scores


FOREGROUND_CLASS_NAMES = (
    "Left ventricular cavity",
    "Myocardium",
    "Left atrium",
)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    """Return mean loss and per-class foreground Dice."""

    model.eval()
    total_loss = 0.0
    dice_sum = torch.zeros(3, device=device)
    number_of_samples = 0

    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_function(logits, masks)
        predictions = logits.argmax(dim=1)
        batch_dice = foreground_dice_scores(predictions, masks)

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        dice_sum += batch_dice.sum(dim=0)
        number_of_samples += batch_size

    mean_loss = total_loss / number_of_samples
    mean_dice = dice_sum / number_of_samples
    return mean_loss, mean_dice.cpu()
