"""Evaluation metrics for CAMUS semantic segmentation."""

import torch


def foreground_dice_scores(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Return one Dice score per image and foreground class."""

    scores = []

    for class_id in range(1, num_classes):
        predicted_class = predictions == class_id
        target_class = targets == class_id

        intersection = (predicted_class & target_class).sum(dim=(1, 2))
        denominator = predicted_class.sum(dim=(1, 2)) + target_class.sum(
            dim=(1, 2)
        )
        class_scores = (
            2.0 * intersection + smooth
        ) / (denominator + smooth)
        scores.append(class_scores)

    return torch.stack(scores, dim=1)
