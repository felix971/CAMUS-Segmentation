"""Loss functions for CAMUS semantic segmentation."""

import torch
from torch import nn
from torch.nn import functional as F


class DiceCrossEntropyLoss(nn.Module):
    """Combine pixel-wise cross entropy with foreground soft Dice loss."""

    def __init__(self, num_classes: int = 4, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        cross_entropy_loss = self.cross_entropy(logits, targets)

        probabilities = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(
            targets,
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).to(dtype=probabilities.dtype)

        spatial_dimensions = (2, 3)
        intersection = (probabilities * targets_one_hot).sum(
            dim=spatial_dimensions
        )
        denominator = probabilities.sum(dim=spatial_dimensions) + (
            targets_one_hot.sum(dim=spatial_dimensions)
        )

        dice_per_class = (
            2.0 * intersection + self.smooth
        ) / (denominator + self.smooth)
        foreground_dice = dice_per_class[:, 1:].mean()
        dice_loss = 1.0 - foreground_dice

        return cross_entropy_loss + dice_loss
