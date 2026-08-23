"""U-Net model for 2D CAMUS semantic segmentation."""

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Apply two 3 x 3 convolutions, each followed by ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet2D(nn.Module):
    """Four-level U-Net that returns one logit map per segmentation class."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()

        channels_1 = base_channels
        channels_2 = base_channels * 2
        channels_3 = base_channels * 4
        channels_4 = base_channels * 8
        bottleneck_channels = base_channels * 16

        self.encoder_1 = DoubleConv(in_channels, channels_1)
        self.encoder_2 = DoubleConv(channels_1, channels_2)
        self.encoder_3 = DoubleConv(channels_2, channels_3)
        self.encoder_4 = DoubleConv(channels_3, channels_4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(channels_4, bottleneck_channels)

        self.up_to_level_4 = nn.ConvTranspose2d(
            bottleneck_channels, channels_4, kernel_size=2, stride=2
        )
        self.decoder_4 = DoubleConv(channels_4 * 2, channels_4)

        self.up_to_level_3 = nn.ConvTranspose2d(
            channels_4, channels_3, kernel_size=2, stride=2
        )
        self.decoder_3 = DoubleConv(channels_3 * 2, channels_3)

        self.up_to_level_2 = nn.ConvTranspose2d(
            channels_3, channels_2, kernel_size=2, stride=2
        )
        self.decoder_2 = DoubleConv(channels_2 * 2, channels_2)

        self.up_to_level_1 = nn.ConvTranspose2d(
            channels_2, channels_1, kernel_size=2, stride=2
        )
        self.decoder_1 = DoubleConv(channels_1 * 2, channels_1)

        self.segmentation_head = nn.Conv2d(
            channels_1, num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_1 = self.encoder_1(x)
        skip_2 = self.encoder_2(self.pool(skip_1))
        skip_3 = self.encoder_3(self.pool(skip_2))
        skip_4 = self.encoder_4(self.pool(skip_3))

        x = self.bottleneck(self.pool(skip_4))

        x = self.up_to_level_4(x)
        x = torch.cat([x, skip_4], dim=1)
        x = self.decoder_4(x)

        x = self.up_to_level_3(x)
        x = torch.cat([x, skip_3], dim=1)
        x = self.decoder_3(x)

        x = self.up_to_level_2(x)
        x = torch.cat([x, skip_2], dim=1)
        x = self.decoder_2(x)

        x = self.up_to_level_1(x)
        x = torch.cat([x, skip_1], dim=1)
        x = self.decoder_1(x)

        return self.segmentation_head(x)
