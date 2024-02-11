import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class ConvBlock(LightningModule):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DeconvBlock(LightningModule):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class ResidualBlock(LightningModule):
    def __init__(self, channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DenoisingAutoencoder(LightningModule):
    def __init__(self, n_channels: int = 1) -> None:
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(n_channels, 64, stride=2),
            ResidualBlock(64),
            ConvBlock(64, 128, stride=2),
            ResidualBlock(128),
            ConvBlock(128, 256, stride=2)
        )

        self.decoder = nn.Sequential(
            ResidualBlock(256),
            DeconvBlock(256, 128),
            ResidualBlock(128),
            DeconvBlock(128, 64),
            ResidualBlock(64),
            DeconvBlock(64, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predicted_noise = self.encoder(x)
        predicted_noise = self.decoder(predicted_noise)
        denoised_image = x - predicted_noise
        # TODO: return noise as well
        return denoised_image
