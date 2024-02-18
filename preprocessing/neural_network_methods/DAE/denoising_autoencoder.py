import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class ConvBlock(LightningModule):
    def __init__(
            self, in_channels: int, out_channels: int, conv_kernel: int = 3, padding: int = 1, 
            stride: int = 1, max_pool_kernel: int = 2
        ) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(max_pool_kernel)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
class DeConvBlock(LightningModule):
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, stride: int = 1
        ) -> None:
        super(DeConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class DenoisingAutoencoder(LightningModule):
    def __init__(self, n_channels: int = 1) -> None:
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels=n_channels, out_channels=16),
            ConvBlock(in_channels=16, out_channels=32),
            ConvBlock(in_channels=32, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128)
        )

        self.decoder = nn.Sequential(
            DeConvBlock(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            DeConvBlock(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            DeConvBlock(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            DeConvBlock(in_channels=16, out_channels=n_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        noise = self.encoder(x)
        noise = self.decoder(noise)
        return x - noise
