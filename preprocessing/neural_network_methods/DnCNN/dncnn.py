import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class DnCNN(LightningModule):
    def __init__(self, depth: int = 17, n_channels: int = 64, image_channels: int = 1):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)]

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size=3, padding=1))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dncnn(x)
        return x - out
    