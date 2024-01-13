import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from UNet.Network.double_conv import DoubleConv

class UNet(LightningModule):
    def __init__(self, n_channels: int, n_classes: int) -> None:
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        base = 64
        multiplier = 2
        layers = [base * multiplier ** i for i in range(5)] # [64, 128, 256, 512, 1024]

        # Downsampling layers
        self.downs = nn.ModuleList(
            [
                DoubleConv(n_channels if i == 0 else layers[i - 1], layers[i])
                for i in range(len(layers))
            ]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsampling layers
        self.ups = nn.ModuleList(
            [
                nn.ConvTranspose2d(layers[i], layers[i - 1], kernel_size=2, stride=2)
                for i in range(len(layers) - 1, 0, -1)
            ]
        )
        self.up_convs = nn.ModuleList(
            [
                DoubleConv(layers[i], layers[i - 1])
                for i in range(len(layers) - 1, 0, -1)
            ]
        )

        # Output layer
        self.outc = nn.Conv2d(base, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsampling
        skip_connections = []
        for down in self.downs[:-1]:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Apply the last downs layer (the bridge)
        x = self.downs[-1](x)
        skip_connections.append(x)

        # Upsampling
        skip_connections = skip_connections[::-1]  # Reverse for correct concatenation order
        for up, up_conv, skip_connection in zip(self.ups, self.up_convs, skip_connections[1:]):
            x = up(x)
            x = torch.cat([x, skip_connection], dim=1)
            x = up_conv(x)

        # Final output
        return self.outc(x)