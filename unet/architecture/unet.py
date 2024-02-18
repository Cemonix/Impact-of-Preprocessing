import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class DoubleConv(LightningModule):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(LightningModule):
    def __init__(self, n_channels: int, n_classes: int) -> None:
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        base = 64
        multiplier = 2
        layers = [
            base * multiplier**i for i in range(5)
        ]  # [64, 128, 256, 512, 1024]

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
                nn.ConvTranspose2d(
                    layers[i], layers[i - 1], kernel_size=2, stride=2
                )
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
        for down in self.downs[:-1]: # type: ignore
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Apply the last downs layer (the bridge)
        x = self.downs[-1](x)
        skip_connections.append(x)

        # Upsampling
        skip_connections = skip_connections[::-1]
        for up, up_conv, skip_connection in zip(
            self.ups, self.up_convs, skip_connections[1:]
        ):
            x = up(x)
            x = torch.cat([x, skip_connection], dim=1)
            x = up_conv(x)

        # Final output
        return self.outc(x)
