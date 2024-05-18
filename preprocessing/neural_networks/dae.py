import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MetricCollection

from preprocessing.neural_networks.model import PreprocessingModel


class ConvBlock(LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DeConvBlock(LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        output_padding: int = 1,
    ) -> None:
        super(DeConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DenoisingAutoencoder(PreprocessingModel):
    def __init__(
        self,
        n_channels: int = 1,
        learning_rate: float = 1e-4,
        metrics: MetricCollection | None = None,
    ) -> None:
        super(DenoisingAutoencoder, self).__init__(
            learning_rate=learning_rate, metrics=metrics
        )        
        self.encoder_layers = nn.ModuleList([
            ConvBlock(n_channels, 64, 3, stride=1),
            ConvBlock(64, 64, 3, stride=2),
            ConvBlock(64, 128, 5, stride=2, padding=2),
            ConvBlock(128, 128, 3, stride=1),
            ConvBlock(128, 256, 5, stride=2, padding=2),
            ConvBlock(256, 512, 3, stride=2),
        ])
        
        self.decoder_layers = nn.ModuleList([
            DeConvBlock(512, 512, 3, stride=2),
            DeConvBlock(512, 256, 3, stride=2),
            DeConvBlock(256, 128, 5, stride=2, padding=2),
            DeConvBlock(128, 64, 3, stride=2),
            DeConvBlock(64, 64, 3, stride=1, output_padding=0),
            DeConvBlock(64, n_channels, 3, output_padding=0),
        ])
        
        self.adjust_layers = nn.ModuleList([
            nn.Conv2d(768, 512, kernel_size=1),
            nn.Conv2d(384, 256, kernel_size=1),
            nn.Conv2d(192, 128, kernel_size=1),
            nn.Conv2d(128, 64, kernel_size=1),
        ])

        # Encoder layers that will be connected to decoder -> n-1
        self.skip_connection_layer_idxs = [4, 2, 1, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        encoder_outputs = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encoder_outputs.append(x)
        
        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(x)
            if i < len(self.skip_connection_layer_idxs):
                x = torch.cat([x, encoder_outputs[self.skip_connection_layer_idxs[i]]], dim=1)  # skip connections
                x = self.adjust_layers[i](x)  # adjust channels
        
        return x
