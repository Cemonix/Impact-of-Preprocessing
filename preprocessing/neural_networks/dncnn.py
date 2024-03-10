import torch
import torch.nn as nn
from torchmetrics import MetricCollection

from preprocessing.neural_networks.model import PreprocessingModel


class DnCNN(PreprocessingModel):
    def __init__(
        self,
        depth: int = 17,
        n_channels: int = 64,
        image_channels: int = 1,
        learning_rate: float = 1e-4,
        metrics: MetricCollection | None = None,
    ) -> None:
        super(DnCNN, self).__init__(learning_rate=learning_rate, metrics=metrics)
        layers = [
            nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size=3, padding=1))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predicted_noise = self.dncnn(x)
        return x - predicted_noise
