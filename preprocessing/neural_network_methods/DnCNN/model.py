from typing import Any, Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchmetrics import MeanSquaredError, MetricCollection

from preprocessing.neural_network_methods.DnCNN.architecture.dncnn import (
    DnCNN,
)


class DnCNNModel(LightningModule):
    def __init__(
        self,
        n_channels: int = 1,
        learning_rate: float = 1e-4,
        metrics: MetricCollection = None,
    ) -> None:
        super(DnCNNModel, self).__init__()
        self.learning_rate = learning_rate
        self.metrics = metrics or MetricCollection({"MSE": MeanSquaredError()})
        self.save_hyperparameters(ignore=["metrics"])
        self.model = DnCNN(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        target, noised_image = batch
        denoised_images = self(noised_image)
        loss = F.mse_loss(denoised_images, target)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        target, noised_image = batch
        denoised_images = self(noised_image)

        loss = F.mse_loss(denoised_images, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        self.log_metrics(denoised_images, target)

    def log_metrics(self, denoised_images, target):
        metrics: Dict[str, Any] = self.metrics(denoised_images, target)
        for name, value in metrics.items():
            self.log(f"val_{name}", value)