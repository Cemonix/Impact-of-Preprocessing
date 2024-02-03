from typing import Any, Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchmetrics import MeanSquaredError, MetricCollection

from preprocessing.neural_network_methods.DAE.architecture.denoising_autoencoder import (
    DenoisingAutoencoder,
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
        self.model = DenoisingAutoencoder(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        images, clean_images = batch
        denoised_images = self(images)
        loss = F.mse_loss(denoised_images, clean_images)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        images, clean_images = batch
        denoised_images = self(images)

        loss = F.mse_loss(denoised_images, clean_images)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        self.log_metrics(denoised_images, clean_images)

    def log_metrics(self, denoised_images, clean_images):
        metrics: Dict[str, Any] = self.metrics(denoised_images, clean_images)
        for name, value in metrics.items():
            self.log(f"val_{name}", value)
