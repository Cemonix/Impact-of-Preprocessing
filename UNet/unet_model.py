import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch.optim import Optimizer, Adam

from UNet.Network.unet import UNet

class UNetModel(pl.LightningModule):
    def __init__(
        self, n_channels: int, n_classes: int, learning_rate: float = 1e-4
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(n_channels, n_classes)
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        images, masks = batch
        predictions = self(images)
        loss = F.binary_cross_entropy_with_logits(predictions, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        images, masks = batch
        predictions = self(images)
        loss = F.binary_cross_entropy_with_logits(predictions, masks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.model.parameters(), lr=self.learning_rate)