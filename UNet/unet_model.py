import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from torchmetrics import MetricCollection, JaccardIndex

from UNet.Network.unet import UNet

class UNetModel(pl.LightningModule):
    def __init__(
        self, n_channels: int, n_classes: int, metrics: MetricCollection = None,
        learning_rate: float = 1e-4
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(n_channels, n_classes)
        self.learning_rate = learning_rate
        self.metrics = metrics or MetricCollection(
            {'jaccard_index': JaccardIndex(task="binary", num_classes=n_classes)}
        )

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.model.parameters(), lr=self.learning_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, masks = batch
        predictions = self(images)
        loss = F.binary_cross_entropy_with_logits(predictions, masks)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        images, masks = batch
        predictions = self(images)
        loss = F.binary_cross_entropy_with_logits(predictions, masks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        binary_masks = (masks > 0).float()
        self.metrics.update(torch.sigmoid(predictions), binary_masks)

    def on_validation_epoch_end(self):
        # Log validation metrics
        metrics = self.metrics.compute()
        for name, value in metrics.items():
            self.log(f'val_{name}', value)

        self.metrics.reset()