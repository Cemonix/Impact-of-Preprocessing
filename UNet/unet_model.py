import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim

class UNetModel(pl.LightningModule):
    def __init__(self, n_channels: int, n_classes: int, lr: float = 1e-4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.lr = lr

    def training_step(self, batch, batch_idx):
        images, masks = batch
        output = self(images)
        loss = F.binary_cross_entropy_with_logits(output, masks)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer