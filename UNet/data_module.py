from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

class LungSegmentationDataModule(LightningDataModule):
    def __init__(self, image_dir, mask_dir, batch_size=4, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            ...

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)