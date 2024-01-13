import os
from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from UNet.lung_dataset import LungSegmentationDataset

class LungSegmentationDataModule(LightningDataModule):
    def __init__(
        self, image_dir: str, mask_dir: str, batch_size: int = 4,
        train_ratio: float = 0.8, transform: Optional[transforms.Compose] = None
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.transform = transform

        # Initialize placeholders for the datasets
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self) -> None:
        assert os.path.exists(self.image_dir), f"Image directory {self.image_dir} does not exist."
        assert os.path.exists(self.mask_dir), f"Mask directory {self.mask_dir} does not exist."

    def setup(self, stage: Optional[str] = None) -> None:
        # Transforms (if not already defined)
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ]
            )

        # Dataset creation logic
        full_dataset = LungSegmentationDataset(self.image_dir, self.mask_dir, self.transform)
        train_size = int(self.train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        self.train, self.test = random_split(full_dataset, [train_size, test_size])

        if stage == "fit" or stage is None:
            # Split dataset into train and validation sets
            train_length = int(self.train_ratio * len(self.train))
            val_length = len(self.train) - train_length
            self.train, self.val = random_split(self.train, [train_length, val_length])

        if stage == "test" or stage is None:
            # TODO: Test dataset
            pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size)