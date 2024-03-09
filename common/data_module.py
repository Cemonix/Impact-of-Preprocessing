from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms


class DataModule(ABC, LightningDataModule):
    def __init__(
        self,
        image_dir: Path,
        batch_size: int = 4,
        num_of_workers: int = 8,
        train_ratio: float = 0.8,
        transform: Optional[transforms.Compose] | None = None,
    ) -> None:
        super().__init__()
        assert image_dir.exists() and image_dir.is_dir(), "Image directory does not exist"
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_of_workers = num_of_workers
        self.train_ratio = train_ratio
        self.transform = transform if transform is not None else transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

        self.train = None
        self.val = None
        self.test = None
        
    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Abstract method for setting up datasets for training, validation, and testing.
        """
        pass

    def train_dataloader(self) -> DataLoader:
        if self.train is None:
            raise Exception("Setup has to be called before creating dataloader.")
        
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val is None:
            raise Exception("Setup has to be called before creating dataloader.")
        
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test is None:
            raise Exception("Setup has to be called before creating dataloader.")
        
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
        )
