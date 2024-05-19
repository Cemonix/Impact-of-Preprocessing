from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import random_split
import torchvision.transforms.v2 as vision_trans

from common.data_module import DataModule
from preprocessing.neural_networks.dataset import PreprocessingDataset


class PreprocessingDataModule(DataModule):
    def __init__(
        self,
        image_dir: Path,
        noised_image_dir: Path,
        batch_size: int = 4,
        num_of_workers: int = 8,
        train_ratio: float = 0.8,
        transform: Optional[vision_trans.Compose] = None,
        augmentations: Optional[vision_trans.Compose] = None,
    ) -> None:
        super().__init__(image_dir, batch_size, num_of_workers, train_ratio, transform, augmentations)
        self.noised_image_dir = noised_image_dir

    def setup(self, stage: Optional[str] = None) -> None:
        full_dataset = PreprocessingDataset(
            image_dir=self.image_dir,
            noised_image_dir=self.noised_image_dir,
            image_transform=self.transform,
            augmentations=self.augmentations
        )
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
