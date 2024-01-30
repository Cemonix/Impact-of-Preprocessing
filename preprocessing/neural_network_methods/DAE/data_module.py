from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import random_split
from torchvision import transforms

from common.data_module import DataModule
from preprocessing.neural_network_methods.DAE.dataset import DAEDataset


class DAEDataModule(DataModule):
    def __init__(
        self,
        image_dir: Path,
        noise_transform_config: Dict[str, Dict[str, Any]],
        batch_size: int = 4,
        num_of_workers: int = 8,
        train_ratio: float = 0.8,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__(
            image_dir, batch_size, num_of_workers, train_ratio, transform
        )
        self.noise_transform_config = noise_transform_config

    def setup(self, stage: Optional[str] = None) -> None:
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ]
            )

        # Dataset creation
        full_dataset = DAEDataset(
            self.image_dir, self.transform, self.noise_transform_config
        )
        train_size = int(self.train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        self.train, self.test = random_split(
            full_dataset, [train_size, test_size]
        )

        if stage == "fit" or stage is None:
            # Split dataset into train and validation sets
            train_length = int(self.train_ratio * len(self.train))
            val_length = len(self.train) - train_length
            self.train, self.val = random_split(
                self.train, [train_length, val_length]
            )

        if stage == "test" or stage is None:
            # TODO: Test dataset
            pass
