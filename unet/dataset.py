import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LungSegmentationDataset(Dataset):
    def __init__(
        self, image_dir: Path, mask_dir: Path, transform: transforms.Compose
    ) -> None:
        """
        Custom dataset for lung segmentation.

        Args:
            image_dir (Path): Path to the directory containing images.
            mask_dir (Path): Path to the directory containing corresponding masks.
            transform (callable): Transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: (image, mask) where both are transformed tensors.
        """
        img_path = self.image_dir / self.images[idx]
        mask_path = self.mask_dir / self.images[idx]

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask
