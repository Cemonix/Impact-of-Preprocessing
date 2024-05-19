import os
from pathlib import Path
from typing import cast

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as vision_trans

from common.data_manipulation import create_mask_from_annotation


class LungSegmentationDataset(Dataset):
    def __init__(
        self, image_dir: Path, mask_dir: Path, transform: vision_trans.Compose
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

        self.images = os.listdir(self.image_dir)

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

        image, mask = self.transform(image, mask)
        return image, mask


class TeethSegmentationDataset(Dataset):
    def __init__(
        self, image_dir: Path, annotation_dir: Path, transform: vision_trans.Compose
    ) -> None:
        """
        Custom dataset for teeth segmentation.

        Args:
            image_dir (Path): Path to the directory containing images.
            annotation_dir (Path): Path to the directory containing corresponding annotations.
            transform (callable): Transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        self.images = sorted(os.listdir(self.image_dir))
        self.annotations = sorted(os.listdir(self.annotation_dir))

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
        annotation_path = self.annotation_dir / self.annotations[idx]

        image = Image.open(img_path).convert("L")
        mask = Image.fromarray(create_mask_from_annotation(annotation_path))

        image, mask = self.transform(image, mask)
        mask = cast(torch.Tensor, mask)
        return image, mask.squeeze(0).type(torch.long)