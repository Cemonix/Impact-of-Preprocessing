import os
from pathlib import Path
from typing import Optional, Callable
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class LungSegmentationDataset(Dataset):
    def __init__(
        self, image_dir: Path, mask_dir: Path, transform: Optional[Callable] = None
    ) -> None:
        """
        Custom dataset for lung segmentation.

        Args:
            image_dir (Path): Path to the directory containing images.
            mask_dir (Path): Path to the directory containing corresponding masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = os.listdir(image_dir)

        # Default transformation: Convert images to tensors
        if self.transform is None:
            self.transform = transforms.ToTensor()

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
        img_path = os.path.join(self.image_dir, self.images[idx])
        if self.images[idx].split('_')[0] == 'MCUCXR':
            mask_path = os.path.join(self.mask_dir, self.images[idx])
        else:
            mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".png", "_mask.png"))

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask