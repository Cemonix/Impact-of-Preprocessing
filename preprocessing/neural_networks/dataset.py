import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PreprocessingDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        noised_image_dir: Path,
        transform: transforms.Compose,
    ) -> None:
        self.image_dir = image_dir
        self.noised_image_dir = noised_image_dir
        self.transform = transform

        self.images = os.listdir(self.image_dir)
        self.noised_images = os.listdir(self.noised_image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.image_dir / self.images[idx]
        noised_img_path = self.noised_image_dir / self.noised_images[idx]

        image = Image.open(img_path).convert("L")
        noised_image = Image.open(noised_img_path).convert("L")

        image = self.transform(image)
        noised_image = self.transform(noised_image)

        return noised_image, image
