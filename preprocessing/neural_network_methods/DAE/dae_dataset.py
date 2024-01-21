import os
import random
from pathlib import Path
from typing import Any, Dict

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common.image_transforms import ImageTransformHandler


class DAEDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        transform: transforms.Compose,
        noise_transform_config: Dict[str, Dict[str, Any]],
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform
        self.noise_transform_config = noise_transform_config

        self.noise_transform_handler = ImageTransformHandler()
        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("L")

        noise_types = list(self.noise_transform_config.keys())
        selected_noise_type = random.choice(noise_types)
        params = self.noise_transform_config[selected_noise_type]

        noisy_image = self.noise_transform_handler.apply_noise(
            image.copy(), selected_noise_type, params
        )

        if self.transform:
            image = self.transform(image)
            noisy_image = self.transform(noisy_image)

        return image, noisy_image
