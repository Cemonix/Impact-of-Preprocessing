import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common.noise_transforms import NoiseTransformHandler


class DnCNNDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        transform: transforms.Compose,
        noise_transform_config: Dict[str, Dict[str, Any]],
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform
        self.noise_transform_config = noise_transform_config

        self.noise_transform_handler = NoiseTransformHandler()
        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("L")

        # noise_types = list(self.noise_transform_config.keys())
        # selected_noise_type = random.choice(noise_types)
        selected_noise_type = 'gaussian_noise'
        params = self.noise_transform_config[selected_noise_type]

        noised_image = self.noise_transform_handler.apply_noise_transform(
            np.array(image.copy()), selected_noise_type, params
        )

        if self.transform:
            image = self.transform(image)
            noised_image = self.transform(noised_image)

        return image, noised_image
