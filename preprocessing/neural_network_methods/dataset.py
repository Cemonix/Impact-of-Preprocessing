import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common.utils import get_random_from_min_max_dict, apply_noise_transform


class PreprocessingDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        transform: transforms.Compose,
        noise_transform_config: Dict[str, Dict[str, Any]],
        noise_types: List[str]
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform
        self.noise_transform_config = noise_transform_config
        self.noise_types = noise_types

        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("L")

        selected_noise_type = random.choice(self.noise_types) if len(self.noise_types) > 1 else self.noise_types[0]
        params = self.noise_transform_config[selected_noise_type]
        params = get_random_from_min_max_dict(params)

        noised_image = apply_noise_transform(np.array(image.copy()), selected_noise_type, params)

        image = self.transform(image)
        noised_image = self.transform(noised_image)

        return noised_image, image
