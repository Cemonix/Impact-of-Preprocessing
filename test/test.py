from pathlib import Path

from common.configs.config import load_config
from common.noise_transforms import test_noise_transforms

if __name__ == "__main__":
    image_path = Path("data/LungSegmentation/CXR_png/CHNCXR_0001_0.png")
    noise_transform_config = load_config(
        Path("Configs/noise_transforms_config.yaml")
    )
    test_noise_transforms(image_path, noise_transform_config)
