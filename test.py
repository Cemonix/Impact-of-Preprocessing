from pathlib import Path
from typing import Any, Dict, cast

from common.configs.config import load_config
from common.noise_transforms import test_noise_transforms

if __name__ == "__main__":
    image_path = Path("data/LungSegmentation/CXR_png/CHNCXR_0001_0.png")
    noise_transform_config = cast(
        Dict[str, Dict[str, Any]], load_config(Path("configs/noise_transforms_config.yaml"))
    )
    test_noise_transforms(image_path, noise_transform_config)
