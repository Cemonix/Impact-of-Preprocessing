from pathlib import Path
from typing import Any, Dict, cast

from common.configs.config import load_config
from common.data_manipulation import load_image
from tests.tests import test_preproccesing_methods

if __name__ == "__main__":
    image_path = Path("data/LungSegmentation/CXR_png/CHNCXR_0001_0.png")
    image = load_image(image_path)
    noise_transform_config = cast(
        Dict[str, Dict[str, Any]], load_config(Path("configs/noise_transforms_config.yaml"))
    )
    # test_noise_transforms(image, noise_transform_config)

    preprocessing_config = cast(
        Dict[str, Dict[str, Any]], load_config(Path("configs/standard_preprocessing_config.yaml"))
    )
    test_preproccesing_methods(image, preprocessing_config)
