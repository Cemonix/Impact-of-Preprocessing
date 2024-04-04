from typing import Dict, Any

import numpy as np
from PIL import Image

from common.utils import apply_noise_transform, apply_preprocessing
from common.visualization import compare_images


def test_noise_transforms(
    image: Image.Image, noise_transform_config: Dict[str, Dict[str, Any]]
) -> None:
    noise_types = list(noise_transform_config.keys())

    for noise_type in noise_types:
        params = noise_transform_config[noise_type]
        noised_image = apply_noise_transform(np.array(image).copy(), noise_type, params)
        compare_images(
            images=[image, noised_image],
            titles=["Original image", "Noised image"],
            images_per_column=2
        )


def test_preproccesing_methods(
    image: Image.Image, preprocessing_config: Dict[str, Dict[str, Any]]
) -> None:
    preprocess_methods = list(preprocessing_config.keys())

    for method in preprocess_methods:
        params = preprocessing_config[method]
        preprocessed_image = apply_preprocessing(np.array(image).copy(), method, params)
        compare_images(
            images=[image, preprocessed_image],
            titles=["Original image", "Preprocessed image"],
            images_per_column=2
        )
