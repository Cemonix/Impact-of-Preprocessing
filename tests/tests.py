from typing import Dict, Any

import numpy as np
from numpy import typing as npt
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
    image: npt.NDArray, noised_image: npt.NDArray,
    noise_type: str, preprocessing_config: Dict[str, Dict[str, Any]]
) -> None:
    preprocess_methods = list(preprocessing_config.keys())

    preprocessed_images = [
        Image.fromarray(image),
        Image.fromarray(noised_image),
    ]

    for method in preprocess_methods:
        params = preprocessing_config[method]
        preprocessed_images.append(apply_preprocessing(image, method, params))
        
    titles = [
        "Originální snímek",
        f"Zašuměný snímek: {noise_type}",
        *preprocess_methods
    ]
    compare_images(images=preprocessed_images, titles=titles, images_per_column=3)
