import random
from typing import Any, Dict

from PIL import Image
from numpy import typing as npt

from common import noise_transforms
from preprocessing.standard_methods import standart_preprocessing


def get_random_from_min_max_dict(
    min_max_dict: Dict[str, Dict[str, int | float] | Any]
) -> Dict[str, int | float]:
    random_dict = {}
    for key, value in min_max_dict.items():
        if isinstance(value, Dict):
            random_dict[key] = random.uniform(value['min'], value['max'])
        else:
            random_dict[key] = value
    return random_dict


def apply_noise_transform(
    image: npt.NDArray, transform_type: str, params: dict | None = None
) -> Image.Image:
    transform_method = getattr(noise_transforms, f"add_{transform_type}")

    if transform_method and callable(transform_method):
        if params is None:
            return Image.fromarray(transform_method(image))
        else:
            return Image.fromarray(transform_method(image, **params))
    else:
        return Image.fromarray(image)


def apply_preprocessing(
    image: npt.NDArray, transform_type: str, params: dict | None = None
) -> Image.Image:
    preprocessing_method = getattr(standart_preprocessing, f"apply_{transform_type}")

    if preprocessing_method and callable(preprocessing_method):
        if params is None:
            return Image.fromarray(preprocessing_method(image))
        else:
            return Image.fromarray(preprocessing_method(image, **params))
    else:
        return Image.fromarray(image)
