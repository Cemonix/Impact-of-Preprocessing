import json
from pathlib import Path
from typing import Dict, List, cast
import os
import random
from flask.cli import F
import numpy as np
from PIL import Image
from numpy import typing as npt

from common import noise_transforms
from common.configs.config import load_config
from common.data_manipulation import load_image
from preprocessing.standard_methods import standart_preprocessing


def apply_noise_transform(
    image: npt.NDArray, transform_type: str, params: Dict | None = None
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
    image: npt.NDArray, transform_type: str, params: Dict | None = None
) -> Image.Image:
    preprocessing_method = getattr(standart_preprocessing, f"apply_{transform_type}")

    if preprocessing_method and callable(preprocessing_method):
        if params is None:
            return Image.fromarray(preprocessing_method(image))
        else:
            return Image.fromarray(preprocessing_method(image, **params))
    else:
        return Image.fromarray(image)
    

def create_dataset(
    image_dir: Path,
    save_dir: Path,
    walk_recursive: bool = False
) -> None:
    if not image_dir.is_dir():
        raise Exception(f"Given path {image_dir} is not a directory!")
    
    noise_transform_config = cast(
        Dict[str, Dict[str, List[float]]], load_config(Path("configs/noise_transforms_config.yaml"))
    )

    noise_types = list(noise_transform_config.keys())

    images_paths: List[Path] = []
    for (dirpath, _, filenames) in os.walk(image_dir):
        images_paths.extend([Path(dirpath, filename) for filename in filenames])
        if not walk_recursive:
            break

    dataset_info = {}
    random.shuffle(images_paths)
    for idx, image_path in enumerate(images_paths):
        image = load_image(image_path)
        chosen_noise_types = []
        if idx < len(images_paths) * 0.2:
            image.save(save_dir / image_path.name)
            dataset_info[image_path.stem] = "No noise applied"
            continue
        elif idx < len(images_paths) * 0.4:
            chosen_noise_types = [noise_types[0]]
        elif idx < len(images_paths) * 0.6:
            chosen_noise_types = [noise_types[1]]
        elif idx < len(images_paths) * 0.8:
            chosen_noise_types = [noise_types[2]]
        else:
            chosen_noise_types = noise_types
        
        noised_image = np.array(image).copy()
        for noise_type in chosen_noise_types:
            params = noise_transform_config[noise_type]
            selected_params = cast(Dict[str, float], params.copy())
            for key, value in params.items():
                selected_params[key] = np.random.uniform(value[0], value[1], size=None)
            noised_image = apply_noise_transform(np.array(noised_image), noise_type, selected_params)

        noised_image = cast(Image.Image, noised_image)
        noised_image.save(save_dir / image_path.name)
        info = (
            f"Noise type: {chosen_noise_types[0]} | {selected_params}" 
            if len(chosen_noise_types) == 1 
            else f"Noise type: {chosen_noise_types} | {selected_params}"
        )

        dataset_info[image_path.stem] = info

    with open(save_dir / "dataset_info.json", "w") as json_file:
        json.dump({k: dataset_info[k] for k in sorted(dataset_info.keys())}, json_file)