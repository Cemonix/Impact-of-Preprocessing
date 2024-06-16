from pathlib import Path
from typing import Any, Dict, List, Optional, cast
import numpy as np
from PIL import Image
from numpy import typing as npt
from rich.progress import track
import torch
from torchmetrics import MetricCollection
import torchvision.transforms.v2 as vision_trans


from common import noise_transforms
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
        print("No noise was added!")
        return Image.fromarray(image)


def apply_noises(
    image: npt.NDArray,
    noise_types: List[str],
    noise_transform_config: Dict[str, Dict[str, List[float]]],
    show_chosen: bool = False,
) -> Image.Image:
    noised_image = np.array(image).copy()
    for noise_type in noise_types:
        params = noise_transform_config[noise_type]
        selected_params = choose_params_from_minmax(
            params, noise_type=noise_type, show_chosen=show_chosen
        )
        noised_image = apply_noise_transform(
            np.array(noised_image), noise_type, selected_params
        )

    return cast(Image.Image, noised_image)


def choose_params_from_minmax(
    params: Dict[str, List[float]],
    noise_type: Optional[str] = None,
    show_chosen: bool = False,
) -> Dict[str, float]:
    selected_params = cast(Dict[str, float], params.copy())
    for key, value in params.items():
        chosen_value = np.random.uniform(value[0], value[1], size=None)
        if show_chosen:
            if noise_type is not None:
                print(f"Chosen value for {noise_type} with {key}: {chosen_value}")
            else:
                print(f"Chosen value for {key}: {chosen_value}")
        selected_params[key] = chosen_value

    return selected_params


def apply_standard_preprocessing(
    image: npt.NDArray, transform_type: str, params: Dict | None = None
) -> Image.Image:
    preprocessing_method = getattr(standart_preprocessing, f"apply_{transform_type}")

    if preprocessing_method and callable(preprocessing_method):
        if params is None:
            return Image.fromarray(preprocessing_method(image))
        else:
            return Image.fromarray(preprocessing_method(image, **params))
    else:
        print("No preprocessing was applied!")
        return Image.fromarray(image)


def standard_preprocessing_ensemble_averaging(
    image: npt.NDArray[np.uint8], preprocessing_config: Dict[str, Dict[str, Any]]
) -> Image.Image:
    preprocessed_images: List[npt.NDArray[np.float64]] = []

    for method, params in track(
        sequence=preprocessing_config.items(),
        description="Applying preprocessing method...",
        total=len(preprocessing_config),
    ):
        image_copy = np.array(image, copy=True)
        processed_image = apply_standard_preprocessing(image_copy, method, params)
        preprocessed_images.append(np.array(processed_image))

    averaged_image = cast(npt.NDArray[np.float64], np.mean(preprocessed_images, axis=0))
    return Image.fromarray(averaged_image.astype(np.uint8))


def metrics_calculation(
    predictions: List[Path] | Path,
    targets: List[Path] | Path,
    metrics: MetricCollection,
    transformations: vision_trans.Compose | None = None,
) -> None:
    if isinstance(predictions, Path):
        predictions = [predictions]

    if isinstance(targets, Path):
        targets = [targets]

    if len(predictions) != len(targets):
        raise Exception(f"Predictions has not same size as targets")

    if transformations is None:
        transformations = vision_trans.Compose([vision_trans.ToTensor()])

    avg_metrics: Dict[str, float] = {}
    for prediction, target in track(
        sequence=zip(predictions, targets),
        description="Measuring metrics...",
        total=len(predictions),
    ):
        with Image.open(prediction) as prediction_img:
            prediction_img = prediction_img.convert("L")
        with Image.open(target) as target_img:
            target_img = target_img.convert("L")

        predition_tensor = cast(torch.Tensor, transformations(prediction_img))
        target_tensor = cast(torch.Tensor, transformations(target_img))

        metric_result: Dict[str, Any] = metrics(predition_tensor.unsqueeze(0), target_tensor.unsqueeze(0))

        for key, value in metric_result.items():
            if key not in avg_metrics:
                avg_metrics[key] = 0.0
            avg_metrics[key] += value.item()

    for key, value in avg_metrics.items():
        print(f"{key}:{value / len(predictions)}")