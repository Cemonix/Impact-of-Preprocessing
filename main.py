from asyncio import as_completed
from typing import Optional, Any, Dict, List, Tuple, cast
import os
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from cv2 import FastFeatureDetector_NONMAX_SUPPRESSION
import numpy as np
from numpy import typing as npt
from PIL import Image
from rich.progress import track
import torch
from torch import nn
from torchmetrics import JaccardIndex, MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchvision.transforms.v2 as vision_trans
from torchvision.transforms.v2.functional import to_pil_image
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchvision.transforms.functional import pil_to_tensor
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from common.data_manipulation import (
    create_mask_from_annotation,
    load_image,
    create_noised_dataset,
)
from common.visualization import plot
from common.configs.config import load_config
from common.configs.unet_config import UNetConfig
from common.configs.preprocessing_net_config import PreprocessingNetConfig
from common.utils import (
    apply_noise_transform,
    apply_noises,
    apply_standard_preprocessing,
    choose_params_from_minmax,
    standard_preprocessing_ensemble_averaging,
    metrics_calculation,
)
from common.visualization import compare_images
from preprocessing.neural_networks.dncnn import DnCNN
from preprocessing.neural_networks.dae import DenoisingAutoencoder
from preprocessing.neural_networks.vae import VariationalAutoencoder
from preprocessing.neural_networks.data_module import PreprocessingDataModule
from preprocessing.neural_networks.model_inference import PreprocessingInference
from statistics_methods.statistics import estimate_noise_in_image
from unet.data_module import LungSegmentationDataModule, TeethSegmentationDataModule
from unet.unet import BinaryUNetModel, MulticlassUNetModel
from unet.model_inference import UnetInference


def create_dataset_main() -> None:
    # Parameters:
    # ---------------
    image_dir = Path("data/TeethSegmentation/chosen_images")
    save_dir = Path("data/TeethSegmentation/noised_images")
    walk_recursive = False
    # ---------------
    create_noised_dataset(image_dir, save_dir, walk_recursive)


def apply_model_and_create_dataset() -> None:
    # Parameters:
    # ---------------
    model_type = DenoisingAutoencoder
    image_dir = Path("data/main_dataset/final_images")
    save_dir = Path("data/main_dataset/dae_denoised_images_lungs")
    path_to_checkpoint = Path(
        "models/DAE/DAE_main_dataset/checkpoints/epoch=17-step=6336.ckpt"
    )
    file_suffix = "png"
    walk_recursive = False
    transformations = vision_trans.Compose(
        [
            vision_trans.Resize((256, 256)),
            vision_trans.ToTensor(),
        ]
    )
    # ---------------

    model = model_type.load_from_checkpoint(checkpoint_path=path_to_checkpoint)
    model.to("cpu")
    model.eval()

    images_paths: List[Path] = []
    for dirpath, _, filenames in os.walk(image_dir):
        current_dir_paths = sorted(
            [
                Path(dirpath, filename)
                for filename in filenames
                if filename.split(".")[-1] == file_suffix
            ]
        )
        images_paths.extend(current_dir_paths)
        if not walk_recursive:
            break

    for image_path in track(
        sequence=images_paths,
        description="Making predictions and moving images...",
        total=len(images_paths),
    ):
        image = load_image(image_path)
        image_tensor: torch.Tensor = transformations(image)
        prediction: torch.Tensor = model(image_tensor.unsqueeze(0))
        prediction = prediction.clamp(0, 1)
        pred_image: Image.Image = to_pil_image(prediction.squeeze(0).squeeze(0))
        pred_image.save(save_dir / image_path.name)


def __process_image(
    image_path: Path, save_dir: Path, transformations, preprocessing_config
) -> None:
    image = load_image(image_path)
    resized_image = transformations(image)
    denoised_image = standard_preprocessing_ensemble_averaging(
        np.array(resized_image), preprocessing_config, show_process=False
    )
    denoised_image.save(save_dir / image_path.name)


def apply_ensemble_and_create_dataset() -> None:
    # Parameters:
    # ---------------
    image_dir = Path("data/main_dataset/final_images")
    save_dir = Path("data/main_dataset/ensemble_denoised_images_lungs")
    preprocessing_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/standard_preprocessing_config.yaml")),
    )
    transformations = vision_trans.Compose([vision_trans.Resize((256, 256))])
    file_suffix = "png"
    walk_recursive = False
    parallel = True  # Cannot be used with Frost filter
    # ---------------
    images_paths: List[Path] = []
    for dirpath, _, filenames in os.walk(image_dir):
        current_dir_paths = sorted(
            [
                Path(dirpath, filename)
                for filename in filenames
                if filename.split(".")[-1] == file_suffix
            ]
        )
        images_paths.extend(current_dir_paths)
        if not walk_recursive:
            break

    if parallel:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    __process_image,
                    image_path,
                    save_dir,
                    transformations,
                    preprocessing_config,
                )
                for image_path in images_paths
            ]

            for future in track(
                futures,
                description="Making predictions and moving images...",
                total=len(futures),
            ):
                future.result()
    else:
        for image_path in track(
            sequence=images_paths,
            description="Making predictions and moving images...",
            total=len(images_paths),
        ):
            __process_image(image_path, save_dir, transformations, preprocessing_config)


def train_unet_model() -> None:
    # Parameters:
    # ---------------
    early_stopping = False
    metrics = None
    transformations = vision_trans.Compose(
        [
            vision_trans.RandomApply(
                [vision_trans.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2
            ),
            vision_trans.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            vision_trans.RandomAutocontrast(p=0.3),
            vision_trans.RandomEqualize(p=0.4),
            vision_trans.Resize((256, 256)),
            vision_trans.ToTensor(),
        ]
    )
    # ---------------

    unet_config: UNetConfig = cast(
        UNetConfig, load_config(Path("configs/unet_config.yaml"))
    )

    model = BinaryUNetModel(
        n_channels=unet_config.model.n_channels,
        n_classes=unet_config.model.n_classes,
        learning_rate=unet_config.model.learning_rate,
        metrics=metrics,
    )

    datamodule = LungSegmentationDataModule(
        image_dir=unet_config.dataloader.image_dir,
        mask_dir=unet_config.dataloader.masks_dir,
        batch_size=unet_config.dataloader.batch_size,
        num_of_workers=unet_config.dataloader.num_workers,
        train_ratio=unet_config.dataloader.train_ratio,
        transform=transformations,
    )

    if early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=5, verbose=False, mode="min"
        )

    trainer = Trainer(
        accelerator=unet_config.training.accelerator,
        max_epochs=unet_config.training.max_epochs,
        log_every_n_steps=unet_config.training.log_every_n_steps,
        callbacks=[early_stop_callback] if early_stopping else None,  # type: ignore
    )

    mlflow.pytorch.autolog(log_every_n_step=unet_config.training.log_every_n_steps)

    with mlflow.start_run(run_name="BinaryUNet"):
        trainer.fit(model, datamodule=datamodule)


def test_unet_model() -> None:
    # Parameters:
    # ---------------
    path_to_checkpoint = Path(
        "models/UNet/BinaryUNet/DnCNN/unet_denoised_dncnn_dataset/checkpoints/epoch=14-step=3000.ckpt"
    )
    images = load_image(
        Path("data/main_dataset/dncnn_denoised_images_lungs/CHNCXR_0086_0.png")
    )
    targets = load_image(Path("data/main_dataset/masks/CHNCXR_0086_0.png"))
    transformations = vision_trans.Compose(
        [
            vision_trans.Resize((256, 256)),
            vision_trans.ToTensor(),
        ]
    )
    is_multiclass = False
    model_type = BinaryUNetModel
    # ---------------

    if isinstance(images, Image.Image):
        images = [images]
    if isinstance(targets, Image.Image) or isinstance(targets, Path):
        targets = [targets]

    unet_inference = UnetInference(
        model_type=model_type,
        path_to_checkpoint=path_to_checkpoint,
        transformations=transformations,
    )
    unet_inference.inference_display(images, targets, multiclass=is_multiclass)


def train_multiclass_unet_model() -> None:
    # Parameters:
    # ---------------
    early_stopping = False
    loss_func = nn.CrossEntropyLoss()
    metrics = MetricCollection(
        {"jaccard_index": JaccardIndex(task="multiclass", num_classes=33)}
    )
    transformations = vision_trans.Compose(
        [
            vision_trans.RandomApply(
                [vision_trans.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2
            ),
            vision_trans.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            vision_trans.RandomAutocontrast(p=0.3),
            vision_trans.RandomEqualize(p=0.4),
            vision_trans.Resize(
                (256, 256), interpolation=vision_trans.InterpolationMode.NEAREST
            ),
            vision_trans.ToTensor(),
        ]
    )
    # ---------------

    unet_config: UNetConfig = cast(
        UNetConfig, load_config(Path("configs/unet_config.yaml"))
    )

    model = MulticlassUNetModel(
        n_channels=unet_config.model.n_channels,
        n_classes=unet_config.model.n_classes,
        learning_rate=unet_config.model.learning_rate,
        loss_func=loss_func,
        metrics=metrics,
    )

    datamodule = TeethSegmentationDataModule(
        image_dir=unet_config.dataloader.image_dir,
        annotation_dir=unet_config.dataloader.masks_dir,
        batch_size=unet_config.dataloader.batch_size,
        num_of_workers=unet_config.dataloader.num_workers,
        train_ratio=unet_config.dataloader.train_ratio,
        transform=transformations,
    )

    if early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=5, verbose=False, mode="min"
        )

    trainer = Trainer(
        accelerator=unet_config.training.accelerator,
        max_epochs=unet_config.training.max_epochs,
        log_every_n_steps=unet_config.training.log_every_n_steps,
        callbacks=[early_stop_callback] if early_stopping else None,  # type: ignore
    )

    mlflow.pytorch.autolog(log_every_n_step=unet_config.training.log_every_n_steps)

    with mlflow.start_run(run_name="MulticlassUNet"):
        trainer.fit(model, datamodule=datamodule)


def train_preprocessing_model() -> None:
    # Parameters:
    # ---------------
    architecture_type = "DenoisingAutoencoder"
    transformations = vision_trans.Compose(
        [
            vision_trans.Resize(
                (256, 256), interpolation=vision_trans.InterpolationMode.NEAREST
            ),
            vision_trans.ToTensor(),
        ]
    )
    metrics = MetricCollection(
        {
            "PSNR": PeakSignalNoiseRatio(),
            "SSIM": StructuralSimilarityIndexMeasure(),
        }
    )
    transformations = vision_trans.Compose(
        [vision_trans.Resize((256, 256)), vision_trans.ToTensor()]
    )
    augmentations = vision_trans.Compose(
        [
            vision_trans.RandomApply(
                [vision_trans.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2
            ),
            vision_trans.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            vision_trans.RandomAutocontrast(p=0.3),
            vision_trans.RandomEqualize(p=0.4),
            vision_trans.RandomHorizontalFlip(p=0.5),
            vision_trans.Resize((256, 256)),
            vision_trans.ToTensor(),
        ]
    )
    # ---------------

    preprocessing_config = cast(
        PreprocessingNetConfig,
        load_config(Path("configs/preprocessing_net_config.yaml")),
    )

    match architecture_type:
        case "DnCNN":
            model = DnCNN(
                image_channels=preprocessing_config.model.n_channels,
                learning_rate=preprocessing_config.model.learning_rate,
                metrics=metrics,
            )
        case "DenoisingAutoencoder":
            model = DenoisingAutoencoder(
                n_channels=preprocessing_config.model.n_channels,
                learning_rate=preprocessing_config.model.learning_rate,
                metrics=metrics,
            )
        case "VariationalAutoencoder":
            # TODO: get image_shape and embedding_dim
            model = VariationalAutoencoder(
                image_shape=(256, 256),
                embedding_dim=0,
                n_channels=preprocessing_config.model.n_channels,
                learning_rate=preprocessing_config.model.learning_rate,
                metrics=metrics,
            )
        case _:
            raise ValueError(f"Architecture type {architecture_type} not found")

    datamodule = PreprocessingDataModule(
        image_dir=preprocessing_config.dataloader.image_dir,
        noised_image_dir=preprocessing_config.dataloader.noised_image_dir,
        batch_size=preprocessing_config.dataloader.batch_size,
        num_of_workers=preprocessing_config.dataloader.num_workers,
        train_ratio=preprocessing_config.dataloader.train_ratio,
        transform=transformations,
        augmentations=augmentations,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=False, mode="min"
    )

    trainer = Trainer(
        accelerator=preprocessing_config.training.accelerator,
        max_epochs=preprocessing_config.training.max_epochs,
        log_every_n_steps=preprocessing_config.training.log_every_n_steps,
        callbacks=[early_stop_callback] if preprocessing_config.training.early_stopping else None,  # type: ignore
    )

    mlflow.pytorch.autolog(
        log_every_n_step=preprocessing_config.training.log_every_n_steps
    )

    with mlflow.start_run(run_name=architecture_type):
        trainer.fit(model, datamodule=datamodule)


def test_preprocessing_model() -> None:
    # Parameters:
    # ---------------
    # images = load_image(Path("data/main_dataset/original_images/CHNCXR_0005_0.png"))
    images = load_image(Path("data/main_dataset/original_images/1.jpg"))
    path_to_checkpoint = Path(
        "models/DnCNN/DnCNN_main_dataset/checkpoints/epoch=55-step=17920.ckpt"
    )
    transformations = vision_trans.Compose(
        [
            vision_trans.Resize((256, 256)),
            vision_trans.ToTensor(),
        ]
    )
    model_type = DnCNN
    model_params = {"architecture_type": model_type}
    noise_types = ["speckle_noise", "poisson_noise", "salt_and_pepper_noise"]
    # ---------------

    if isinstance(images, Image.Image):
        images = [images]

    noise_transform_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/noise_transforms_config.yaml")),
    )
    dncnn_inference = PreprocessingInference(
        model_type=model_type,
        path_to_checkpoint=path_to_checkpoint,
        transformations=transformations,
        noise_transform_config=noise_transform_config,
        noise_types=noise_types,
        metrics=None,
        **model_params,
    )
    dncnn_inference.inference_display(images)


def test_noise_transforms() -> None:
    # Parameters:
    # ---------------
    image = load_image(Path("data/main_dataset/original_images/CHNCXR_0005_0.png"))
    # ---------------

    noise_transform_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/noise_transforms_config.yaml")),
    )
    noise_types = list(noise_transform_config.keys())

    # Separate noise application
    for noise_type in noise_types:
        params = noise_transform_config[noise_type]
        selected_params = choose_params_from_minmax(params, show_chosen=True)
        noised_image = apply_noise_transform(
            np.array(image).copy(), noise_type, selected_params
        )
        compare_images(
            images=[image, noised_image],
            titles=["Original image", "Noised image"],
            images_per_column=2,
        )

    # All noise application
    noised_image = apply_noises(
        np.array(image), noise_types, noise_transform_config, True
    )

    compare_images(
        images=[image, noised_image],
        titles=["Original image", "Noised image"],
        images_per_column=2,
    )


def test_standard_preproccesing_methods() -> None:
    # Parameters:
    # ---------------
    image = load_image(Path("data/LungSegmentation/CXR_png/CHNCXR_0005_0.png"))
    noise_types = ["speckle_noise", "poisson_noise", "salt_and_pepper_noise"]
    resize = (256, 256)
    # ---------------

    noise_transform_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/noise_transforms_config.yaml")),
    )
    noised_image = apply_noises(
        np.array(image), noise_types, noise_transform_config, True
    ).resize(resize)

    preprocessing_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/standard_preprocessing_config.yaml")),
    )
    preprocessed_images = [image.resize(resize), noised_image]

    titles = [
        "Originální snímek",
        "Zašuměný snímek",
        "Odfiltrovaný snímek",
    ]
    for method, params in preprocessing_config.items():
        preprocessed_image = apply_standard_preprocessing(
            np.array(noised_image), method, params
        )
        compare_images(
            images=[*preprocessed_images, preprocessed_image],
            titles=titles,
            images_per_column=3,
        )


def test_preproccesing_ensemble_method() -> None:
    # Parameters:
    # ---------------
    image = load_image(Path("data/LungSegmentation/CXR_png/CHNCXR_0005_0.png"))
    noise_types = ["speckle_noise", "poisson_noise", "salt_and_pepper_noise"]
    resize = (256, 256)
    # ---------------

    noise_transform_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/noise_transforms_config.yaml")),
    )
    noised_image = apply_noises(
        np.array(image), noise_types, noise_transform_config, True
    ).resize(resize)

    preprocessing_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/standard_preprocessing_config.yaml")),
    )
    preprocessed_images = [image.resize(resize), noised_image]

    preprocessed_images.append(
        standard_preprocessing_ensemble_averaging(
            np.array(noised_image), preprocessing_config
        )
    )

    titles = [
        "Originální snímek",
        "Zašuměný snímek",
        "Odfiltrovaný snímek",
    ]
    compare_images(images=preprocessed_images, titles=titles, images_per_column=3)


def measure_metrics_for_images() -> None:
    # Parameters:
    # ---------------
    model_names = ["dncnn", "dae", "ensemble"]
    image_types = ["lungs", "teeth"]
    file_suffixes = ["png", "jpg"]
    predictions_path = Path("data/main_dataset/dncnn_denoised_images_lungs/")
    targets_path = Path("data/main_dataset/original_images/")
    metrics = MetricCollection(
        {
            "PSNR": PeakSignalNoiseRatio(),
            "SSIM": StructuralSimilarityIndexMeasure(),
        }
    )
    transformations = vision_trans.Compose(
        [
            vision_trans.Resize((256, 256)),
            vision_trans.ToTensor(),
        ]
    )
    # ---------------
    for model_name in model_names:
        for image_type, file_suffix in zip(image_types, file_suffixes):
            predictions_path = Path(
                f"data/main_dataset/{model_name}_denoised_images_{image_type}/"
            )
            predictions = [
                predictions_path / img_path
                for img_path in sorted(os.listdir(predictions_path))
                if img_path.split(".")[-1] == file_suffix
            ]
            targets = [
                targets_path / img_path
                for img_path in sorted(os.listdir(targets_path))
                if img_path.split(".")[-1] == file_suffix
            ]
            metrics_calculation(
                predictions,
                targets,
                metrics,
                transformations,
                f"{model_name}: {image_type}",
            )


def __calculate_metrics_ensemble_averaging(
    original_image_path: Path,
    noise_image_path: Path,
    preprocessing_config: Dict[str, Any],
    transformations,
    metrics,
) -> Tuple[Path, Dict[str, torch.Tensor]]:
    with Image.open(original_image_path) as img:
        img = img.convert("L")
    with Image.open(noise_image_path) as noised_img:
        noised_img = noised_img.convert("L")

    image_tensor: torch.Tensor = transformations(img).unsqueeze(0)
    denoised_image = standard_preprocessing_ensemble_averaging(
        np.array(noised_img), preprocessing_config, False
    )
    prediction = pil_to_tensor(denoised_image) / 255
    prediction: torch.Tensor = transformations(prediction).unsqueeze(0)
    image_tensor = image_tensor.detach()
    prediction = prediction.detach()
    return original_image_path, metrics(prediction, image_tensor)


def measure_metrics_for_denoised_images() -> None:
    # Parameters:
    # ---------------
    original_images_path = Path("data/main_dataset/original_images/")
    noised_images_path = Path("data/main_dataset/final_images/")
    noise_jsons = [
        # Path("data/main_dataset/lungs_dataset_info.json"),
        Path("data/main_dataset/teeth_dataset_info.json"),
    ]
    noise_types = [
        ["poisson_noise"],
        ["speckle_noise"],
        ["salt_and_pepper_noise"],
        ["poisson_noise", "speckle_noise"],
        ["poisson_noise", "salt_and_pepper_noise"],
        ["speckle_noise", "salt_and_pepper_noise"],
        ["poisson_noise", "speckle_noise", "salt_and_pepper_noise"],
    ]
    model_info = {
        "ensemble_averaging": "",
        DnCNN: Path(
            "models/DnCNN/DnCNN_main_dataset/checkpoints/epoch=55-step=17920.ckpt"
        ),
        DenoisingAutoencoder: Path(
            "models/DAE/DAE_main_dataset/checkpoints/epoch=17-step=6336.ckpt"
        ),
    }
    transformations = vision_trans.Compose(
        [
            vision_trans.Resize((256, 256), antialias=True),
            vision_trans.ToTensor(),
        ]
    )
    metrics = MetricCollection(
        # {
        #     "PSNR": PeakSignalNoiseRatio(),
        #     "SSIM": StructuralSimilarityIndexMeasure(),
        # }
        {
            "PSNR": PeakSignalNoiseRatio(),
            "SSIM": StructuralSimilarityIndexMeasure(),
        }
    )

    # ---------------
    def get_image_paths(
        original_images_path,
        noised_images_path: Path,
        json_files: List[Path],
        noise_types: List[str],
    ) -> Tuple[List[Path], List[Path]]:
        import ast

        original_images_paths = []
        noised_image_paths = []

        json_data = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                json_data.append(json.load(f))

        combined_json_data = {k: v for d in json_data for k, v in d.items()}

        for img_path in noised_images_path.iterdir():
            img_name = img_path.stem
            if img_name in combined_json_data:
                noise_info: str = combined_json_data[img_name]
                noise_type_str = noise_info.split("|")[0].split(":")[1].strip()
                if noise_type_str[0] == "[":
                    noise_types_in_img = ast.literal_eval(noise_type_str)
                else:
                    noise_types_in_img = [noise_type_str]

                if all(
                    noise_type in noise_types_in_img for noise_type in noise_types
                ) and len(noise_types_in_img) == len(noise_types):
                    original_images_paths.append(
                        original_images_path / (img_name + img_path.suffix)
                    )
                    noised_image_paths.append(img_path)

        return original_images_paths, noised_image_paths

    preprocessing_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/standard_preprocessing_config.yaml")),
    )
    for model_type, model_path in model_info.items():
        for noise_type_arr in noise_types:
            if not isinstance(model_type, str):
                model = model_type.load_from_checkpoint(checkpoint_path=model_path)
                model.to("cpu")
                model.eval()

            original_images_paths, noised_image_paths = get_image_paths(
                original_images_path, noised_images_path, noise_jsons, noise_type_arr
            )
            avg_metrics: Dict[str, float] = {str(key): 0.0 for key in metrics.keys()}
            img_metric_result: dict[str, dict[str, float]] = {}

            if not isinstance(model_type, str):
                for original_image_path, noise_image_path in track(
                    zip(original_images_paths, noised_image_paths),
                    f"Making predictions and calculating metrics...\nModel: {model_type}\nNoises: {noise_type_arr}\n",
                    total=len(noised_image_paths),
                ):
                    with Image.open(original_image_path) as original_img:
                        original_img = original_img.convert("L")

                    with Image.open(noise_image_path) as noised_img:
                        noised_img = noised_img.convert("L")

                    image_tensor: torch.Tensor = transformations(
                        original_img
                    ).unsqueeze(0)
                    noised_image_tensor: torch.Tensor = transformations(
                        noised_img
                    ).unsqueeze(0)
                    prediction: torch.Tensor = model(noised_image_tensor).clamp(0, 1)
                    image_tensor = image_tensor.detach()
                    prediction = prediction.detach()
                    metric_result: Dict[str, Any] = metrics(prediction, image_tensor)

                    img_metric_result[str(original_image_path)] = {}

                    for key, value in metric_result.items():
                        avg_metrics[key] += value.item()
                        img_metric_result[str(original_image_path)][key] = value.item()

            elif model_type == "ensemble_averaging":
                with ProcessPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            __calculate_metrics_ensemble_averaging,
                            original_images_path,
                            noise_image_path,
                            preprocessing_config,
                            transformations,
                            metrics,
                        )
                        for original_images_path, noise_image_path in zip(
                            original_images_paths, noised_image_paths
                        )
                    ]

                    for future in track(
                        futures,
                        f"Making predictions and calculating metrics...\nModel: {model_type}\nNoises: {noise_type_arr}",
                        total=len(futures),
                    ):
                        image_path, metric_result = future.result()
                        img_metric_result[str(image_path)] = {}
                        for key, value in metric_result.items():
                            avg_metrics[key] += value.item()
                            img_metric_result[str(image_path)][key] = value.item()

            with open(f"jsons_denoising/result_{model_type}_{noise_type_arr}.json", "w") as f:
                json.dump(img_metric_result, f)

            for key in avg_metrics:
                avg_metrics[key] /= len(noised_image_paths)
                print(f"{key}: {avg_metrics[key]:.4f}")

            print()


def measure_noise_std() -> None:
    # Parameters:
    # ---------------
    image = load_image(Path("data/main_dataset/original_images/CHNCXR_0005_0.png"))
    save_path = None
    noise_types = ["poisson_noise", "speckle_noise", "salt_and_pepper_noise"]
    noise_transform_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/noise_transforms_config.yaml")),
    )
    # ---------------
    noised_image = apply_noises(
        np.array(image), noise_types, noise_transform_config, True
    )
    eigenvalues, noise_std = estimate_noise_in_image(np.array(noised_image))

    plot(
        data=eigenvalues,
        fig_size=(10, 6),
        fontsize=25,
        marker="o",
        title="Hodnoty vlastních čísel PCA",
        xlabel="Index komponenty",
        ylabel="Vlastní číslo (směrodatná odchylka)",
        save_path=save_path,
    )
    print("Estimated noise standard deviation:", noise_std)


def images_pixel_intensities() -> None:
    # Parameters:
    # ---------------
    image_dir_path = Path("data/main_dataset/original_images/")
    images_suffixes = ["png"]
    noise_types = ["poisson_noise", "speckle_noise", "salt_and_pepper_noise"]
    noise_transform_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/noise_transforms_config.yaml")),
    )
    # ---------------
    image_paths = [
        image_dir_path / img_path
        for img_path in sorted(os.listdir(image_dir_path))
        if img_path.split(".")[-1] in images_suffixes
    ][:100]

    images: List[npt.NDArray] = [
        np.array(load_image(image_path)) for image_path in image_paths
    ]

    noise_combinations = (
        ["original"]
        + [[noise] for noise in noise_types]
        + [[noise_types[i], noise_types[i + 1]] for i in range(len(noise_types) - 1)]
        + [noise_types]
    )

    from matplotlib import pyplot as plt

    _, axes = plt.subplots(3, 3, figsize=(15, 10))
    for idx, noise_comb in enumerate(noise_combinations):
        ax = axes[idx // 3, idx % 3]
        if noise_comb == "original":
            all_pixel_intensities = np.concatenate([img.flatten() for img in images])
            ax.hist(all_pixel_intensities, bins=128, range=(0, 255), density=True)
            ax.set_title("Histogram intenzit pixelů: Originální snímky")
        elif not isinstance(noise_comb, str):
            noised_images = [
                np.array(apply_noises(image.copy(), noise_comb, noise_transform_config))
                for image in images
            ]
            all_pixel_intensities = np.concatenate(
                [img.flatten() for img in noised_images]
            )
            ax.hist(all_pixel_intensities, bins=128, range=(0, 255), density=True)
            ax.set_title(f"Histogram intenzit pixelů: {'_and_'.join(noise_comb)}")

        ax.set_xlabel("Intenzita pixelů")
        ax.set_ylabel("Frekvence")

    plt.tight_layout()
    plt.show()


def plot_mlflow_runs_metrics() -> None:
    # Parameters:
    # ---------------
    run_names = {
        "MulticlassUNet_original": "Vícetřídní U-Net - originální snímky",
        "MulticlassUNet_noised": "Vícetřídní U-Net - zašuměné snímky",
        "MulticlassUNet_ensemble": "Vícetřídní U-Net - ensemble averaging",
        "MulticlassUNet_dncnn": "Vícetřídní U-Net - DnCNN",
        "MulticlassUNet_dae": "Vícetřídní U-Net - DAE",
    }
    metric_name = "val_jaccard_index"
    title = "Metrika Jaccard index v čase učení pro vícetřídní U-Net modely"
    # ---------------
    client = MlflowClient(tracking_uri="http://localhost:5000")
    experiment = client.get_experiment_by_name("Default")
    if experiment is not None:
        runs = client.search_runs(experiment_ids=experiment.experiment_id)

        data = {name: {"steps": [], "values": []} for name in run_names.values()}
        for run in runs:
            run_name = cast(str, run.info.run_name)
            if run_name in run_names:
                run_id = run.info.run_id
                metrics = client.get_metric_history(run_id, metric_name)

                for metric in metrics:
                    data[run_names[run_name]]["steps"].append(metric.step)
                    data[run_names[run_name]]["values"].append(metric.value)

        plot(
            data=[data[key]["values"] for key in data.keys()],
            fig_size=(10, 6),
            fontsize=25,
            marker="o",
            title=title,
            xlabel="Epocha",
            ylabel="Jaccard index",
            labels=[run_name for run_name in run_names.values()],
            legend=True,
        )


def measure_metrics_for_unet_models() -> None:
    # Parameters:
    # ---------------
    original_images_path = Path("data/main_dataset/original_images/")
    target_dir = Path("data/TeethSegmentation/chosen_anns/")
    denoised_images_dir = Path("data/main_dataset/dncnn_denoised_images_teeth/")
    noise_json = Path("data/main_dataset/teeth_dataset_info.json")
    noise_types = [
        ["poisson_noise"],
        ["speckle_noise"],
        ["salt_and_pepper_noise"],
        ["poisson_noise", "speckle_noise"],
        ["poisson_noise", "salt_and_pepper_noise"],
        ["speckle_noise", "salt_and_pepper_noise"],
        ["poisson_noise", "speckle_noise", "salt_and_pepper_noise"],
    ]
    model_info = {
        "Ensemble": Path(
            "models/UNet/MultiClassUNet/Ensemble/multiclass_unet_teeth_ensemble_dataset/checkpoints/epoch=59-step=1380.ckpt"
        ),
        "DNCNN": Path(
            "models/UNet/MultiClassUNet/DnCNN/multiclass_unet_teeth_dncnn_dataset/checkpoints/epoch=59-step=1380.ckpt"
        ),
        "DAE": Path(
            "models/UNet/MultiClassUNet/DAE/multiclass_unet_teeth_dae_dataset/checkpoints/epoch=59-step=1380.ckpt"
        ),
    }
    transformations = vision_trans.Compose(
        [
            vision_trans.Resize((256, 256), antialias=True),
            vision_trans.ToTensor(),
        ]
    )
    metrics = MetricCollection(
        {
            "JaccardIndex": JaccardIndex(task="multiclass", num_classes=33),
        }
    )

    # ---------------
    def get_image_paths(
        original_images_path,
        noised_images_path: Path,
        json_files: List[Path],
        noise_types: List[str],
    ) -> Tuple[List[Path], List[Path]]:
        import ast

        original_images_paths = []
        denoised_image_paths = []

        json_data = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                json_data.append(json.load(f))

        combined_json_data = {k: v for d in json_data for k, v in d.items()}

        for img_path in noised_images_path.iterdir():
            img_name = img_path.stem
            if img_name in combined_json_data:
                noise_info: str = combined_json_data[img_name]
                noise_type_str = noise_info.split("|")[0].split(":")[1].strip()
                if noise_type_str[0] == "[":
                    noise_types_in_img = ast.literal_eval(noise_type_str)
                else:
                    noise_types_in_img = [noise_type_str]

                if all(
                    noise_type in noise_types_in_img for noise_type in noise_types
                ) and len(noise_types_in_img) == len(noise_types):
                    original_images_paths.append(
                        original_images_path / (img_name + img_path.suffix)
                    )
                    denoised_image_paths.append(img_path)

        return original_images_paths, denoised_image_paths


    for model_name, model_path in model_info.items():
        for noise_type_arr in noise_types:
            model = MulticlassUNetModel.load_from_checkpoint(checkpoint_path=model_path)
            model.to("cpu")
            model.eval()

            original_images_paths, denoised_image_paths = get_image_paths(
                original_images_path, denoised_images_dir, [noise_json], noise_type_arr
            )
            avg_metrics: Dict[str, float] = {str(key): 0.0 for key in metrics.keys()}
            img_metric_result: dict[str, dict[str, float]] = {}

            for original_image_path, denoise_image_path in track(
                zip(original_images_paths, denoised_image_paths),
                f"Making predictions and calculating metrics...\nModel: {model_name}\nNoises: {noise_type_arr}\n",
                total=len(denoised_image_paths),
            ):
                with Image.open(denoise_image_path) as denoised_img:
                    denoised_img = denoised_img.convert("L")

                mask = Image.fromarray(create_mask_from_annotation(target_dir / f"{original_image_path.stem}.jpg.json"))
                mask = mask.resize((256, 256), resample=Image.NEAREST) 
                mask = pil_to_tensor(mask)
                mask = mask.type(torch.long)

                denoised_image_tensor: torch.Tensor = transformations(denoised_img).unsqueeze(0)
                prediction: torch.Tensor = model(denoised_image_tensor) #.clamp(0, 1)
                prediction = prediction.detach()
                metric_result: Dict[str, Any] = metrics(torch.argmax(prediction, dim=1), mask)

                img_metric_result[str(original_image_path)] = {}

                for key, value in metric_result.items():
                    avg_metrics[key] += value.item()
                    img_metric_result[str(original_image_path)][key] = value.item()

            with open(f"jsons_segmentation/result_{model_name}_{noise_type_arr}.json", "w") as f:
                json.dump(img_metric_result, f)

            for key in avg_metrics:
                avg_metrics[key] /= len(denoised_image_paths)
                print(f"{key}: {avg_metrics[key]:.4f}")

            print()



def load_jsons_by_model(n=50):
    from collections import defaultdict
    metric = "JaccardIndex"
    jsons = Path("jsons_segmentation")

    model_results = defaultdict(list)

    for model_name in [
        # "ensemble_averaging",
        # "<class 'preprocessing.neural_networks.dncnn.DnCNN'>",
        # "<class 'preprocessing.neural_networks.dae.DenoisingAutoencoder'>",
        "Ensemble",
        "DNCNN",
        "DAE"
    ]:
        json_files = jsons.glob(f"result_{model_name}_*.json")
        metrics_data = {}

        for file_path in json_files:
            with open(file_path, "r") as file:
                data = json.load(file)

                # Collect the metrics for each image in the JSON
                for image_path, metrics in data.items():
                    metrics_data[image_path] = metrics[metric]

        # Sort the images by the specified metric in ascending order
        sorted_images = sorted(metrics_data.items(), key=lambda x: x[1])

        # Take the n image paths with the lowest metrics and store them
        for image_path, metric_value in sorted_images[:n]:
            model_results[image_path].append(metric_value)

    # Now, filter to find images that appear in at least two models
    common_images = {
        image: values for image, values in model_results.items() if len(values) > 1
    }

    # Print the results
    for image_path, metric_values in common_images.items():
        print(f"Path: {image_path}, {metric}: {metric_values}")


if __name__ == "__main__":
    # train_unet_model()
    # test_unet_model()
    # train_multiclass_unet_model()
    # train_preprocessing_model()
    # test_preprocessing_model()
    # apply_model_and_create_dataset()
    # apply_ensemble_and_create_dataset()
    # measure_metrics_for_images()
    # measure_noise_std()
    # plot_mlflow_runs_metrics()
    # images_pixel_intensities()

    # measure_metrics_for_denoised_images()
    # measure_metrics_for_unet_models()
    load_jsons_by_model()

    # TODO: Loss - https://github.com/francois-rozet/piqa | https://stackoverflow.com/questions/53956932/use-pytorch-ssim-loss-function-in-my-model
