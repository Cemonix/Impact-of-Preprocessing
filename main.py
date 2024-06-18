from typing import Optional, Any, Dict, List, cast
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
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
import mlflow.pytorch

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
        "models/UNet/MultiClassUNet/DnCNN/multiclass_unet_teeth_dncnn_dataset/checkpoints/epoch=59-step=1380.ckpt"
    )
    # images = load_image(Path("data/TeethSegmentation/img/10.jpg"))
    images = load_image(Path("data/main_dataset/dae_denoised_images_teeth/10.jpg"))
    targets = Path("data/TeethSegmentation/chosen_anns/10.jpg.json")

    # path_to_checkpoint = Path(
    #     "models/UNet/BinaryUNet/Noised/unet_noised_dataset/checkpoints/epoch=14-step=3000.ckpt"
    # )
    # images = load_image(Path("data/main_dataset/final_images/CHNCXR_0086_0.png"))
    # targets = load_image(Path("data/main_dataset/masks/CHNCXR_0086_0.png"))
    transformations = vision_trans.Compose(
        [
            vision_trans.Resize((256, 256)),
            vision_trans.ToTensor(),
        ]
    )
    is_multiclass = True
    model_type = MulticlassUNetModel
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
            vision_trans.Resize((256, 256), interpolation=vision_trans.InterpolationMode.NEAREST),
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
        "models/DnCNN/DNCNN_main_dataset/checkpoints/epoch=55-step=17920.ckpt"
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


def measure_noise_std() -> None:
    # Parameters:
    # ---------------
    image = load_image(Path("data/dataset/images/CHNCXR_0005_0.png"))
    save_path = None
    noise_type = "poisson_noise"
    # ---------------

    noise_transform_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/noise_transforms_config.yaml")),
    )
    params = noise_transform_config[noise_type]
    noised_image = apply_noise_transform(np.array(image).copy(), noise_type, params)

    eigenvalues, noise_std = estimate_noise_in_image(np.array(noised_image))
    plot(
        data=eigenvalues,
        fig_size=(10, 6),
        marker="o",
        title="PCA Eigenvalues of Image Patches",
        xlabel="Component Index",
        ylabel="Eigenvalue (Explained Variance)",
        save_path=save_path,
    )
    print("Estimated noise standard deviation:", noise_std)


if __name__ == "__main__":
    # train_unet_model()
    # test_unet_model()
    train_multiclass_unet_model()
    # train_preprocessing_model()
    # test_preprocessing_model()
    # apply_model_and_create_dataset()
    # apply_ensemble_and_create_dataset()
    # measure_metrics_for_images()

    # TODO: Loss - https://github.com/francois-rozet/piqa | https://stackoverflow.com/questions/53956932/use-pytorch-ssim-loss-function-in-my-model
