from typing import Optional, Any, Dict, List, cast
import os
from pathlib import Path
import numpy as np
from numpy import typing as npt
from PIL import Image
from rich.progress import track
from torch import nn
from torchmetrics import JaccardIndex, MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image
from pytorch_lightning import Trainer
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
    model_type = DnCNN
    image_dir = Path("data/main_dataset/final_images")
    save_dir = Path("data/main_dataset/dncnn_denoised_images")
    path_to_checkpoint = Path(
        "lightning_logs/DNCNN_main_dataset/checkpoints/epoch=55-step=17920.ckpt"
    )
    walk_recursive = False
    # ---------------
    transformations = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    model = model_type.load_from_checkpoint(checkpoint_path=path_to_checkpoint)
    model.to("cpu")
    model.eval()

    images_paths: List[Path] = []
    for dirpath, _, filenames in os.walk(image_dir):
        images_paths.extend([Path(dirpath, filename) for filename in filenames])
        if not walk_recursive:
            break

    for image_path in track(
        sequence=images_paths,
        description="Making predictions and moving images...",
        total=len(images_paths),
    ):
        image = load_image(image_path)
        image_tensor = transformations(image)
        prediction = model(image_tensor.unsqueeze(0))  # type: ignore
        pred_image: Image.Image = to_pil_image(prediction.squeeze(0).squeeze(0))
        pred_image.save(save_dir / image_path.name)


def train_unet_model() -> None:
    # Parameters:
    # ---------------
    metrics = None
    transformations = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
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

    trainer = Trainer(
        accelerator=unet_config.training.accelerator,
        max_epochs=unet_config.training.max_epochs,
        log_every_n_steps=unet_config.training.log_every_n_steps,
    )

    mlflow.pytorch.autolog(log_every_n_step=unet_config.training.log_every_n_steps)

    with mlflow.start_run(run_name="BinaryUNet"):
        trainer.fit(model, datamodule=datamodule)


def test_unet_model() -> None:
    # Parameters:
    # ---------------
    path_to_checkpoint = Path(
        "lightning_logs/unet_overfitted_dncnn_dataset/checkpoints/epoch=30-step=6200.ckpt"
    )
    images = load_image(
        Path("data/main_dataset/dncnn_denoised_images/CHNCXR_0005_0.png")
    )
    targets = load_image(Path("data/main_dataset/masks/CHNCXR_0005_0.png"))
    transformations = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    # ---------------

    if isinstance(images, Image.Image):
        images = [images]
    if isinstance(targets, Image.Image):
        targets = [targets]

    unet_inference = UnetInference(
        model_type=BinaryUNetModel,
        path_to_checkpoint=path_to_checkpoint,
        transformations=transformations,
    )
    unet_inference.inference_display(images, targets)


def train_multiclass_unet_model() -> None:
    # Parameters:
    # ---------------
    loss_func = nn.CrossEntropyLoss()
    metrics = MetricCollection(
        {"jaccard_index": JaccardIndex(task="multiclass", num_classes=33)}
    )
    transformations = transforms.Compose(
        [
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
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

    trainer = Trainer(
        accelerator=unet_config.training.accelerator,
        max_epochs=unet_config.training.max_epochs,
        log_every_n_steps=unet_config.training.log_every_n_steps,
    )

    mlflow.pytorch.autolog(log_every_n_step=unet_config.training.log_every_n_steps)

    with mlflow.start_run(run_name="MulticlassUNet"):
        trainer.fit(model, datamodule=datamodule)


def train_preprocessing_model() -> None:
    # Parameters:
    # ---------------
    architecture_type = "DnCNN"
    transformations = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    metrics = MetricCollection(
        {
            "PSNR": PeakSignalNoiseRatio(),
            "SSIM": StructuralSimilarityIndexMeasure(),
        }
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
    )

    trainer = Trainer(
        accelerator=preprocessing_config.training.accelerator,
        max_epochs=preprocessing_config.training.max_epochs,
        log_every_n_steps=preprocessing_config.training.log_every_n_steps,
    )

    mlflow.pytorch.autolog(
        log_every_n_step=preprocessing_config.training.log_every_n_steps
    )

    with mlflow.start_run(run_name=architecture_type):
        trainer.fit(model, datamodule=datamodule)


def test_preprocessing_model() -> None:
    # Parameters:
    # ---------------
    images = load_image(Path("data/main_dataset/original_images/CHNCXR_0005_0.png"))
    path_to_checkpoint = Path(
        "lightning_logs/DNCNN_main_dataset/checkpoints/epoch=55-step=17920.ckpt"
    )
    transformations = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    model_params = {"architecture_type": DnCNN}
    noise_types = ["speckle_noise", "poisson_noise", "salt_and_pepper_noise"]
    # ---------------

    if isinstance(images, Image.Image):
        images = [images]

    noise_transform_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/noise_transforms_config.yaml")),
    )
    dncnn_inference = PreprocessingInference(
        model_type=DnCNN,
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
    prediction = load_image(Path("data/main_dataset/final_images/CHNCXR_0005_0.png"))
    target = load_image(Path("data/main_dataset/original_images/CHNCXR_0005_0.png"))
    metrics = MetricCollection(
        {
            "PSNR": PeakSignalNoiseRatio(),
            "SSIM": StructuralSimilarityIndexMeasure(),
        }
    )
    transformations = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    # ---------------
    metrics_calculation(prediction, target, metrics, transformations)


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
    train_multiclass_unet_model()

    # TODO: Vizualizace výsledků multiclass UNet

    # TODO: Natrénovat Multiclass UNet se vstupem z DnCNN

    # TODO: Natrénovat DAE
    # TODO: Natrénovat UNet se vstupem z DAE
    # TODO: Natrénovat Multiclass UNet se vstupem z DAE

    # TODO: Vyzkoušet všechny klasické filtrovací algoritmy a nastavit parametry
    # TODO: Vytvořit datovou sadu s využitím klasických filtrovacích technik
    # TODO: Natrénovat UNet se vstupem z klasických filtrů
    # TODO: Natrénovat Multiclass UNet se vstupem z klasických filtrů

    # TODO: Úprava obrazu pomocí ekvalizace a kontrastu
    # TODO: Augmentace dat

    # TODO: Loss - https://github.com/francois-rozet/piqa | https://stackoverflow.com/questions/53956932/use-pytorch-ssim-loss-function-in-my-model
    # TODO: Denoising autoencoder skip connections

    # TODO: Citovat datasety
    # TODO: https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images
