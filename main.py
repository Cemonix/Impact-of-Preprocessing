from typing import Optional, Any, Dict, cast
from pathlib import Path
import numpy as np
from numpy import typing as npt
from PIL import Image
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import transforms
from pytorch_lightning import Trainer
import mlflow.pytorch

from common.data_manipulation import load_image
from common.visualization import plot
from common.configs.config import load_config
from common.configs.unet_config import UnetConfig
from common.configs.preprocessing_net_config import PreprocessingNetConfig
from common.utils import apply_noise_transform, apply_preprocessing, create_dataset
from common.visualization import compare_images
from preprocessing.neural_networks.dncnn import DnCNN
from preprocessing.neural_networks.dae import DenoisingAutoencoder
from preprocessing.neural_networks.vae import VariationalAutoencoder
from preprocessing.neural_networks.data_module import PreprocessingDataModule
from preprocessing.neural_networks.model import PreprocessingModel
from preprocessing.neural_networks.model_inference import PreprocessingInference
from statistics_methods.statistics import estimate_noise_in_image
from unet.data_module import LungSegmentationDataModule
from unet.unet import UNetModel
from unet.model_inference import UnetInference


def create_dataset_main() -> None:
    # Parameters:
    #---------------
    image_dir = Path("data/test/from")
    save_dir = Path("data/test/to")
    walk_recursive = False
    #---------------
    create_dataset(image_dir, save_dir, walk_recursive)


def train_unet_model() -> None:
    # Parameters:
    #---------------
    metrics = None
    #---------------
    
    unet_config: UnetConfig = cast(UnetConfig, load_config(Path("configs/unet_config.yaml")))

    model = UNetModel(
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
    )

    trainer = Trainer(
        accelerator=unet_config.training.accelerator,
        max_epochs=unet_config.training.max_epochs,
        log_every_n_steps=unet_config.training.log_every_n_steps,
    )

    mlflow.pytorch.autolog(
        log_every_n_step=unet_config.training.log_every_n_steps
    )

    with mlflow.start_run(run_name="UNet"):
        trainer.fit(model, datamodule=datamodule)


def test_unet_model() -> None:
    # Parameters:
    #---------------
    path_to_checkpoint = Path(
        "lightning_logs/unet_model_v0/checkpoints/epoch=99-step=3200.ckpt"
    )
    images = load_image(Path("data/dataset/images/CHNCXR_0005_0.png"))
    targets = load_image(Path("data/dataset/masks/CHNCXR_0005_0_mask.png"))
    transformations = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    #---------------

    if isinstance(images, Image.Image):
        images = [images]
    if isinstance(targets, Image.Image):
        targets = [targets]

    unet_inference = UnetInference(
        model_type=UNetModel, path_to_checkpoint=path_to_checkpoint, transformations=transformations
    )
    unet_inference.inference_display(images, targets)


def train_preprocessing_model() -> None:
    # Parameters:
    #---------------
    architecture_type = "DenoisingAutoencoder"
    metrics = MetricCollection(
        {
            "PSNR": PeakSignalNoiseRatio(),
            "SSIM": StructuralSimilarityIndexMeasure(),
        }
    )
    #---------------

    preprocessing_config = cast(
        PreprocessingNetConfig,
        load_config(Path("configs/preprocessing_net_config.yaml")),
    )
    noise_transform_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/noise_transforms_config.yaml")),
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
        noise_transform_config=noise_transform_config,
        noise_types=preprocessing_config.dataloader.noise_types,
        batch_size=preprocessing_config.dataloader.batch_size,
        num_of_workers=preprocessing_config.dataloader.num_workers,
        train_ratio=preprocessing_config.dataloader.train_ratio,
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
    #---------------
    images = load_image(Path("data/dataset/images/CHNCXR_0005_0.png"))
    path_to_checkpoint = Path(
        "lightning_logs/dncnn_model_v0_512/checkpoints/epoch=49-step=750.ckpt"
    )
    transformations = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    model_params = model_params = {"architecture_type": DnCNN}
    noise_type = "poisson_noise"
    #---------------

    if isinstance(images, Image.Image):
        images = [images]

    noise_transform_config = cast(
        Dict[str, Dict[str, Any]],
        load_config(Path("configs/noise_transforms_config.yaml")),
    )
    dncnn_inference = PreprocessingInference(
        model_type=PreprocessingModel,
        path_to_checkpoint=path_to_checkpoint,
        transformations=transformations,
        noise_transform_config=noise_transform_config,
        noise_type=noise_type,
        **model_params
    )
    dncnn_inference.inference_display(images)


def test_noise_transforms() -> None:
    # Parameters:
    #---------------
    image = load_image(Path("data/LungSegmentation/CXR_png/CHNCXR_0001_0.png"))
    #---------------

    noise_transform_config = cast(
        Dict[str, Dict[str, Any]], load_config(Path("configs/noise_transforms_config.yaml"))
    )
    noise_types = list(noise_transform_config.keys())

    for noise_type in noise_types:
        params = noise_transform_config[noise_type]
        noised_image = apply_noise_transform(np.array(image).copy(), noise_type, params)
        compare_images(
            images=[image, noised_image],
            titles=["Original image", "Noised image"],
            images_per_column=2
        )


def test_preproccesing_methods() -> None:
    # Parameters:
    #---------------
    image = np.array(load_image(Path("data/dataset/images/CHNCXR_0005_0.png")))
    noise_type = "poisson_noise"
    #---------------

    noise_transform_config = cast(
        Dict[str, Dict[str, Any]], load_config(Path("configs/noise_transforms_config.yaml"))
    )
    params = noise_transform_config[noise_type]
    noised_image = apply_noise_transform(np.array(image).copy(), noise_type, params)

    preprocessing_config = cast(
        Dict[str, Dict[str, Any]], load_config(Path("configs/standard_preprocessing_config.yaml"))
    )
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


def statistics() -> None:
    # Parameters:
    #---------------
    image = load_image(Path("data/dataset/images/CHNCXR_0005_0.png"))
    save_path = None
    noise_type = "poisson_noise"
    #---------------

    noise_transform_config = cast(
        Dict[str, Dict[str, Any]], load_config(Path("configs/noise_transforms_config.yaml"))
    )
    params = noise_transform_config[noise_type]
    noised_image = apply_noise_transform(np.array(image).copy(), noise_type, params)

    eigenvalues, noise_std = estimate_noise_in_image(np.array(noised_image))
    plot(
        data=eigenvalues, fig_size=(10, 6), marker='o', title="PCA Eigenvalues of Image Patches", 
        xlabel="Component Index", ylabel="Eigenvalue (Explained Variance)", save_path=save_path
    )
    print("Estimated noise standard deviation:", noise_std)


if __name__ == "__main__":
    create_dataset_main()


    # TODO: Denoising autoencoder skip connections
    # TODO: Augmentace dat
    # TODO: VAE

    # TODO: Citovat dataset
