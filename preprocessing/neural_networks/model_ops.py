from pathlib import Path
from typing import Any, Dict, List, cast

from PIL import Image
import mlflow.pytorch
from pytorch_lightning import Trainer
from torchmetrics import MetricCollection
from torchvision.transforms import transforms

from common.configs.config import load_config
from common.configs.preprocessing_net_config import PreprocessingNetConfig
from preprocessing.neural_networks.dae import DenoisingAutoencoder
from preprocessing.neural_networks.dncnn import DnCNN
from preprocessing.neural_networks.vae import VariationalAutoencoder
from preprocessing.neural_networks.data_module import PreprocessingDataModule
from preprocessing.neural_networks.model import PreprocessingModel
from preprocessing.neural_networks.model_inference import PreprocessingInference


def train_preprocessing_model(
    architecture_type: str, metrics: MetricCollection | None = None
) -> None:
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

    with mlflow.start_run(run_name=architecture_type.__name__):
        trainer.fit(model, datamodule=datamodule)


def test_preprocessing_model(
    images: Image.Image | List[Image.Image],
    path_to_checkpoint: Path,
    transformations: transforms.Compose | None,
    model_params: Dict[str, Any],
    noise_type: str,
) -> None:
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
