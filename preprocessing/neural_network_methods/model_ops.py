from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from pytorch_lightning import Trainer, LightningModule
from torchmetrics import MetricCollection
from torchvision.transforms import transforms

from common.configs.config import load_config
from common.configs.preprocessing_net_config import PreprocessingNetConfig
from preprocessing.neural_network_methods.data_module import PreprocessingDataModule
from preprocessing.neural_network_methods.model import PreprocessingModel
from preprocessing.neural_network_methods.model_inference import PreprocessingInference


def train_preprocessing_model(architecture_type: LightningModule, metrics: MetricCollection | None = None) -> None:
    preprocessing_config: PreprocessingNetConfig = load_config(Path("configs/preprocessing_net_config.yaml"))
    noise_transform_config: Dict[str, Dict[str, Any]] = load_config(
        Path("configs/noise_transforms_config.yaml")
    )

    model = PreprocessingModel(
        architecture_type=architecture_type,
        n_channels=preprocessing_config.model.n_channels,
        learning_rate=preprocessing_config.model.learning_rate,
        metrics=metrics
    )

    datamodule = PreprocessingDataModule(
        image_dir=preprocessing_config.dataloader.image_dir,
        noise_transform_config=noise_transform_config,
        batch_size=preprocessing_config.dataloader.batch_size,
        num_of_workers=preprocessing_config.dataloader.num_workers,
        train_ratio=preprocessing_config.dataloader.train_ratio,
    )

    trainer = Trainer(
        accelerator=preprocessing_config.training.accelerator,
        max_epochs=preprocessing_config.training.max_epochs,
    )
    trainer.fit(model, datamodule=datamodule)


def test_preprocessing_model(
    images: Image.Image | List[Image.Image], path_to_checkpoint: Path, transformations: transforms.Compose | None,
    model_params: Dict[str, Any], noise_type: str
) -> None:
    if isinstance(images, Image.Image):
        images = [images]

    noise_transform_config: Dict[str, Dict[str, Any]] = load_config(
        Path("configs/noise_transforms_config.yaml")
    )
    dncnn_inference = PreprocessingInference(
        model_type=PreprocessingModel, path_to_checkpoint=path_to_checkpoint, transformations=transformations,
        noise_transform_config=noise_transform_config, **model_params
    )
    dncnn_inference.inference_display(images, noise_type)
