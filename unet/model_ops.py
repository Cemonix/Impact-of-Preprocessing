from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from pytorch_lightning import Trainer
from torchmetrics import MetricCollection
from torchvision.transforms import transforms

from common.configs.config import load_config
from common.configs.unet_config import UnetConfig
from unet.data_module import LungSegmentationDataModule
from unet.model import UNetModel
from unet.model_inference import UnetInference


def train_unet_model(metrics: MetricCollection | None = None) -> None:
    unet_config: UnetConfig = load_config(Path("configs/unet_config.yaml"))

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
    )
    trainer.fit(model, datamodule=datamodule)

# TODO: Send target
def test_unet_model(
    path_to_checkpoint: Path, images: Image.Image | List[Image.Image], transformations: transforms.Compose | None
) -> None:
    if isinstance(images, Image.Image):
        images = [images]

    # TODO: Send list of images to inference
    unet_inference = UnetInference(
        model_type=UNetModel, path_to_checkpoint=path_to_checkpoint, transformations=transformations
    )
    unet_inference.inference_display(images[0])
