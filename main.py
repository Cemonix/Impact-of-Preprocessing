from pathlib import Path

from pytorch_lightning import Trainer
from torchmetrics import (
    MetricCollection,
    JaccardIndex,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from typing import Any, Dict

from common.configs.config import load_config
from common.configs.unet_config import UnetConfig
from common.configs.dae_config import DAEConfig
from unet.data_module import LungSegmentationDataModule
from unet.unet_model import UNetModel
from preprocessing.neural_network_methods.DAE.dae_model import DAEModel
from preprocessing.neural_network_methods.DAE.data_module import DAEDataModule

def dae_train():
    dae_config: DAEConfig = load_config(Path("configs/dae_config.yaml"))
    noise_transform_config: Dict[str, Dict[str, Any]] = load_config(
        Path("configs/noise_transforms_config.yaml")
    )

    metrics = MetricCollection(
        {
            "PSNR": PeakSignalNoiseRatio(),
            "SSIM": StructuralSimilarityIndexMeasure(),
        }
    )

    model = DAEModel(
        n_channels=dae_config.model.n_channels,
        learning_rate=dae_config.model.learning_rate,
        metrics=metrics,
    )

    datamodule = DAEDataModule(
        image_dir=dae_config.dataloader.image_dir,
        noise_transform_config=noise_transform_config,
        batch_size=dae_config.dataloader.batch_size,
        num_of_workers=dae_config.dataloader.num_workers,
        train_ratio=dae_config.dataloader.train_ratio,
    )

    trainer = Trainer(
        accelerator=dae_config.training.accelerator,
        max_epochs=dae_config.training.max_epochs,

    )
    trainer.fit(model, datamodule=datamodule)

def unet_train():
    unet_config: UnetConfig = load_config(Path("configs/unet_config.yaml"))

    metrics = MetricCollection(
        {"jaccard_index": JaccardIndex(task="binary", num_classes=1)}
    )

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

if __name__ == "__main__":
    dae_train()

    # TODO: Implementovat načítání modelů
    # TODO: Implementovat vizualizaci výstupu z modelu
    # TODO: První DAE - denoising autoencoder

    # TODO: Citovat dataset
