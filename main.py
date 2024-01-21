from pathlib import Path

from pytorch_lightning import Trainer
from torchmetrics import JaccardIndex, MetricCollection

from common.configs.config import load_config
from common.configs.unet_config import UnetConfig
from unet.data_module import LungSegmentationDataModule
from unet.unet_model import UNetModel

if __name__ == "__main__":
    unet_config: UnetConfig = load_config(Path("Configs/unet_config.yaml"))

    metrics = MetricCollection(
        {"jaccard_index": JaccardIndex(task="binary", num_classes=1)}
    )

    # Create model instance
    model = UNetModel(
        n_channels=unet_config.model.n_channels,
        n_classes=unet_config.model.n_classes,
        learning_rate=unet_config.model.learning_rate,
        metrics=metrics,
    )

    # Initialize the data module
    datamodule = LungSegmentationDataModule(
        image_dir=unet_config.dataloader.image_dir,
        mask_dir=unet_config.dataloader.masks_dir,
        batch_size=unet_config.dataloader.batch_size,
        num_workers=unet_config.dataloader.num_workers,
        train_ratio=unet_config.dataloader.train_ratio,
    )

    trainer = Trainer(
        accelerator=unet_config.training.accelerator,
        max_epochs=unet_config.training.max_epochs,
    )
    trainer.fit(model, datamodule=datamodule)

    # TODO: Implementovat načítání modelů
    # TODO: Implementovat vizualizaci výstupu z modelu
    # TODO: První DAE - denoising autoencoder

    # TODO: Citovat dataset
