from pathlib import Path
from typing import Any, Dict

from pytorch_lightning import Trainer
from torchmetrics import (
    MetricCollection,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from common.configs.config import load_config
from common.configs.dae_config import DAEConfig
from preprocessing.neural_network_methods.DAE.dae_model import DAEModel
from preprocessing.neural_network_methods.DAE.data_module import DAEDataModule

if __name__ == "__main__":
    dae_config: DAEConfig = load_config(Path("Configs/dae_config.yaml"))
    noise_transform_config: Dict[str, Dict[str, Any]] = load_config(
        Path("Configs/noise_transforms_config.yaml")
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
        num_workers=dae_config.dataloader.num_workers,
        train_ratio=dae_config.dataloader.train_ratio,
    )

    trainer = Trainer(
        accelerator=dae_config.training.accelerator,
        max_epochs=dae_config.training.max_epochs,
    )
    trainer.fit(model, datamodule=datamodule)
