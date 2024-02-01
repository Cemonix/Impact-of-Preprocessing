from pathlib import Path
from typing import Any, Dict

from pytorch_lightning import Trainer
from torchmetrics import (
    JaccardIndex,
    MetricCollection,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from common.configs.config import load_config
from common.configs.dae_config import DAEConfig
from common.configs.unet_config import UnetConfig
from common.data_manipulation import load_image
from preprocessing.neural_network_methods.DAE.data_module import DAEDataModule
from preprocessing.neural_network_methods.DAE.model import DAEModel
from preprocessing.neural_network_methods.DAE.model_inference import (
    DAEInference,
)
from unet.data_module import LungSegmentationDataModule
from unet.model import UNetModel
from unet.model_inference import UnetInference


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


def dae_test_model(path_to_model: Path, path_to_image: Path) -> None:
    image = load_image(path_to_image)
    dae_inference = DAEInference(path_to_model)
    img_tensor = dae_inference.process_image(image)
    prediction = dae_inference.perform_inference(img_tensor)
    dae_inference.postprocess_and_display(image, prediction)


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


def unet_test_model(path_to_model: Path, path_to_image: Path) -> None:
    image = load_image(path_to_image)
    unet_inference = UnetInference(path_to_model)
    img_tensor = unet_inference.process_image(image)
    prediction = unet_inference.perform_inference(img_tensor)
    unet_inference.postprocess_and_display(image, prediction)


if __name__ == "__main__":
    unet_test_model(
        Path(
            "lightning_logs/main_model/checkpoints/epoch=99-step=3200.ckpt"
        ),
        Path("data/LungSegmentation/CXR_png/CHNCXR_0250_0.png"),
    )

    # dae_test_model(
    #     Path(
    #         "lightning_logs/dae_model_v_0/checkpoints/epoch=49-step=1600.ckpt"
    #     ),
    #     Path("data/LungSegmentation/CXR_png/CHNCXR_0001_0.png"),
    # )

    # TODO: Think about dae model - apply only one noise, make net smaller, ...

    # TODO: Citovat dataset
