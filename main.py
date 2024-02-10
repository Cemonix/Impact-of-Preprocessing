from pathlib import Path
from typing import Any, Dict

from pytorch_lightning import Trainer
from torchmetrics import (
    JaccardIndex,
    MetricCollection,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision.transforms import transforms

from common.configs.config import load_config
from common.configs.dae_config import DAEConfig
from common.configs.unet_config import UnetConfig
from common.data_manipulation import load_image
from preprocessing.neural_network_methods.DAE.data_module import DAEDataModule
from preprocessing.neural_network_methods.DAE.model import DAEModel
from preprocessing.neural_network_methods.DAE.model_inference import (
    DAEInference,
)
from preprocessing.neural_network_methods.DnCNN.data_module import DnCNNDataModule
from preprocessing.neural_network_methods.DnCNN.model import DnCNNModel
from preprocessing.neural_network_methods.DnCNN.model_inference import (
    DnCNNInference,
)
from unet.data_module import LungSegmentationDataModule
from unet.model import UNetModel
from unet.model_inference import UnetInference


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
    unet_inference = UnetInference(UNetModel, path_to_model)
    unet_inference.inference_display(image)


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


def dae_test_model(path_to_checkpoint: Path, path_to_image: Path) -> None:
    noise_transform_config: Dict[str, Dict[str, Any]] = load_config(
        Path("configs/noise_transforms_config.yaml")
    )
    image = load_image(path_to_image)
    dae_inference = DAEInference(
        model_type=DnCNNModel, path_to_checkpoint=path_to_checkpoint, noise_transform_config=noise_transform_config
    )
    dae_inference.inference_display(image, "gaussian_noise")


def dncnn_train():
    dncnn_config: DAEConfig = load_config(Path("configs/dae_config.yaml"))
    noise_transform_config: Dict[str, Dict[str, Any]] = load_config(
        Path("configs/noise_transforms_config.yaml")
    )

    metrics = MetricCollection(
        {
            "PSNR": PeakSignalNoiseRatio(),
            "SSIM": StructuralSimilarityIndexMeasure(),
        }
    )

    model = DnCNNModel(
        n_channels=dncnn_config.model.n_channels,
        learning_rate=dncnn_config.model.learning_rate,
        metrics=metrics,
    )

    datamodule = DnCNNDataModule(
        image_dir=dncnn_config.dataloader.image_dir,
        noise_transform_config=noise_transform_config,
        batch_size=dncnn_config.dataloader.batch_size,
        num_of_workers=dncnn_config.dataloader.num_workers,
        train_ratio=dncnn_config.dataloader.train_ratio,
    )

    trainer = Trainer(
        accelerator=dncnn_config.training.accelerator,
        max_epochs=dncnn_config.training.max_epochs,
    )
    trainer.fit(model, datamodule=datamodule)


def dncnn_test_model(path_to_checkpoint: Path, path_to_image: Path) -> None:
    noise_transform_config: Dict[str, Dict[str, Any]] = load_config(
        Path("configs/noise_transforms_config.yaml")
    )
    image = load_image(path_to_image)
    transformations = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )
    dncnn_inference = DnCNNInference(
        model_type=DnCNNModel, path_to_checkpoint=path_to_checkpoint,
        transformations=transformations, noise_transform_config=noise_transform_config
    )
    dncnn_inference.inference_display(image, "gaussian_noise")


if __name__ == "__main__":
    # unet_train()
    
    # dae_train()

    # dncnn_train()

    # unet_test_model(
    #     Path(
    #         "lightning_logs/unet_model_v0/checkpoints/epoch=99-step=3200.ckpt"
    #     ),
    #     Path("data/LungSegmentation/CXR_png/CHNCXR_0250_0.png"),
    # )

    # dae_test_model(
    #     Path(
    #         "lightning_logs/dae_model_v_3/checkpoints/epoch=60-step=915.ckpt"
    #     ),
    #     Path("data/LungSegmentation/CXR_png/CHNCXR_0001_0.png"),
    # )

    dncnn_test_model(
        Path(
            "lightning_logs/dncnn_model_v0_512/checkpoints/epoch=49-step=750.ckpt"
        ),
        Path("data/LungSegmentation/CXR_png/CHNCXR_0001_0.png"),
    )

    # TODO: Citovat dataset
