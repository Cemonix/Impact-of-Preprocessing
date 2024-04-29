from typing import Optional
from pathlib import Path
from numpy import typing as npt 

from torchmetrics import (
    MetricCollection,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision.transforms import transforms

from common.data_manipulation import load_image
from common.visualization import plot
from preprocessing.neural_networks.dncnn import DnCNN
from preprocessing.neural_networks.model_ops import (
    test_preprocessing_model,
    train_preprocessing_model,
)
from statistics_methods.statistics import estimate_noise_in_image
from unet.model_ops import test_unet_model, train_unet_model

def statistics(self, image: npt.NDArray, save_path: Optional[Path] = None) -> None:
    eigenvalues, noise_std = estimate_noise_in_image(image)
    plot(
        data=eigenvalues, fig_size=(10, 6), marker='o', title="PCA Eigenvalues of Image Patches", 
        xlabel="Component Index", ylabel="Eigenvalue (Explained Variance)", save_path=save_path
    )
    print("Estimated noise standard deviation:", noise_std)

TRAIN = True
UNET = False

if __name__ == "__main__":
    if TRAIN:
        if UNET:
            train_unet_model()
        else:
            metrics = MetricCollection(
                {
                    "PSNR": PeakSignalNoiseRatio(),
                    "SSIM": StructuralSimilarityIndexMeasure(),
                }
            )
            train_preprocessing_model("DenoisingAutoencoder", metrics)
    else:
        image = load_image(Path("data/dataset/images/CHNCXR_0005_0.png"))
        target = load_image(Path("data/dataset/masks/CHNCXR_0005_0_mask.png"))
        transformations = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

        if UNET:
            unet_checkpoint_path = Path(
                "lightning_logs/unet_model_v0/checkpoints/epoch=99-step=3200.ckpt"
            )
            test_unet_model(unet_checkpoint_path, image, target, transformations)
        else:
            preprocessing_checkpoint_path = Path(
                "lightning_logs/dncnn_model_v0_512/checkpoints/epoch=49-step=750.ckpt"
            )
            model_params = {"architecture_type": DnCNN}
            noise_type = "gaussian_noise"
            test_preprocessing_model(
                image,
                preprocessing_checkpoint_path,
                transformations,
                model_params,
                noise_type,
            )

    # TODO: Otestovat a natrenovat VAE

    # TODO: Implementovat denoising autoencoder jako UNet - skip connections
    # TODO: Implementovat pipeline preprocessing -> UNet -> mereni vlivu
    # TODO: Vyzkouset pouziti klasickych metod
    # TODO: Vyzkouset vliv augmentace dat

    # TODO: Citovat dataset
