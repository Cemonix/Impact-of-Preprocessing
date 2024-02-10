from pathlib import Path
from torchmetrics import (
    JaccardIndex,
    MetricCollection,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision.transforms import transforms

from common.data_manipulation import load_image
from preprocessing.neural_network_methods.DAE.denoising_autoencoder import DenoisingAutoencoder
from preprocessing.neural_network_methods.DnCNN.dncnn import DnCNN
from preprocessing.neural_network_methods.model_ops import test_preprocessing_model, train_preprocessing_model
from unet.model_ops import test_unet_model, train_unet_model

if __name__ == "__main__":
    # Unet train
    # train_unet_model()

    # Preprocessing train
    # metrics = MetricCollection(
    #     {
    #         "PSNR": PeakSignalNoiseRatio(),
    #         "SSIM": StructuralSimilarityIndexMeasure(),
    #     }
    # )
    # train_preprocessing_model(DenoisingAutoencoder, metrics)
    # train_preprocessing_model(DnCNN, metrics)

    image = load_image(Path("data/LungSegmentation/CXR_png/CHNCXR_0001_0.png"))

    # Unet test
    # transformations = transforms.Compose(
    #     [
    #         transforms.Resize((256, 256)),
    #         transforms.ToTensor(),
    #     ]
    # )
    # unet_checkpoint_path = Path("lightning_logs/unet_model_v0/checkpoints/epoch=99-step=3200.ckpt")
    # test_unet_model(unet_checkpoint_path, image, transformations)

    # Preprocessing test
    transformations = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )

    preprocessing_checkpoint_path = Path("lightning_logs/dncnn_model_v0_512/checkpoints/epoch=49-step=750.ckpt")
    model_params = {"architecture_type": DnCNN}
    noise_type = 'gaussian_noise'
    test_preprocessing_model(image, preprocessing_checkpoint_path, transformations, model_params, noise_type)

    # TODO: Citovat dataset
