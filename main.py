from pytorch_lightning import Trainer
from UNet.unet_model import UNetModel
from UNet.data_module import LungSegmentationDataModule

if __name__ == '__main__':
    # Create model instance
    model = UNetModel(n_channels=1, n_classes=1)

    # Initialize the data module
    datamodule = LungSegmentationDataModule(
        image_dir="data/LungSegmentation/CXR_png/",
        mask_dir="data/LungSegmentation/masks/",
        batch_size=4,
    )

    trainer = Trainer(accelerator="cuda", max_epochs=50)
    trainer.fit(model, datamodule=datamodule)

    # TODO: Citovat dataset