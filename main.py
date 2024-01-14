from pytorch_lightning import Trainer
from torchmetrics import MetricCollection,JaccardIndex
from UNet.unet_model import UNetModel
from UNet.data_module import LungSegmentationDataModule

if __name__ == '__main__':
    metrics = MetricCollection(
        {'jaccard_index': JaccardIndex(task="binary", num_classes=1)}
    )
    
    # Create model instance
    model = UNetModel(n_channels=1, n_classes=1, metrics=metrics)

    # Initialize the data module
    datamodule = LungSegmentationDataModule(
        image_dir="data/LungSegmentation/CXR_png_200/",
        mask_dir="data/LungSegmentation/masks_200/"
    )

    trainer = Trainer(accelerator="cuda", max_epochs=100)
    trainer.fit(model, datamodule=datamodule)

    # TODO: Načíst model a podívat se jaké generuje masky
    # TODO: Udělat konfigurační soubor pro nastavování hyperparametrů, metrik
    # TODO: Citovat dataset