from pytorch_lightning import Trainer
from torchmetrics import MetricCollection,JaccardIndex
from UNet.unet_model import UNetModel
from UNet.data_module import LungSegmentationDataModule

if __name__ == '__main__':
    metrics = MetricCollection(
        {'jaccard_index': JaccardIndex(num_classes=2)}
    )
    
    # Create model instance
    model = UNetModel(n_channels=1, n_classes=2, metrics=metrics)

    # Initialize the data module
    datamodule = LungSegmentationDataModule(
        image_dir="data/LungSegmentation/CXR_png/",
        mask_dir="data/LungSegmentation/masks/"
    )

    trainer = Trainer(accelerator="cuda", max_epochs=100)
    trainer.fit(model, datamodule=datamodule)

    # TODO: Udělat konfigurační soubor pro nastavování hyperparametrů, metrik
    # TODO: Citovat dataset