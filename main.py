import yaml
from pathlib import Path
from pytorch_lightning import Trainer
from torchmetrics import MetricCollection,JaccardIndex

from utils.configs import UnetConfig
from UNet.unet_model import UNetModel
from UNet.data_module import LungSegmentationDataModule

def load_config(config_path: Path) -> UnetConfig:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
        return UnetConfig(**config_dict)

if __name__ == '__main__':
    unet_config = load_config(Path('Configs/unet_config.yaml'))

    metrics = MetricCollection(
        {'jaccard_index': JaccardIndex(task="binary", num_classes=1)}
    )
    
    # Create model instance
    model = UNetModel(
        n_channels=unet_config.model.n_channels,
        n_classes=unet_config.model.n_classes,
        learning_rate=unet_config.model.learning_rate,
        metrics=metrics
    )

    # Initialize the data module
    datamodule = LungSegmentationDataModule(
        image_dir=unet_config.dataloader.image_dir, 
        mask_dir=unet_config.dataloader.masks_dir,
        batch_size=unet_config.dataloader.batch_size, 
        num_workers=unet_config.dataloader.num_workers,
        train_ratio=unet_config.dataloader.train_ratio
    )

    trainer = Trainer(
        accelerator=unet_config.training.accelerator, max_epochs=unet_config.training.max_epochs
    )
    trainer.fit(model, datamodule=datamodule)

    # TODO: Načíst model a podívat se jaké generuje masky
    # TODO: Udělat konfigurační soubor pro nastavování hyperparametrů, metrik
    # TODO: Citovat dataset