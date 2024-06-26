from pathlib import Path

from pydantic import BaseModel


class UnetModelConfig(BaseModel):
    n_channels: int
    n_classes: int
    learning_rate: float


class UnetDataLoaderConfig(BaseModel):
    image_dir: Path
    masks_dir: Path
    batch_size: int
    num_workers: int
    train_ratio: float


class UnetTrainingConfig(BaseModel):
    accelerator: str
    max_epochs: int
    log_every_n_steps: int


class UNetConfig(BaseModel):
    model: UnetModelConfig
    dataloader: UnetDataLoaderConfig
    training: UnetTrainingConfig
