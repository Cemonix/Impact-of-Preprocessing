from pathlib import Path

from pydantic import BaseModel


class DAEModelConfig(BaseModel):
    n_channels: int
    learning_rate: float


class DAEDataLoaderConfig(BaseModel):
    image_dir: Path
    batch_size: int
    num_workers: int
    train_ratio: float


class DAETrainingConfig(BaseModel):
    accelerator: str
    max_epochs: int


class DAEConfig(BaseModel):
    model: DAEModelConfig
    dataloader: DAEDataLoaderConfig
    training: DAETrainingConfig
