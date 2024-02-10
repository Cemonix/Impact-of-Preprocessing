from pathlib import Path

from pydantic import BaseModel


class PreprocessingModelConfig(BaseModel):
    n_channels: int
    learning_rate: float


class PreprocessingDataLoaderConfig(BaseModel):
    image_dir: Path
    batch_size: int
    num_workers: int
    train_ratio: float


class PreprocessingTrainingConfig(BaseModel):
    accelerator: str
    max_epochs: int


class PreprocessingNetConfig(BaseModel):
    model: PreprocessingModelConfig
    dataloader: PreprocessingDataLoaderConfig
    training: PreprocessingTrainingConfig
