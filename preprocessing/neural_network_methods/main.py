from pathlib import Path
from typing import Any, Dict

from torchmetrics import (
    MetricCollection,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from common.configs.config import load_config
from preprocessing.neural_network_methods.DAE.dae_model import DAEModel

if __name__ == "__main__":
    noise_transform_config: Dict[str, Dict[str, Any]] = load_config(
        Path("Configs/noise_transforms_config.yaml")
    )

    metrics = MetricCollection(
        {
            "psnr": PeakSignalNoiseRatio(),
            "ssim": StructuralSimilarityIndexMeasure(),
        }
    )

    dae_model = DAEModel(n_channels=1, learning_rate=1e-4, metrics=metrics)

    # TODO: Finish DAEConfig
    # TODO: Finish DAE main
    # TODO: Debug code
