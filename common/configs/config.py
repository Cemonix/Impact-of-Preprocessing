from pathlib import Path
from typing import Any, Dict

import yaml

from common.configs.preprocessing_net_config import PreprocessingNetConfig
from common.configs.unet_config import UNetConfig


def load_config(
    config_path: Path,
) -> PreprocessingNetConfig | UNetConfig | Dict[str, Any]:
    if "preprocessing_net" in config_path.stem:
        config_type = PreprocessingNetConfig
    elif "unet" in config_path.stem:
        config_type = UNetConfig
    elif (
        "noise_transforms" in config_path.stem
        or "standard_preprocessing" in config_path.stem
    ):
        config_type = dict
    else:
        raise ValueError("Invalid config file name or type")

    with open(config_path) as file:
        config_dict = yaml.safe_load(file)

    return config_dict if config_type == dict else config_type(**config_dict)
