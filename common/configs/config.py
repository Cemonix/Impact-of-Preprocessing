from pathlib import Path
from typing import Any, Dict

import yaml

from common.configs.dae_config import DAEConfig
from common.configs.unet_config import UnetConfig


def load_config(config_path: Path) -> DAEConfig | UnetConfig | Dict[str, Any]:
    if "dae" in config_path.stem:
        config_type = DAEConfig
    elif "unet" in config_path.stem:
        config_type = UnetConfig
    elif "noise" in config_path.stem:
        config_type = dict
    else:
        raise ValueError("Invalid config file name or type")

    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return config_dict if config_type == dict else config_type(**config_dict)
