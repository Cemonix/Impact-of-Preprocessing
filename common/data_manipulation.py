from pathlib import Path

import numpy as np
from numpy import typing as npt
from PIL import Image


def load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("L")
