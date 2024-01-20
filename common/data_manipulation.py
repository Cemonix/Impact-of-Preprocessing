from pathlib import Path

import numpy as np
from numpy import typing as npt
from PIL import Image


def load_image(image_path: Path) -> npt.NDArray:
    """Loads an image as a grayscale numpy array."""
    with Image.open(image_path) as img:
        return np.array(img.convert("L"))
