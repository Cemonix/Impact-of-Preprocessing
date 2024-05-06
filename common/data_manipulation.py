import json
import os
from pathlib import Path
import random
from typing import Dict, List, Tuple, cast
import cv2
import numpy as np
from numpy import typing as npt
from PIL import Image, ImageDraw
from sympy import Point, Polygon
from skimage import transform
from rich.progress import track

from common.configs.config import load_config
from common.utils import apply_noise_transform, choose_params_from_minmax


def load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("L")


def resize_image(image: npt.NDArray, new_shape: tuple) -> npt.NDArray:
    """
    Resize an image to a new shape.

    Args:
        image (npt.NDArray): The original image array.
        new_shape (tuple): The desired shape (height, width) for the resized image. 
                           For a 3D array, this should include the number of channels (height, width, channels).

    Returns:
        np.ndarray: The resized image.
    """
    resized_image = transform.resize(image, new_shape, anti_aliasing=True)

    if image.dtype == np.uint8:
        resized_image = (resized_image * 255).astype(np.uint8)

    return resized_image


def mask_to_polygons(mask: npt.NDArray) -> List[Polygon]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = [Point(point[0][0], point[0][1]) for point in approx]

        # A polygon needs at least 3 points
        if len(points) > 2:
            polygon = Polygon(*points)
            polygons.append(polygon)

    return polygons


def scale_polygons(
    polygons: List[Polygon], image_size: Tuple[int, int], mask_size: Tuple[int, int]
) -> List[Polygon]:
    """
    Scale the coordinates of polygons from mask size to match the image size.

    Parameters:
    - polygons: List of SymPy Polygons.
    - mask_shape: Tuple of (height, width) representing the size of the mask.
    - image_size: Tuple of (width, height) representing the target image size.

    Returns:
    - List of scaled SymPy Polygons.
    """
    img_width, img_height = image_size
    mask_height, mask_width = mask_size
    scale_x = img_width / mask_width
    scale_y = img_height / mask_height

    scaled_polygons = []
    for polygon in polygons:
        scaled_vertices = [
            Point(point.x * scale_x, point.y * scale_y) for point in polygon.vertices # type: ignore
        ] 
        scaled_polygons.append(Polygon(*scaled_vertices))

    return scaled_polygons


def draw_polygons_on_image(
    image: Image.Image, polygons: List[Polygon],
    fill_color: Tuple[int, int, int, int] = (0, 255, 0, 128)
) -> Image.Image:
    """
    Draws polygons on an image with transparency.

    Parameters:
    - image: Image on which polygons will be drawn.
    - polygons: List of SymPy polygons.
    - fill_color: List of fill colors with an alpha value for transparency
        (default is semi-transparent red and green).

    Returns:
    - Image with polygons.
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    for polygon in polygons:
        vertices = [(float(p.x), float(p.y)) for p in polygon.vertices] # type: ignore
        draw_overlay.polygon(vertices, fill=fill_color)

    return Image.alpha_composite(image, overlay)


def create_dataset(
    image_dir: Path, save_dir: Path, walk_recursive: bool = False
) -> None:
    if not image_dir.is_dir():
        raise Exception(f"Given path {image_dir} is not a directory!")

    noise_transform_config = cast(
        Dict[str, Dict[str, List[float]]],
        load_config(Path("configs/noise_transforms_config.yaml")),
    )

    noise_types = list(noise_transform_config.keys())

    images_paths: List[Path] = []
    for dirpath, _, filenames in os.walk(image_dir):
        images_paths.extend([Path(dirpath, filename) for filename in filenames])
        if not walk_recursive:
            break

    dataset_info = {}
    random.shuffle(images_paths)
    for idx, image_path in track(
        sequence=enumerate(images_paths),
        description="Adding noise and moving images...",
        total=len(images_paths),
    ):
        image = load_image(image_path)
        chosen_noise_types = []
        if idx < len(images_paths) * 0.2:
            chosen_noise_types = [noise_types[0]]
        elif idx < len(images_paths) * 0.4:
            chosen_noise_types = [noise_types[1]]
        elif idx < len(images_paths) * 0.6:
            chosen_noise_types = [noise_types[2]]
        elif idx < len(images_paths) * 0.8:
            chosen_noise_types = random.choice(
                [
                    [noise_types[0], noise_types[1]],
                    [noise_types[0], noise_types[2]],
                    [noise_types[1], noise_types[2]],
                ]
            )
        else:
            chosen_noise_types = noise_types

        selected_params = []
        noised_image = np.array(image).copy()
        for idx, noise_type in enumerate(chosen_noise_types):
            params = noise_transform_config[noise_type]
            selected_params.append(choose_params_from_minmax(params))
            noised_image = apply_noise_transform(
                np.array(noised_image), noise_type, selected_params[idx]
            )

        noised_image = cast(Image.Image, noised_image)
        noised_image.save(save_dir / image_path.name)
        info = (
            f"Noise type: {chosen_noise_types[0]} | {selected_params}"
            if len(chosen_noise_types) == 1
            else f"Noise type: {chosen_noise_types} | {selected_params}"
        )

        dataset_info[image_path.stem] = info

    with open("../" / save_dir / "dataset_info.json", "w") as json_file:
        json.dump({k: dataset_info[k] for k in sorted(dataset_info.keys())}, json_file)
