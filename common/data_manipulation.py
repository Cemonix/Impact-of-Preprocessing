from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import cv2
from numpy import typing as npt
from PIL import Image, ImageDraw
from sympy import Point, Polygon


def load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("L")
    
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
        scaled_vertices = [Point(point.x * scale_x, point.y * scale_y) for point in polygon.vertices]  # type:ignore
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
        vertices = [(float(p.x), float(p.y)) for p in polygon.vertices] # type:ignore
        draw_overlay.polygon(vertices, fill=fill_color)

    return Image.alpha_composite(image, overlay)
