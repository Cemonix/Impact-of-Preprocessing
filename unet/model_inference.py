from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
from numpy import typing as npt
import torch
from sympy import Point, Polygon
from PIL import Image
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from common.model_inference import ModelInference
from unet.model import UNetModel


class UnetInference(ModelInference):
    def load_model(self, path_to_model: Path) -> Any:
        return UNetModel.load_from_checkpoint(path_to_model)

    def postprocess_and_display(self, image: Image.Image, mask: torch.Tensor, threshold: float = 0.5) -> None:
        mask_prob = torch.sigmoid(mask)
        mask_binary = (mask_prob > threshold).float()
        mask_squeezed = mask_binary.squeeze(0).squeeze(0)
        mask_uint8 = (mask_squeezed * 255).numpy().astype(np.uint8)
        polygons = self.__mask_to_polygons(mask_uint8)
        polygons = self.__scale_polygons(polygons, image.size, mask_uint8.shape)
        self.__display_polygon_on_image(np.array(image), polygons)

    @staticmethod
    def __mask_to_polygons(mask: npt.NDArray) -> List[Polygon]:
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

    @staticmethod
    def __display_polygon_on_image(img_array: npt.NDArray, polygons: List[Polygon]) -> None:
        fig = make_subplots(rows=1, cols=2)

        fig.add_trace(
            go.Heatmap(z=img_array, colorscale='gray', showscale=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=img_array, colorscale='gray', showscale=False),
            row=1, col=2
        )

        for polygon in polygons:
            x, y = [], []
            for point in polygon.vertices:
                x.append(float(point.x))
                y.append(float(point.y))

            fig.add_trace(
                go.Scatter(x=x, y=y, mode='lines', fill='toself'),
                row=1, col=2
            )

        fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=1, scaleanchor="x", autorange="reversed")

        fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=2)
        fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=2, scaleanchor="x", autorange="reversed")
        fig.show()

    @staticmethod
    def __scale_polygons(
        polygons: List[Polygon], image_size: Tuple[int, int], mask_size: Tuple[int, int]
    ) -> List[Polygon]:
        """
        Scale the coordinates of polygons from mask size to match the image size.

        Parameters:
        - polygons: List of SymPy Polygon objects.
        - mask_shape: Tuple of (height, width) representing the size of the mask.
        - image_size: Tuple of (width, height) representing the target image size.

        Returns:
        - List of scaled SymPy Polygon objects.
        """
        img_width, img_height = image_size
        mask_height, mask_width = mask_size
        scale_x = img_width / mask_width
        scale_y = img_height / mask_height

        scaled_polygons = []
        for polygon in polygons:
            scaled_vertices = [Point(point.x * scale_x, point.y * scale_y) for point in polygon.vertices]
            scaled_polygons.append(Polygon(*scaled_vertices))

        return scaled_polygons
