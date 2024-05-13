from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
from numpy import typing as npt
import torch
from PIL import Image
from common.data_manipulation import create_mask_from_annotation, draw_polygons_on_image, mask_to_polygons, scale_polygons

from common.model_inference import ModelInference
from common.visualization import compare_images


class UnetInference(ModelInference):
    def inference_display(
        self, images: List[Image.Image], targets: List[Image.Image] | List[Path], threshold: float = 0.5,
        multiclass: bool = False
    ) -> None:        
        images_for_comparison = []
        for image, target in zip(images, targets):
            if isinstance(target, Path):
                target = create_mask_from_annotation(target)
            else:
                target = np.array(target)

            mask = self.__pipeline(image)

            if multiclass:
                target_with_polygons, pred_with_polygons = self.__process_multiclass(
                    image, mask, target
                )
            else:
                target_with_polygons, pred_with_polygons = self.__process_binary(
                    image, mask, target, threshold
                )

            images_for_comparison.extend([image, target_with_polygons, pred_with_polygons])

        compare_images(
            images=images_for_comparison, titles=["Original image", "Target", "Prediction"], zoom=False
        )

    def perform_inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prediction: torch.Tensor = self.model(image_tensor)
        return prediction.to(self.device)
    
    def __pipeline(self, image: Image.Image) -> torch.Tensor:
        image_tensor = self.process_image(image)
        prediction = self.perform_inference(image_tensor)
        return prediction
    
    def __create_mask_with_polygons(
        self, image: Image.Image, mask: npt.NDArray
    ) -> Image.Image:
        mask_polygons = mask_to_polygons(mask)
        mask_polygons = scale_polygons(mask_polygons, image.size, cast(tuple, mask.shape))
        mask_with_polygons = draw_polygons_on_image(image, mask_polygons)
        return mask_with_polygons

    def __process_binary(
        self, image: Image.Image, mask: torch.Tensor, target: npt.NDArray, threshold: float
    ) -> Tuple[Image.Image, Image.Image]:
        target_with_polygons = self.__create_mask_with_polygons(image, target)

        mask_prob = torch.sigmoid(mask)
        mask_binary = (mask_prob > threshold).float().squeeze(0).squeeze(0)
        pred_mask = (mask_binary * 255).to(torch.uint8).numpy()
        pred_with_polygons = self.__create_mask_with_polygons(image, pred_mask)
        return target_with_polygons, pred_with_polygons

    def __process_multiclass(
        self, image: Image.Image, mask: torch.Tensor, target: npt.NDArray
    ) -> Tuple[Image.Image, Image.Image]:
        target_with_polygons = self.__create_mask_with_polygons(image, target)
        pred_mask = torch.argmax(mask, dim=1).squeeze(0).to(torch.uint8).numpy()
        pred_with_polygons = self.__create_mask_with_polygons(image, pred_mask)
        return target_with_polygons, pred_with_polygons