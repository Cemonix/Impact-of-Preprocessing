from pathlib import Path
from typing import Callable, List, cast

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
        def create_mask_with_polygons(image: Image.Image, mask: npt.NDArray) -> Image.Image:
            mask_polygons = mask_to_polygons(mask)
            mask_polygons = scale_polygons(mask_polygons, image.size, cast(tuple, mask.shape))
            mask_with_polygons = draw_polygons_on_image(image, mask_polygons)
            return mask_with_polygons
        
        images_for_comparison = []
        for image, target in zip(images, targets):
            if isinstance(target, Path):
                target = create_mask_from_annotation(target)
            else:
                target = np.array(target)
            target_with_polygons = create_mask_with_polygons(image, target)

            mask = self.__pipeline(image)

            if multiclass:
                pred_mask = torch.argmax(mask, dim=1).squeeze(0).numpy()
            else:
                mask_prob = torch.sigmoid(mask)
                mask_binary = (mask_prob > threshold).float().squeeze(0).squeeze(0)
                pred_mask = (mask_binary * 255).to(torch.uint8).numpy()

            pred_with_polygons = create_mask_with_polygons(image, pred_mask)

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
