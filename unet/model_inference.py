from typing import List, cast

import numpy as np
import torch
from PIL import Image
from common.data_manipulation import draw_polygons_on_image, mask_to_polygons, scale_polygons

from common.model_inference import ModelInference
from common.visualization import compare_images


class UnetInference(ModelInference):
    def inference_display(
        self, images: List[Image.Image], targets: List[Image.Image], threshold: float = 0.5
    ) -> None:
        images_for_comparison = []
        for image, target in zip(images, targets):
            target = np.array(target)
            target_polygons = mask_to_polygons(target)
            target_polygons = scale_polygons(target_polygons, image.size, cast(tuple, target.shape))
            target_with_polygons = draw_polygons_on_image(image, target_polygons)

            mask = self.__pipeline(image)
            mask_prob = torch.sigmoid(mask)
            mask_binary = (mask_prob > threshold).float().squeeze(0).squeeze(0)
            mask_uint8 = (mask_binary * 255).to(torch.uint8).numpy()
            polygons = mask_to_polygons(mask_uint8)
            polygons = scale_polygons(polygons, image.size, mask_uint8.shape)
            image_with_polygons = draw_polygons_on_image(image, polygons)

            images_for_comparison.extend([image, target_with_polygons, image_with_polygons])

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
