import numpy as np
import torch
from PIL import Image
from common.data_manipulation import draw_polygons_on_image, mask_to_polygons, scale_polygons

from common.model_inference import ModelInference
from common.visualization import compare_images


class UnetInference(ModelInference):
    # TODO: Refactor to multiple images
    def inference_display(self, image: Image.Image, target: Image.Image, threshold: float = 0.5) -> None:
        mask = self.__pipeline(image)
        mask_prob = torch.sigmoid(mask)
        mask_binary = (mask_prob > threshold).float()
        mask_squeezed = mask_binary.squeeze(0).squeeze(0)
        mask_uint8 = (mask_squeezed * 255).numpy().astype(np.uint8)
        polygons = mask_to_polygons(mask_uint8)
        polygons = scale_polygons(polygons, image.size, mask_uint8.shape)
        image_with_polygons = draw_polygons_on_image(image, polygons)

        target_polygons = mask_to_polygons(np.array(target))
        target_with_polygons = draw_polygons_on_image(image, target_polygons)
        compare_images(
            [image, target_with_polygons, image_with_polygons],
            ["Original image", "Target", "Prediction"],
            zoom = False
        )

    def __pipeline(self, image: Image.Image) -> torch.Tensor:
        image_tensor = self.process_image(image)
        prediction = self.perform_inference(image_tensor)
        return prediction