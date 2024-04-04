from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from common.model_inference import ModelInference
from common.utils import apply_noise_transform
from common.visualization import compare_images


class PreprocessingInference(ModelInference):
    def __init__(
        self,
        noise_transform_config: Dict[str, Dict[str, Any]],
        noise_type: str,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.noise_transform_config = noise_transform_config
        self.noise_type = noise_type

    def inference_display(self, images: List[Image.Image]) -> None:
        images_for_display: List[Image.Image] = []
        for image in images:
            img = self.transform(image.copy())
            noised_tensor, prediction = self.__pipeline(image.copy())
            images_for_display.extend(
                [
                    to_pil_image(img),
                    to_pil_image(noised_tensor.squeeze(0).squeeze(0)),
                    to_pil_image(prediction.squeeze(0).squeeze(0)),
                ]
            )

        compare_images(
            images_for_display, ["Original image", "Noised image", "Prediction"]
        )

    def perform_inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prediction: torch.Tensor = self.model(image_tensor)
        return prediction.to(self.device)

    def __pipeline(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        noised_tensor = self.__preprocess_image(image)
        prediction = self.perform_inference(noised_tensor)
        return noised_tensor, prediction

    def __preprocess_image(self, image: Image.Image) -> torch.Tensor:
        noise_type_params = self.noise_transform_config[self.noise_type]
        noised_image = apply_noise_transform(
            np.array(image), self.noise_type, noise_type_params
        )
        return self.process_image(noised_image)
