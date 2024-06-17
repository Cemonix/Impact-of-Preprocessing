from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch
from PIL import Image
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms.functional import to_pil_image

from common.model_inference import ModelInference
from common.utils import apply_noises
from common.visualization import compare_images


class PreprocessingInference(ModelInference):
    def __init__(
        self,
        noise_transform_config: Dict[str, Dict[str, Any]],
        noise_types: List[str],
        metrics: MetricCollection | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.noise_transform_config = noise_transform_config
        self.noise_types = noise_types
        self.metrics = metrics if metrics is not None else MetricCollection(
            {
                "PSNR": PeakSignalNoiseRatio(),
                "SSIM": StructuralSimilarityIndexMeasure(),
            }
        )

    def inference_display(self, images: List[Image.Image]) -> None:
        images_for_display: List[Image.Image] = []
        for image in images:
            image_tensor = cast(torch.Tensor, self.transform(image.copy()))
            noised_tensor, prediction = self.__pipeline(image.copy())
            prediction = prediction.clamp(0, 1)
            images_for_display.extend(
                [
                    to_pil_image(image_tensor),
                    to_pil_image(noised_tensor.squeeze(0).squeeze(0)),
                    to_pil_image(prediction.squeeze(0).squeeze(0)),
                ]
            )

            print(f"Noised image metrics: {self.metrics(noised_tensor, image_tensor.unsqueeze(0))}")
            print(f"Model prediction metrics: {self.metrics(prediction, image_tensor.unsqueeze(0))}")

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
        noised_image = apply_noises(np.array(image), self.noise_types, self.noise_transform_config)
        return self.process_image(noised_image)
