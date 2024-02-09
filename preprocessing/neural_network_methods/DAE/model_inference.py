from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from common.model_inference import ModelInference
from common.noise_transforms import NoiseTransformHandler
from common.visualization import compare_images
from preprocessing.neural_network_methods.DAE.model import DAEModel


class DAEInference(ModelInference):
    def load_model(self, path_to_model: Path) -> Any:
        return DAEModel.load_from_checkpoint(path_to_model)

    # TODO: Refactor
    def postprocess_and_display(self, image: Image.Image, prediction: torch.Tensor) -> None:
        img = self.transform(image.copy())
        img = to_pil_image(img)

        noise_transform_handler = NoiseTransformHandler()
        selected_noise_type = 'salt_and_pepper_noise'

        noised_image = noise_transform_handler.apply_noise_transform(
            np.array(image.copy()), selected_noise_type
        )

        noised_image = self.transform(noised_image)
        noised_image = to_pil_image(noised_image)

        pred = to_pil_image(prediction.squeeze(0).squeeze(0))

        compare_images(
            [img, noised_image, pred], ["Original image", "Noised image", "Prediction"], 3
        )
