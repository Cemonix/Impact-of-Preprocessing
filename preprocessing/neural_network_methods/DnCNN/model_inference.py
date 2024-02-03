from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from common.model_inference import ModelInference
from common.noise_transforms import NoiseTransformHandler
from preprocessing.neural_network_methods.DnCNN.model import DnCNNModel


class DnCNNInference(ModelInference):
    def load_model(self, path_to_model: Path) -> Any:
        return DnCNNModel.load_from_checkpoint(path_to_model)

    def postprocess_and_display(self, image: Image.Image, prediction: torch.Tensor) -> None:
        # TODO: Fix
        from torchvision import transforms
        from torchvision.transforms.functional import to_pil_image

        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        img = transform(image.copy())
        to_pil_image(img).show()

        noise_transform_handler = NoiseTransformHandler()
        selected_noise_type = 'salt_and_pepper_noise'

        noised_image = noise_transform_handler.apply_noise_transform(
            np.array(image.copy()), selected_noise_type
        )
        noised_image: Image.Image = to_pil_image(transform(noised_image))
        noised_image.show()

        output_img = prediction.squeeze(0).squeeze(0).numpy()
        output_img = (output_img * 255).astype("uint8")
        output_pil = Image.fromarray(output_img)
        output_pil.show()
