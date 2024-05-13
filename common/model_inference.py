from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type, cast

import torch
from PIL import Image
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms


class ModelInference(ABC):
    def __init__(
        self, model_type: Type[LightningModule], path_to_checkpoint: Path,
        transformations: transforms.Compose | None = None, device: str = "cpu", **kwargs
    ) -> None:
        self.device = device
        self.transform = transformations if transformations else transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        self.model = self.__load_model(model_type, path_to_checkpoint, **kwargs)
        self.model.to(self.device)
        self.model.eval()

    def process_image(self, image: Image.Image) -> torch.Tensor:
        img_tensor: torch.Tensor = cast(torch.Tensor, self.transform(image)).unsqueeze(0)
        return img_tensor.to(self.device)

    @abstractmethod
    def perform_inference(self, image_tensor: torch.Tensor) -> Any:
        pass

    @staticmethod
    def __load_model(
        model_type: Type[LightningModule], path_to_checkpoint: Path, **kwargs
    ) -> LightningModule:
        """
        Loads a PyTorch Lightning model from a checkpoint file.

        Parameters:
        - model_class: The class of the model to load.
        - checkpoint_path: The path to the checkpoint file.

        Returns:
        - An instance of the model loaded with weights from the checkpoint.
        """
        return model_type.load_from_checkpoint(checkpoint_path=path_to_checkpoint, **kwargs)
