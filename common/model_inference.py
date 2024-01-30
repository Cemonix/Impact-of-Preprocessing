from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import transforms


class ModelInference(metaclass=ABCMeta):
    def __init__(
        self, path_to_model: Path, transformations: transforms = None
    ) -> None:
        self.model = self.load_model(path_to_model)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.transform = (
            transformations
            if transformations
            else transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ]
            )
        )
        self.model.to(self.device)
        self.model.eval()

    def process_image(self, path_to_img: Path) -> torch.Tensor:
        img = Image.open(path_to_img).convert("L")
        img_tensor: torch.Tensor = self.transform(img).unsqueeze(0)
        return img_tensor.to(self.device)

    def perform_inference(self, img_tensor) -> torch.Tensor:
        with torch.no_grad():
            output: torch.Tensor = self.model(img_tensor)
        return output.cpu()

    @abstractmethod
    def load_model(self, path_to_model: Path) -> Any:
        pass

    @abstractmethod
    def postprocess_and_display(self, output: Any) -> None:
        pass
