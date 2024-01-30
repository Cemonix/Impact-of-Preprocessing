from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from preprocessing.neural_network_methods.DAE.model import DAEModel

# TODO: Create abstract class and inherit it here and in Unet


class DAEInference:
    def __init__(self, path_to_model: Path):
        self.model = self.load_model(path_to_model)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, path_to_model: Path) -> DAEModel:
        model = DAEModel.load_from_checkpoint(checkpoint_path=path_to_model)
        return model

    def process_image(self, path_to_img: str) -> torch.Tensor:
        img = Image.open(path_to_img).convert("L")
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        img_tensor: torch.Tensor = transform(img).unsqueeze(0)
        return img_tensor.to(self.device)

    def perform_inference(self, img_tensor) -> torch.Tensor:
        with torch.no_grad():
            output: torch.Tensor = self.model(img_tensor)
        return output.cpu()

    def postprocess_and_display(self, output: torch.Tensor) -> None:
        output_img = output.squeeze(0).squeeze(0).numpy()
        output_img = (output_img * 255).astype("uint8")
        output_pil = Image.fromarray(output_img)
        output_pil.show()
