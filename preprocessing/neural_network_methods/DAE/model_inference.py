from pathlib import Path
from typing import Any

from numpy import typing as npt
import torch
from PIL import Image

from common.model_inference import ModelInference
from preprocessing.neural_network_methods.DAE.model import DAEModel


class DAEInference(ModelInference):
    def load_model(self, path_to_model: Path) -> Any:
        return DAEModel.load_from_checkpoint(path_to_model)

    def postprocess_and_display(self, image: Image.Image, prediction: torch.Tensor) -> None:
        output_img = prediction.squeeze(0).squeeze(0).numpy()
        output_img = (output_img * 255).astype("uint8")
        output_pil = Image.fromarray(output_img)
        output_pil.show()
