# from https://www.kaggle.com/competitions/google-universal-image-embedding/overview/evaluation
import logging
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    ConvertImageDtype,
    Normalize,
    PILToTensor,
    Resize,
)
from torchvision.transforms.functional import InterpolationMode

logger = logging.getLogger(__name__)

SAMPLE_IMG_PATH = Path("./images/sample_image.png")

INPUT_MEAN = (0.485, 0.456, 0.406)
INPUT_STD = (0.229, 0.224, 0.225)
EMBED_DIM_64 = 64


def save_model_as_jit(model: nn.Module, saved_model_path: Path) -> None:
    model.eval()
    saved_model = torch.jit.script(model)
    saved_model.save(str(saved_model_path))
    logger.info(f"save jit model at {str(saved_model_path)}")


def load_jit_model(saved_model_path: Path) -> nn.Module:
    # Model loading.
    logger.info(f"loading jit model from {str(saved_model_path)}")
    model = torch.jit.load(str(saved_model_path))
    model.eval()
    embedding_fn = model
    return embedding_fn


def extract_embedding(embedding_fn: nn.Module, image_path: Path) -> np.ndarray:
    # Load image and extract its embedding.
    input_image = Image.open(image_path).convert("RGB")
    convert_to_tensor = Compose([PILToTensor()])
    input_tensor = convert_to_tensor(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = torch.flatten(embedding_fn(input_batch)[0]).cpu().data.numpy()
    return embedding


class Transform(torch.nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            # Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            ConvertImageDtype(torch.float32),
            Normalize(mean, std),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transforms(x)
        return x


class CenterCropTransform(torch.nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        crop_pct: float = 0.875,
    ) -> None:
        super().__init__()

        if isinstance(image_size, (tuple, list)):
            scale_size = [int(x / crop_pct) for x in image_size]
        else:
            scale_size = [int(math.floor(image_size / crop_pct))] * 2

        self.transforms = torch.nn.Sequential(
            Resize(scale_size, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float32),
            Normalize(mean, std),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transforms(x)
        return x


class Embed64Wrapper(nn.Module):
    def __init__(self, image_encoder: nn.Module, transform: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = image_encoder
        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform(x)
        x = self.feature_extractor(x)
        return x
