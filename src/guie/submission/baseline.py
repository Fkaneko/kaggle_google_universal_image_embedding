# from https://www.kaggle.com/competitions/google-universal-image-embedding/overview/evaluation
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from src.common.utils import set_logger
from src.guie.submission.check_submission import test_embedding_format
from src.guie.submission.interface import (
    EMBED_DIM_64,
    INPUT_MEAN,
    INPUT_STD,
    SAMPLE_IMG_PATH,
    Embed64Wrapper,
    Transform,
    extract_embedding,
    load_jit_model,
    save_model_as_jit,
)

logger = logging.getLogger(__name__)

BASELINE_INPUT_SIZE = (224, 224)


class BaselineEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        inception_model = models.inception_v3(pretrained=True)
        inception_model.fc = nn.AdaptiveAvgPool1d(EMBED_DIM_64)
        self.feature_extractor = inception_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x).logits


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        inception_model = models.inception_v3(pretrained=True)
        # inception_model.fc = nn.Linear(2048, EMBED_DIM)
        inception_model.fc = nn.AdaptiveAvgPool1d(EMBED_DIM_64)
        self.feature_extractor = inception_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = transforms.functional.resize(x, size=[224, 224])
        x = x / 255.0
        x = transforms.functional.normalize(
            x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return self.feature_extractor(x).logits


def main() -> None:

    set_logger(logger=logger)
    model = MyModel()
    saved_model_path = Path("saved_model.pt")

    save_model_as_jit(model=model, saved_model_path=saved_model_path)

    embedding_fn = load_jit_model(saved_model_path=saved_model_path)
    embedding = extract_embedding(embedding_fn=embedding_fn, image_path=SAMPLE_IMG_PATH)
    logger.info(f"Output embedding: {embedding.shape}, {embedding.dtype}, \n {embedding}")

    baseline_transform = Transform(image_size=BASELINE_INPUT_SIZE, mean=INPUT_MEAN, std=INPUT_STD)
    baseline_encoder = BaselineEncoder()

    model_with_wrapper = Embed64Wrapper(
        image_encoder=baseline_encoder, transform=baseline_transform
    )
    save_model_as_jit(model=model_with_wrapper, saved_model_path=saved_model_path)
    embedding_fn_with_wrapper = load_jit_model(saved_model_path=saved_model_path)
    embedding_with_wrapper = extract_embedding(
        embedding_fn=embedding_fn_with_wrapper, image_path=SAMPLE_IMG_PATH
    )
    logger.info(
        f"Output embedding: {embedding_with_wrapper.shape}, {embedding_with_wrapper.dtype}, \n {embedding_with_wrapper}"
    )
    test_embedding_format(embedding=embedding_with_wrapper)

    assert np.all(embedding == embedding_with_wrapper)


if __name__ == "__main__":
    main()
