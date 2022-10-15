import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, PILToTensor, Resize
from torchvision.transforms.functional import InterpolationMode

from src.common.utils import set_logger
from src.guie.model.clip.constant import CLIP_INPUT_MEAN, CLIP_INPUT_SIZE, CLIP_INPUT_STD
from src.guie.submission.check_submission import test_embedding_format
from src.guie.submission.interface import (
    EMBED_DIM_64,
    SAMPLE_IMG_PATH,
    Embed64Wrapper,
    Transform,
    extract_embedding,
    load_jit_model,
    save_model_as_jit,
)

logger = logging.getLogger(__name__)

CLIP_SAVE_NAME = "saved_model.pt"


class ClipEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_e16",
        use_512_dim: bool = False,
    ) -> None:
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        # delete text part
        model.transformer = nn.Identity()
        model.token_embedding = nn.Identity()

        # "ViT-B-32-quickgelu", pretrained="laion400m_e32"
        self.feature_extractor = model.visual
        self.pool = nn.AdaptiveAvgPool1d(EMBED_DIM_64)
        self.use_512_dim = use_512_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = F.normalize(x, dim=-1)
        if self.use_512_dim:
            return x
        x = self.pool(x)
        x = F.normalize(x, dim=-1)
        return x


def main() -> None:

    set_logger(logger=logging.getLogger())
    # open_clip_model = ("ViT-H-14", "laion2b_s32b_b79k")
    open_clip_model = ("ViT-L-14", "laion2b_s32b_b82k")

    model_name = open_clip_model[0]
    pretrained = open_clip_model[1]

    save_dir = Path("../input/guit_submission/avg_pool")
    save_name = model_name + "_" + pretrained + ".pt"
    saved_model_path = save_dir / save_name
    # saved_model_path = Path(CLIP_SAVE_NAME)
    if model_name == "ViT-L-14":
        logger.info("use special mean, std")
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        mean = CLIP_INPUT_MEAN
        std = CLIP_INPUT_STD

    # jit model save
    baseline_transform = Transform(image_size=CLIP_INPUT_SIZE, mean=mean, std=std)
    baseline_encoder = ClipEncoder(model_name=model_name, pretrained=pretrained)
    model_with_wrapper = Embed64Wrapper(
        image_encoder=baseline_encoder, transform=baseline_transform
    )
    save_model_as_jit(model=model_with_wrapper, saved_model_path=saved_model_path)

    # test saved jit model
    embedding_fn_with_wrapper = load_jit_model(saved_model_path=saved_model_path)
    embedding_with_wrapper = extract_embedding(
        embedding_fn=embedding_fn_with_wrapper, image_path=SAMPLE_IMG_PATH
    )
    logger.info(
        f"Output embedding: {embedding_with_wrapper.shape}, {embedding_with_wrapper.dtype}, \n {embedding_with_wrapper}"
    )
    test_embedding_format(embedding=embedding_with_wrapper)


if __name__ == "__main__":
    main()
