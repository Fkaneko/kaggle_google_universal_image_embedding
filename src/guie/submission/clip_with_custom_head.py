import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.utils import set_logger
from src.guie.model.clip.constant import CLIP_INPUT_MEAN, CLIP_INPUT_SIZE, CLIP_INPUT_STD
from src.guie.model.model_factory import create_model_for_wit, get_submission_pool_head
from src.guie.submission.check_submission import test_embedding_format
from src.guie.submission.interface import (
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
        model: Optional[nn.Module] = None,
        ckpt_path: Optional[Path] = None,
        compressor_name: str = "linear",
        use_baseline: bool = True,
    ) -> None:
        super().__init__()
        if model is None:
            logger.info("create model...")
            model = create_model_for_wit(
                ckpt_path=ckpt_path, compressor_name=compressor_name, use_baseline=use_baseline
            )

        self.feature_extractor = model.clip.visual
        self.pool = get_submission_pool_head(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # get clip embed
        x = self.feature_extractor(x)
        x = F.normalize(x, dim=-1)

        # compress clip embed
        image_embeds_compressed = self.pool(x)
        image_embeds_compressed = F.normalize(image_embeds_compressed, dim=-1)
        return image_embeds_compressed


def main() -> None:

    compressor_name = "linear"
    ckpt_path = Path(
        "../working/guie/train_log/2022-09-11T03:42:25.136496/checkpoint-7850/pytorch_model.bin"
    )
    saved_model_path = Path(f"../input/guit_submission/wit_{compressor_name}.pt")
    use_baseline = True

    set_logger(logger=logging.getLogger())
    # saved_model_path = Path(CLIP_SAVE_NAME)
    # jit model save
    baseline_transform = Transform(
        image_size=CLIP_INPUT_SIZE, mean=CLIP_INPUT_MEAN, std=CLIP_INPUT_STD
    )
    baseline_encoder = ClipEncoder(
        ckpt_path=ckpt_path, use_baseline=use_baseline, compressor_name=compressor_name
    )
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
