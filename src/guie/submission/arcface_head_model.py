import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from src.common.dataset.image_transform import IN_1K_IMG_MEAN, IN_1K_IMG_STD
from src.common.utils import set_logger
from src.guie.model.model_factory import create_model_for_arcface, get_submission_pool_head
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


class ArcfaceEncoder(nn.Module):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        ckpt_path: Optional[Path] = None,
        model_config: Optional[DictConfig] = None,
        use_original_embedding: bool = False,
        use_avg_pool: bool = False,
    ) -> None:
        super().__init__()
        if model is None:
            assert model_config is not None
            logger.info("create model...")
            model = create_model_for_arcface(model_config=model_config, ckpt_path=ckpt_path)

        self.feature_extractor = model.model
        if use_avg_pool:
            self.projection = nn.AdaptiveAvgPool1d(64)
        else:
            self.projection = model.projection
        self.use_original_embedding = use_original_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get embed
        x = self.feature_extractor(x)
        if self.use_original_embedding:
            return F.normalize(x, dim=-1)

        image_embeds_compressed = self.projection(x)
        image_embeds_compressed = F.normalize(image_embeds_compressed, dim=-1)
        return image_embeds_compressed


def main() -> None:

    set_logger(logger=logging.getLogger())
    save_name = "cos_distill"
    saved_model_path = Path(f"../input/guit_submission/in1k_{save_name}.pt")
    conf = OmegaConf.load("../working/guie/train_log/18-09-2022_23-58-52/top_config.yaml")
    logger.info(conf.keys())
    conf = conf.model
    ckpt_path = Path(
        "../working/guie/train_log/18-09-2022_23-58-52/checkpoint-4320/pytorch_model.bin"
    )

    # jit model save
    baseline_transform = Transform(
        image_size=tuple(conf.input_size), mean=IN_1K_IMG_MEAN, std=IN_1K_IMG_STD
    )

    baseline_encoder = ArcfaceEncoder(model_config=conf, ckpt_path=ckpt_path)
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
