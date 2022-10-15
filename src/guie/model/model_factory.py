import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.common.dataset.clip_embedding.fit_embedding_size import (
    convert_embed_size_with_average_pooling,
)
from src.common.dataset.image_transform import IN_1K_IMG_MEAN, IN_1K_IMG_STD
from src.guie.model.clip.constant import CLIP_EMBED_DIM, CLIP_INPUT_MEAN, CLIP_INPUT_STD
from src.guie.model.clip.huggingface_style_modeling import ClipForCompressedEmbed
from src.guie.model.ensemble.simple_concat import ENSEMBLE_SINGLE
from src.guie.model.head.embedding import MLPCompressor, SimpleHead, TransformerCompress
from src.guie.model.losses.loss import ReconstructionLoss
from src.guie.model.multi_domain.multi_domain_learning import MultiDomainArcFace, md_embed_mode
from src.guie.model.timm_models.metric_learning import ArcFaceWithTimm
from src.guie.submission.interface import EMBED_DIM_64

logger = logging.getLogger(__name__)

SUBMISSION_POOL_NAME = "compress_module"
IMAGE_HEAD_NAME = "image_head"


def create_model_for_arcface(
    model_config: DictConfig,
    ckpt_path: Optional[Path] = None,
    domain_to_num_labels: Optional[Dict[str, int]] = None,
) -> nn.Module:

    precomputed_class_centers = None
    if model_config.precomputed_class_centers_path is not None:
        logger.info(
            f"load precomputed_class_centers from {model_config.precomputed_class_centers_path}"
        )
        precomputed_class_centers = np.load(model_config.precomputed_class_centers_path)
        precomputed_class_centers = convert_embed_size_with_average_pooling(
            embeddings=precomputed_class_centers, target_size=model_config.embed_dim
        )

    logger.info(f"create {model_config.architecture} model")
    if model_config.architecture == "multi_domain":
        distill_cos_loss = ReconstructionLoss()
        assert domain_to_num_labels is not None
        model = MultiDomainArcFace(
            timm_model_name=model_config.timm_model_name,
            is_clip_backbone=model_config.is_clip_backbone,
            model_name=model_config.open_clip_model_name,
            pretrained=model_config.open_clip_pretrained,
            ensemble_mode=model_config.get("ensemble_mode", ENSEMBLE_SINGLE),
            use_weight_averaging=model_config.get("use_weight_averaging", False),
            weight_averaging_start_steps=model_config.get("weight_averaging_start_steps", None),
            num_head_layers=model_config.num_head_layers,
            domain_to_num_labels=domain_to_num_labels,
            domain_cls_loss_weight=model_config.domain_cls_loss_weight,
            num_sub_centers=model_config.num_sub_centers,
            arcface_scale=model_config.arcface_scale,
            embed_dim=model_config.embed_dim,
            embed_mode=model_config.get("embed_mode", md_embed_mode.SEPARATE),
            use_two_layer_on_projection=model_config.use_two_layer_on_projection,
            margin=model_config.arcface_margin,
            precomputed_class_centers=precomputed_class_centers,
            drop_rate=model_config.drop_out_rate,
            drop_path_rate=model_config.drop_path_rate,
            is_freeze_backbone=model_config.is_freeze_backbone,
            teacher_clip_model=model_config.teacher_clip_model,
            teacher_pretrained=model_config.teacher_pretrained,
            class_loss_weight=model_config.class_loss_weight,
            distill_loss_weight=model_config.distill_loss_weight,
            distill_cos_loss_weight=model_config.distill_cos_loss_weight,
            distill_cos_loss_fn=distill_cos_loss,
            use_binary_teacher=model_config.use_binary_teacher,
        )
    elif model_config.architecture == "single_domain":
        distill_cos_loss = ReconstructionLoss()
        model = ArcFaceWithTimm(
            timm_model_name=model_config.timm_model_name,
            num_sub_centers=model_config.num_sub_centers,
            arcface_scale=model_config.arcface_scale,
            embed_dim=model_config.embed_dim,
            use_two_layer_on_projection=model_config.use_two_layer_on_projection,
            margin=model_config.arcface_margin,
            precomputed_class_centers=precomputed_class_centers,
            drop_rate=model_config.drop_out_rate,
            drop_path_rate=model_config.drop_path_rate,
            is_freeze_backbone=model_config.is_freeze_backbone,
            teacher_clip_model=model_config.teacher_clip_model,
            teacher_pretrained=model_config.teacher_pretrained,
            class_loss_weight=model_config.class_loss_weight,
            distill_loss_weight=model_config.distill_loss_weight,
            distill_cos_loss_weight=model_config.distill_cos_loss_weight,
            distill_cos_loss_fn=distill_cos_loss,
            use_binary_teacher=model_config.use_binary_teacher,
        )
    else:
        raise ValueError(f"Unsupported architecture, {model_config.architecture}")

    if ckpt_path is not None:
        load_ckpt(model=model, ckpt_path=ckpt_path)
    return model


def create_model_for_wit(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_e16",
    ckpt_path: Optional[Path] = None,
    compressor_name: str = "linear",
    use_baseline: bool = False,
    transformer_d_model: int = 64,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:

    logger.info(f"use_{compressor_name} compressor")
    if compressor_name == "transformer":
        compress_module = TransformerCompress(
            in_features=CLIP_EMBED_DIM,
            out_features=EMBED_DIM_64,
            d_model=transformer_d_model,
            use_baseline=use_baseline,
            device=device,
        )
    elif compressor_name == "mlp":
        compress_module = MLPCompressor(
            in_features=CLIP_EMBED_DIM,
            out_features=EMBED_DIM_64,
            use_baseline=use_baseline,
            device=device,
        )
    elif compressor_name == "linear":
        compress_module = None
    else:
        raise RuntimeError(f"Not supported compressor_name:{compressor_name}")

    image_head = SimpleHead(
        in_features=CLIP_EMBED_DIM,
        out_features=EMBED_DIM_64,
        compress_module=compress_module,
        device=device,
    )
    text_head = image_head
    loss_fn = ReconstructionLoss()

    model = ClipForCompressedEmbed(
        image_head=image_head,
        text_head=text_head,
        model_name=model_name,
        pretrained=pretrained,
        loss_fn=loss_fn,
        device=device,
    )
    if ckpt_path is not None:
        load_ckpt(model=model, ckpt_path=ckpt_path)

    model.freeze_clip()

    # submission requirement
    assert hasattr(model, IMAGE_HEAD_NAME)
    assert hasattr(getattr(model, IMAGE_HEAD_NAME), SUBMISSION_POOL_NAME)
    return model


def get_submission_pool_head(model: nn.Module) -> nn.Module:
    return getattr(getattr(model, IMAGE_HEAD_NAME), SUBMISSION_POOL_NAME)


def load_ckpt(model: nn.Module, ckpt_path: Path) -> None:
    logger.info(f"loading pretrained weights from {str(ckpt_path)}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
    # which takes *args instead of **kwargs
    load_result = model.load_state_dict(state_dict, strict=False)
    if len(load_result.missing_keys) > 0:
        logger.warn(f"Missing weights are found when loading, {load_result.missing_keys}")
    if len(load_result.unexpected_keys) > 0:
        logger.warn(f"Unexpected weights were found when loading, {load_result.unexpected_keys}")


def get_model_input_factor(
    is_clip: bool, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s32b_b82k"
) -> Tuple[float, float]:
    if is_clip:
        if model_name == "ViT-L-14":
            assert pretrained == "laion2b_s32b_b82k"
            logger.warn("CLIP use special input mean, std: assert using laion2b_s32b_b82k weights")
            input_mean = (0.5, 0.5, 0.5)
            input_std = (0.5, 0.5, 0.5)
        else:
            logger.info("CLIP use normal input_mean, std")
            input_mean = CLIP_INPUT_MEAN
            input_std = CLIP_INPUT_STD
    else:
        input_mean = IN_1K_IMG_MEAN
        input_std = IN_1K_IMG_STD

    logger.info(f" Input mean: {input_mean}, Input std:{input_std}")
    return input_mean, input_std
