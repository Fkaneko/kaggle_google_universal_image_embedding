import logging
import math
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import open_clip
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

OPEN_CLIP_ENSEMBLE_POOL = (
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-B-32-quickgelu", "openai"),
    ("ViT-B-32", "laion400m_e32"),
)

# ensemble mode
ENSEMBLE_CONCAT = "concat"
ENSEMBLE_MEAN = "mean"
ENSEMBLE_SINGLE = "single"


class OpenCLIPWrapper(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_e16",
    ) -> None:
        super().__init__()

        model = open_clip.create_model(model_name=model_name, pretrained=pretrained)

        self.visual = model.visual
        self.backbone_dim = model.visual.conv1.out_channels

        # delete text part
        model.transformer = nn.Identity()
        model.token_embedding = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        feature_map = x[:, :, :].clone()

        x = self.visual.ln_post(x[:, 0, :])

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return feature_map, x


class SimpleEnsemble(nn.Module):
    def __init__(self, ensemble_mode: str = "concat_2") -> None:
        super().__init__()
        backbones = {}
        backbone_dims = []

        num_concat_models = 0
        if ensemble_mode.find("_") > -1:
            num_concat_models = int(ensemble_mode.split("_")[-1])
        else:
            logger.warn("loading old model for ensemble_mode, use num ensemble 2")
            num_concat_models = 2

        for model_idx, open_clip_model in enumerate(OPEN_CLIP_ENSEMBLE_POOL):
            model_name = open_clip_model[0]
            pretrained = open_clip_model[1]
            backbone = OpenCLIPWrapper(model_name, pretrained)
            backbones[model_name + "_" + pretrained] = backbone
            backbone_dims.append(backbone.visual.proj.data.shape[1])
            if model_idx >= (num_concat_models - 1):
                break

        if ensemble_mode.startswith(ENSEMBLE_MEAN):
            assert len(set(backbone_dims)) == 1
            backbone_dim = backbone_dims[0]
        elif ensemble_mode.startswith(ENSEMBLE_CONCAT):
            backbone_dim = sum(backbone_dims)

        logger.info(f"total ensemble {ensemble_mode} dim: {backbone_dim}")
        self.backbone_dim = backbone_dim
        self.ensemble_mode = ensemble_mode
        self.backbones = nn.ModuleDict(backbones)

        # for jit compile
        self.ENSEMBLE_MEAN = ENSEMBLE_MEAN
        self.ENSEMBLE_CONCAT = ENSEMBLE_CONCAT

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        outputs_feat_map = []
        for model_name, backbone in self.backbones.items():
            feature_map, x = backbone(pixel_values)
            x = F.normalize(x, p=2.0, dim=-1)
            outputs_feat_map.append(feature_map)
            outputs.append(x)

        if self.ensemble_mode.startswith(self.ENSEMBLE_MEAN):
            feature_map = torch.stack(outputs_feat_map, dim=1).mean(dim=1)
            x = torch.stack(outputs, dim=1).mean(dim=1)

        elif self.ensemble_mode.startswith(self.ENSEMBLE_CONCAT):
            feature_map = torch.concat(outputs_feat_map, dim=-1)
            x = torch.concat(outputs, dim=-1)

        return feature_map, x


if __name__ == "__main__":

    image_size = 224
    batch_size = 8
    num_channels = 3
    ensemble_mode = "mean_2"
    # ensemble_mode = "concat_2"

    model = SimpleEnsemble(ensemble_mode=ensemble_mode)
    pixel_values = torch.randn(
        (batch_size, num_channels, image_size, image_size), dtype=torch.float32
    )
    out = model(pixel_values)
    print(out[0].shape)
    print(out[1].shape)
