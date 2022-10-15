# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
from dataclasses import dataclass
from typing import List, Tuple

import timm

NUM_LAYERS = 12


@dataclass
class LayerPosition:
    is_backbone: bool
    stage_idx: int
    is_downsample: bool
    block_idx: int
    is_head: bool


def get_num_layer_for_clip(
    var_name: str,
    backbone_prefix: Tuple[str, ...] = ("model.", "visual."),
    num_max_layer: int = 14,
) -> int:

    is_backbone = False
    for prefix in backbone_prefix:
        if var_name.startswith(prefix):
            var_name = var_name[len(prefix) :]
            is_backbone = True

    # vit inside
    if is_backbone:
        if var_name in ("class_embedding", "positional_embedding"):
            return 0
        elif var_name.startswith("conv1"):
            return 0
        elif var_name.startswith("ln_pre"):
            return 0
        elif var_name.startswith("transformer"):
            layer_id = int(var_name.split(".")[2])
            return layer_id + 1
        elif var_name.startswith("ln_post"):
            return num_max_layer - 2
        elif var_name in ("proj"):
            return num_max_layer - 2

    return num_max_layer - 1


def parse_convnext_var_name(
    var_name: str,
    backbone_prefix: Tuple[str, ...] = ("model.",),
    stage_ids_pos: int = 1,
    block_pos: int = 2,
    block_idx_pos: int = 3,
) -> LayerPosition:

    stage_idx, block_idx = 0, 0
    is_downsample, is_backbone, is_head = False, False, False

    is_backbone_check = []
    for prefix in backbone_prefix:
        if var_name.startswith(prefix):
            var_name = var_name[len(prefix) :]
            is_backbone_check.append(True)
        else:
            is_backbone_check.append(False)

    is_backbone = any(is_backbone_check) | (len(backbone_prefix) == 0)
    if is_backbone:
        if var_name.startswith("stem"):
            pass
        elif var_name.startswith("stages"):
            var_name_split = var_name.split(".")
            stage_idx = int(var_name_split[stage_ids_pos])
            blocks_or_downsample = var_name_split[block_pos]
            is_downsample = blocks_or_downsample == "downsample"
            if not is_downsample:
                block_idx = int(var_name_split[block_idx_pos])
        else:
            is_head = True

    return LayerPosition(
        is_backbone=is_backbone,
        stage_idx=stage_idx,
        is_downsample=is_downsample,
        block_idx=block_idx,
        is_head=is_head,
    )


def get_num_layer_for_convnext(
    var_name: str,
    backbone_prefix: Tuple[str, ...] = ("model.",),
    stage_ids_pos: int = 1,
    block_pos: int = 2,
    block_idx_pos: int = 3,
):
    """
    Divide [3, 3, 27, 3] layers into 12 groups; each group is three
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """

    num_max_layer = 12
    var_layer_position = parse_convnext_var_name(
        var_name=var_name,
        backbone_prefix=backbone_prefix,
        stage_ids_pos=stage_ids_pos,
        block_pos=block_pos,
        block_idx_pos=block_idx_pos,
    )
    if var_layer_position.is_head:
        return num_max_layer + 1
    elif var_layer_position.is_downsample:
        stage_idx = var_layer_position.stage_idx
        if stage_idx == 0:
            layer_id = 0
        elif stage_idx == 1 or stage_idx == 2:
            layer_id = stage_idx + 1
        elif stage_idx == 3:
            layer_id = 12
        return layer_id

    elif var_layer_position.is_backbone:
        stage_idx = var_layer_position.stage_idx
        block_idx = var_layer_position.block_idx
        if stage_idx == 0 or stage_idx == 1:
            layer_id = stage_idx + 1
        elif stage_idx == 2:
            layer_id = 3 + block_idx // 3
        elif stage_idx == 3:
            layer_id = 12
        return layer_id
    else:
        return num_max_layer + 1


class LayerDecayValueAssigner(object):
    def __init__(
        self,
        values: List[float],
        backbone_prefix: Tuple[str, ...] = ("model.", "visual."),
        stage_ids_pos: int = 1,
        block_pos: int = 2,
        block_idx_pos: int = 3,
        num_max_layer: int = 14,
        is_clip_model: bool = True,
    ) -> None:

        """
        adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
        """
        self.values = values
        self.backbone_prefix = backbone_prefix
        self.stage_ids_pos = stage_ids_pos
        self.block_pos = block_pos
        self.block_idx_pos = block_idx_pos
        self.num_max_layer = num_max_layer
        self.is_clip_model = is_clip_model

    def get_scale(self, layer_id: int) -> float:
        return self.values[layer_id]

    def get_layer_id(self, var_name: str) -> int:
        if self.is_clip_model:
            return get_num_layer_for_clip(
                var_name=var_name,
                backbone_prefix=self.backbone_prefix,
                num_max_layer=self.num_max_layer,
            )
        else:
            return get_num_layer_for_convnext(
                var_name=var_name,
                backbone_prefix=self.backbone_prefix,
                stage_ids_pos=self.stage_ids_pos,
                block_pos=self.block_pos,
                block_idx_pos=self.block_idx_pos,
            )


def get_parameter_groups(
    model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None
):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


if __name__ == "__main__":
    num_layers = 12
    layer_decay = 0.8
    timm_model_name: str = "convnext_small"
    model = timm.create_model(
        timm_model_name,
        pretrained=False,
        # num_classes=0,
    )
    skip = ()
    weight_decay = 0.05

    values = list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
    assigner = LayerDecayValueAssigner(
        values=values,
        backbone_prefix=(),
    )
    parameters = get_parameter_groups(
        model,
        weight_decay,
        skip,
        get_num_layer=assigner.get_layer_id,
        get_layer_scale=assigner.get_scale,
    )
