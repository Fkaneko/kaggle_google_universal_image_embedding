import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import open_clip
import timm
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy
from torchvision.transforms import InterpolationMode, Normalize, Resize
from transformers.modeling_outputs import ModelOutput

from src.common.dataset.image_transform import IN_1K_IMG_MEAN, IN_1K_IMG_STD
from src.common.model.losses.arcface import ArcFace, ArcMarginProduct_subcenter
from src.guie.dataset.multi_domain.constants import md_const
from src.guie.model.clip.constant import CLIP_INPUT_MEAN, CLIP_INPUT_STD
from src.guie.model.ensemble.simple_concat import (
    ENSEMBLE_SINGLE,
    OpenCLIPWrapper,
    SimpleEnsemble,
)
from src.guie.model.simple_weight_average.swa import MovingAvg

logger = logging.getLogger(__name__)

IGNORED_LABEL_ID = -1

CLIP_HEAD_WIDTH = 64
# embed mode
EmbedMode = namedtuple(
    "EmbedMode",
    ["UNIFIED", "TILE", "SEPARATE"],
)
md_embed_mode = EmbedMode("unified", "tile", "separate")


@dataclass
class MultiDomainOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    domain_cls_logits: torch.FloatTensor = None
    top_5_accs_for_each_domain: Optional[torch.FloatTensor] = None
    loss_for_each_domain: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class MultiDomainProjection(nn.Module):
    def __init__(
        self,
        domain_to_num_labels: Dict[str, int],
        out_dims_for_each_domain: Dict[str, int],
        in_features: int = 512,
        num_domains: int = 3,
        use_two_layer_on_projection: bool = False,
    ):
        super().__init__()
        self.num_domains = num_domains
        self.domain_to_num_labels = domain_to_num_labels
        projectors = {}
        logger.info(f"use two layer projection: {use_two_layer_on_projection}")
        for domain_name, num_labels in domain_to_num_labels.items():
            # non target domain case
            if num_labels == 1:
                continue
            if use_two_layer_on_projection:
                # from https://github.com/microsoft/unilm/blob/74ae0ed7287d0eabb11c32922e8b88aef0f8c839/beit2/modeling_vqkd.py#L87-L90
                projectors[domain_name] = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=in_features),
                    nn.Tanh(),
                    nn.Linear(
                        in_features=in_features, out_features=out_dims_for_each_domain[domain_name]
                    ),
                )
            else:
                projectors[domain_name] = nn.Linear(
                    in_features=in_features, out_features=out_dims_for_each_domain[domain_name]
                )
        self.multi_projector = nn.ModuleDict(projectors)

    def forward(self, x: torch.FloatTensor) -> Dict[str, torch.Tensor]:
        bs, _ = x.shape
        domain_logits = {}
        for domain_name, projector in self.multi_projector.items():
            # (bs, out_features)
            this_domain_logits = projector(x)
            domain_logits[domain_name] = this_domain_logits

        return domain_logits


class ArcFaceSingleDomain(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        num_labels: int = 100,
        num_sub_centers: int = 1,
        precomputed_class_centers: Optional[np.ndarray] = None,
        arcface_scale: float = 30.0,
        margin: float = 0.3,
    ):
        super().__init__()
        self.arc_margin_product = ArcMarginProduct_subcenter(
            in_features=embed_dim,
            out_features=num_labels,
            k=num_sub_centers,
            precomputed_class_centers=precomputed_class_centers,
        )
        self.arcface = ArcFace(s=arcface_scale, margin=margin)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # arcface
        logits = self.arc_margin_product(x)

        # cast to fp32 for arcface calculation
        # orig_dtype = x.dtype
        logits_with_margin = self.arcface(logits=logits.float(), labels=labels)
        # logits = logits.to(orig_dtype)
        return logits, logits_with_margin


class Projection(nn.Module):
    def __init__(
        self,
        in_features: int = 768,
        middle_size: int = 512,
        out_features: int = 64,
        use_two_layer: bool = False,
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(64)
        if use_two_layer:
            self.fc1 = nn.Linear(in_features=in_features, out_features=middle_size)
            self.fc2 = nn.Identity()
            self.norm = nn.LayerNorm(middle_size)
            self.gelu = nn.GELU()
            self.fc3 = nn.Linear(in_features=middle_size, out_features=out_features)
        else:
            fc2_out = int(out_features * 4)
            self.fc1 = nn.Linear(in_features=in_features, out_features=middle_size)
            self.fc2 = nn.Linear(in_features=middle_size, out_features=fc2_out)
            self.norm = nn.LayerNorm(fc2_out)
            self.gelu = nn.GELU()
            self.fc3 = nn.Linear(in_features=fc2_out, out_features=out_features)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.fc1(x)
        baseline = self.avg_pool(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.fc3(x)
        x += baseline
        return x


class MultiDomainArcFace(nn.Module):
    """adopted from https://github.com/huggingface/transformers/blob/v4.21.3/src/transformers/models/convnext/modeling_convnext.py#L405"""

    def __init__(
        self,
        is_clip_backbone: bool = True,
        timm_model_name: str = "convnext_small_in22k",
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_e16",
        ensemble_mode: str = "concat",
        use_weight_averaging: bool = False,
        weight_averaging_start_steps: int = 10,
        num_head_layers: int = 0,
        domain_to_num_labels: dict = {"landmark": 8000},
        domain_cls_loss_weight: float = 10.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        problem_type: str = "single_label_classification",
        embed_mode: str = "separate",
        embed_dim: int = 64,
        use_two_layer_on_projection: bool = False,
        num_sub_centers: int = 1,
        arcface_scale: float = 30.0,
        margin: float = 0.3,
        precomputed_class_centers: Optional[np.ndarray] = None,
        is_freeze_backbone: bool = True,
        teacher_clip_model: Optional[str] = None,
        teacher_pretrained: Optional[str] = None,
        class_loss_weight: float = 1.0,
        distill_loss_weight: float = 0.5,
        distill_cos_loss_weight: float = 3.0,
        distill_cos_loss_fn: Optional[nn.Module] = None,
        use_binary_teacher: bool = False,
    ) -> None:
        super().__init__()

        self.domain_to_num_labels = domain_to_num_labels
        self.num_domains = len(domain_to_num_labels)
        self.problem_type = problem_type
        self.arcface_scale = arcface_scale
        self.embed_mode = embed_mode
        self.is_clip_backbone = is_clip_backbone

        self.is_vit_backbone = False
        input_size = 224
        if is_clip_backbone:
            logger.info("Create clip as backbone")
            if ensemble_mode.startswith(ENSEMBLE_SINGLE):
                self.model = OpenCLIPWrapper(model_name=model_name, pretrained=pretrained)
                backbone_dim = self.model.visual.proj.data.shape[1]
                feature_width = self.model.visual.conv1.out_channels
                num_heads = feature_width // CLIP_HEAD_WIDTH
            else:
                self.model = SimpleEnsemble(ensemble_mode=ensemble_mode)
                backbone_dim = self.model.backbone_dim
                feature_width = None
                num_heads = None
        else:
            logger.info("Create timm model as backbone")
            self.model = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=0,  # without final fc layer
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                global_pool="",
            )
            if timm_model_name.startswith("convnext"):
                backbone_dim = self.model.head.norm.normalized_shape[0]
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.backbone_grid_size = input_size // 32
            elif timm_model_name.startswith("beit"):
                self.is_vit_backbone = True
                backbone_dim = self.model.norm.normalized_shape[0]
                self.model.fc_norm = nn.LayerNorm(backbone_dim)
                self.pool = self.model.forward_head
                self.backbone_grid_size = input_size // 16
            else:
                raise ValueError(f"Unsupported timm model {timm_model_name}")

        self.head = None
        self.num_head_layers = num_head_layers
        if self.num_head_layers > 0:
            self.head = nn.Sequential(
                open_clip.model.LayerNorm(feature_width),
                open_clip.model.Transformer(
                    width=feature_width, layers=num_head_layers, heads=num_heads
                ),
                open_clip.model.LayerNorm(feature_width),
            )
            backbone_dim += feature_width

        logger.info(f"Multi domain embed mode: {embed_mode}")
        valid_domain_to_num_labels = {
            domain_name: num_labels
            for domain_name, num_labels in domain_to_num_labels.items()
            if num_labels != 1
        }
        valid_num_domains = len(valid_domain_to_num_labels)

        logger.info(f"valid {valid_num_domains} domains are detected")
        if embed_mode == md_embed_mode.UNIFIED:
            assert len(valid_domain_to_num_labels) == 1
            domain_name, _ = valid_domain_to_num_labels.popitem()
            self.embed_dim_for_each_domain = {domain_name: embed_dim}
        elif embed_mode == md_embed_mode.TILE:
            self.embed_dim_for_each_domain = {
                domain_name: embed_dim // valid_num_domains
                for domain_name in valid_domain_to_num_labels.keys()
            }
            # make 64 dim
            if valid_num_domains == 3:
                logger.info("add +1 dim on product 10k for tile mode with 3 domains")
                self.embed_dim_for_each_domain[md_const.PRODUCT_10K.name] += 1
            # assert int(self.embed_dim_for_each_domain * valid_num_domains) == embed_dim
        elif embed_mode == md_embed_mode.SEPARATE:
            self.embed_dim_for_each_domain = {
                domain_name: embed_dim - len(domain_to_num_labels)
                for domain_name in valid_domain_to_num_labels.keys()
            }

        else:
            raise ValueError(f"unsupported embed mode is given, {embed_mode}")

        self.projection = MultiDomainProjection(
            in_features=backbone_dim,
            out_dims_for_each_domain=self.embed_dim_for_each_domain,
            num_domains=self.num_domains,
            domain_to_num_labels=domain_to_num_labels,
            use_two_layer_on_projection=use_two_layer_on_projection,
        )
        self.domain_cls_loss_weight = domain_cls_loss_weight
        arcfaces = {}
        for domain_name, num_labels in domain_to_num_labels.items():
            # non target domain case
            if num_labels == 1:
                continue

            arcfaces[domain_name] = ArcFaceSingleDomain(
                embed_dim=self.embed_dim_for_each_domain[domain_name],
                num_labels=num_labels,
                num_sub_centers=num_sub_centers,
                precomputed_class_centers=precomputed_class_centers,
                arcface_scale=arcface_scale,
                margin=margin,
            )
        self.arcface_for_each_domain = nn.ModuleDict(arcfaces)
        self.domain_classifacation = nn.Sequential(
            nn.Linear(in_features=backbone_dim, out_features=self.num_domains)
        )

        if is_freeze_backbone:
            self.freeze_backbone()

        self.class_loss_weight = class_loss_weight
        self.teacher_clip_model = teacher_clip_model
        self.distill_loss_weight = distill_loss_weight
        self.distill_cos_loss_weight = distill_cos_loss_weight
        self.distill_cos_loss_fn = distill_cos_loss_fn
        self.use_binary_teacher = use_binary_teacher
        if teacher_clip_model:
            assert teacher_pretrained is not None
            self.teacher = OpenCLIPWrapper(model_name=model_name, pretrained=pretrained)
            # delete language part
            self.teacher.transformer = nn.Identity()
            self.teacher.token_embedding = nn.Identity()

            # whiten layer Feature Distill paper
            teacher_dim = self.teacher.visual.conv1.out_channels
            self.whiten = nn.LayerNorm(teacher_dim)
            self.whiten.weight.requires_grad = False
            self.whiten.bias.requires_grad = False
            # student projection layer Feature Distill paper, depthwise, resize, ch-projection
            self.student_projection = nn.Sequential(
                nn.Conv2d(
                    in_channels=backbone_dim,
                    out_channels=backbone_dim,
                    kernel_size=(3, 3),
                    groups=backbone_dim,
                    padding=1,
                ),
                Resize(
                    size=self.teacher.visual.grid_size, interpolation=InterpolationMode.BILINEAR
                ),
                nn.Conv2d(in_channels=backbone_dim, out_channels=teacher_dim, kernel_size=(1, 1)),
            )

            for param in self.teacher.parameters():
                param.requires_grad = False
            self.in1k_to_clip = self.__get_imagenet_norm_to_clip_norm()

        # # Initialize weights and apply final processing
        # self.post_init()
        self.use_weight_averaging = use_weight_averaging
        if use_weight_averaging:
            freeze_part_prefix = ("model",) if is_freeze_backbone else ()
            self.weight_averaging = MovingAvg(
                network=self,
                sma_start_iter=weight_averaging_start_steps,
                freeze_part_prefix=freeze_part_prefix,
            )

    def freeze_backbone(self) -> None:
        logger.info("freeze backbone pretrained weight")
        for param in self.model.parameters():
            param.requires_grad = False

    def __get_imagenet_norm_to_clip_norm(self) -> nn.Module:
        mean_clip = np.array(CLIP_INPUT_MEAN)
        std_clip = np.array(CLIP_INPUT_STD)

        mean = np.array(IN_1K_IMG_MEAN)
        std = np.array(IN_1K_IMG_STD)

        in1k_to_clip = Normalize(+(mean_clip - mean) / std, std_clip / std)
        return in1k_to_clip

    def forward_embed(
        self, pixel_values: torch.FloatTensor, l2_norm_return: bool = True
    ) -> torch.FloatTensor:
        if self.is_clip_backbone:
            student_feat_map, x = self.model(pixel_values)
        else:
            x = self.model.forward_features(pixel_values)
            student_feat_map = x.clone()
            x = self.pool(x).view(x.shape[0], -1)
            if self.is_vit_backbone:
                # remove cls token
                student_feat_map = student_feat_map[:, 1:]
                bs, _, ch = student_feat_map.shape
                # bs, ch, gird**2
                student_feat_map = student_feat_map.permute(0, 2, 1)
                student_feat_map = student_feat_map.reshape(
                    bs, ch, self.backbone_grid_size, self.backbone_grid_size
                )

        if self.num_head_layers > 0:
            student_feat_map = student_feat_map.permute(1, 0, 2)  # NLD -> LND
            student_feat_map = self.head(student_feat_map)
            student_feat_map = student_feat_map.permute(1, 0, 2)  # LND -> NLD
            x = torch.concat([x, student_feat_map[:, 1:].mean(dim=1)], dim=1)

        # multi domain projection
        each_domain_projections = self.projection(x)
        if l2_norm_return:
            if self.embed_mode == "unified":
                _, embed = each_domain_projections.popitem()
                assert len(each_domain_projections) == 0
                return F.normalize(embed, dim=-1)
            else:
                raise ValueError("Unsupported mode for forward_embed")
        else:
            return x, student_feat_map, each_domain_projections

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        label_domain_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultiDomainOutput]:
        r""" """
        if self.use_weight_averaging:
            if self.training:
                self.weight_averaging.update_sma()

        x, student_feat_map, each_domain_projections = self.forward_embed(
            pixel_values=pixel_values, l2_norm_return=False
        )
        domain_cls_logits = self.domain_classifacation(x)

        loss = 0.0 if labels is not None else None
        distill_loss, distill_cos_loss = None, None
        if self.teacher_clip_model is not None:
            clip_input = self.in1k_to_clip(pixel_values)
            with torch.no_grad():
                teacher_feat_map, teacher_embed = self.forward_clip_for_feature_map(
                    clip_vision_model=self.teacher.visual, x=clip_input
                )
                # drop cls token
                teacher_feat_map = teacher_feat_map[:, 1:]
                teacher_feat_map = self.whiten(teacher_feat_map)

            teacher_feat_map = teacher_feat_map.permute(0, 2, 1)  # shape = [*, width, grid ** 2]

            student_feat_map = self.student_projection(student_feat_map)
            student_feat_map = student_feat_map.reshape(
                student_feat_map.shape[0], student_feat_map.shape[1], -1
            )  # shape = [*, width, grid ** 2]

            distill_loss = F.smooth_l1_loss(
                input=student_feat_map, target=teacher_feat_map, beta=2.0
            )
            loss += self.distill_loss_weight * distill_loss

            if self.distill_cos_loss_fn is not None:
                student_compressed_out = F.normalize(
                    each_domain_projections[md_const.OTHER.name], p=2.0, dim=-1
                )
                teacher_out = F.normalize(teacher_embed, p=2.0, dim=-1)
                distill_cos_loss = self.distill_cos_loss_fn(
                    image_embeds_original=teacher_out,
                    text_embeds_original=teacher_out,
                    image_embeds_new=student_compressed_out,
                    text_embeds_new=student_compressed_out,
                )
                loss += self.distill_loss_weight * distill_cos_loss

        # arcface
        loss_for_each_domain = torch.zeros((1, self.num_domains)).to(x.device)
        top_5_accs = torch.zeros((1, self.num_domains)).to(x.device)
        if labels is not None:
            if self.embed_mode == md_embed_mode.UNIFIED:
                arcface = self.arcface_for_each_domain[md_const.OTHER.name]
                domain_logits, domain_logits_with_margin = arcface(
                    x=each_domain_projections[md_const.OTHER.name], labels=labels
                )
                num_unified_labels = self.domain_to_num_labels[md_const.OTHER.name]
                unified_loss = F.cross_entropy(
                    input=domain_logits_with_margin.view(-1, num_unified_labels),
                    target=labels.view(-1),
                    reduction="none",
                )
                unified_loss_mean = unified_loss.mean()
                loss += unified_loss_mean
                loss_for_each_domain[0, md_const.OTHER.id] += unified_loss_mean

                for domain in md_const.all_domains:
                    domain_mask = label_domain_ids == domain.id
                    num_valid_samples = domain_mask.sum()
                    if num_valid_samples == 0:
                        continue

                    this_domain_loss = (unified_loss * domain_mask).sum() / num_valid_samples
                    # acc calcs
                    this_domain_logits = domain_logits[domain_mask]
                    this_domain_labels = labels[domain_mask]
                    acc = accuracy(
                        preds=F.softmax(self.arcface_scale * this_domain_logits, dim=-1),
                        target=this_domain_labels,
                        top_k=5,
                        num_classes=num_unified_labels,
                    )
                    loss_for_each_domain[0, domain.id] += this_domain_loss.squeeze()
                    top_5_accs[0, domain.id] += acc

                # for other, unified class
                acc = accuracy(
                    preds=F.softmax(self.arcface_scale * domain_logits, dim=-1),
                    target=labels,
                    top_k=5,
                    num_classes=num_unified_labels,
                )
                top_5_accs[0, md_const.OTHER.id] += acc

            else:
                for domain_name, arcface in self.arcface_for_each_domain.items():
                    this_domain_num_labels = self.domain_to_num_labels[domain_name]
                    domain_id = md_const.domain_name_to_id[domain_name]

                    # for arcface calc clip max label is
                    this_domain_labels = torch.where(
                        label_domain_ids == domain_id, labels, IGNORED_LABEL_ID
                    )
                    if (this_domain_labels == IGNORED_LABEL_ID).all():
                        continue
                    domain_logits, domain_logits_with_margin = arcface(
                        x=each_domain_projections[domain_name], labels=this_domain_labels
                    )
                    this_domain_loss = F.cross_entropy(
                        input=domain_logits_with_margin.view(-1, this_domain_num_labels),
                        target=this_domain_labels.view(-1),
                        ignore_index=IGNORED_LABEL_ID,
                    )
                    loss += this_domain_loss.squeeze()
                    loss_for_each_domain[0, domain_id] += this_domain_loss.squeeze()
                    # acc calc
                    acc = accuracy(
                        preds=F.softmax(self.arcface_scale * domain_logits, dim=-1),
                        target=this_domain_labels,
                        top_k=5,
                        num_classes=this_domain_num_labels,
                        ignore_index=IGNORED_LABEL_ID,
                    )
                    top_5_accs[0, domain_id] += acc

            domain_cls_loss = F.cross_entropy(
                input=domain_cls_logits.view(-1, self.num_domains),
                target=label_domain_ids.view(-1),
            )
            loss += self.domain_cls_loss_weight * domain_cls_loss

        if not return_dict:
            output = (x, domain_cls_logits, top_5_accs)
            return ((loss,) + output) if loss is not None else output

        return MultiDomainOutput(
            loss=loss,
            logits=x,
            domain_cls_logits=domain_cls_logits,
            top_5_accs_for_each_domain=top_5_accs,
            loss_for_each_domain=loss_for_each_domain,
        )
