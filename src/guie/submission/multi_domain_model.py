import json
import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, set_seed

from src.common.utils import set_logger
from src.guie.dataset.multi_domain.constants import md_const
from src.guie.model.model_factory import (
    create_model_for_arcface,
    get_model_input_factor,
    get_submission_pool_head,
)
from src.guie.model.multi_domain.multi_domain_learning import md_embed_mode
from src.guie.submission.check_submission import test_embedding_format
from src.guie.submission.conf.default import parse_submission_conf
from src.guie.submission.interface import (
    SAMPLE_IMG_PATH,
    CenterCropTransform,
    Embed64Wrapper,
    Transform,
    extract_embedding,
    load_jit_model,
    save_model_as_jit,
)

logger = logging.getLogger(__name__)


class MultiDomainArcfaceEncoder(nn.Module):
    def __init__(
        self,
        domain_to_num_labels: Optional[Dict[str, int]] = None,
        model: Optional[nn.Module] = None,
        ckpt_path: Optional[Path] = None,
        model_config: Optional[DictConfig] = None,
        use_original_embedding: bool = False,
        domain_cls_scale: float = 3.0,
        use_soft_domain_embed: bool = False,
        avg_pool_domain_names: Optional[List[str]] = None,
        domain_cls_mappings: Optional[Dict[str, str]] = None,
        soft_embed_max_scale: Optional[float] = None,
        soft_embed_range: Optional[float] = None,
        delete_teacher: bool = False,
    ) -> None:
        super().__init__()
        if model is None:
            assert model_config is not None
            assert domain_to_num_labels is not None
            logger.info("create model...")
            model = create_model_for_arcface(
                model_config=model_config,
                ckpt_path=ckpt_path,
                domain_to_num_labels=domain_to_num_labels,
            )

        # remove teacher
        if delete_teacher:
            if hasattr(model, "teacher"):
                delattr(model, "teacher")

        # for open clip case
        self.is_clip_backbone = model.is_clip_backbone
        if self.is_clip_backbone:
            self.feature_extractor = model.model
            self.num_head_layers = model.num_head_layers
            if self.num_head_layers == 0:
                self.head = nn.Identity()
            else:
                self.head = model.head

        else:
            self.feature_extractor = nn.Sequential(
                model.model.forward_features,
                model.pool,
            )

        self.num_domains = len(model.domain_to_num_labels)
        self.domain_classifacation = model.domain_classifacation
        self.embed_mode = model.embed_mode

        assert (
            self.domain_classifacation[-1].weight.shape[0] == self.num_domains
        ), self.domain_classifacation[-1].weight.shape[0]

        avg_embed_dim = list(model.embed_dim_for_each_domain.values())[0]
        assert avg_embed_dim > 1
        self.avg_pool = nn.AdaptiveAvgPool1d(avg_embed_dim)

        self.projection = model.projection
        self.use_original_embedding = use_original_embedding

        self.domain_cls_scale = domain_cls_scale
        self.use_soft_domain_embed = use_soft_domain_embed
        self.avg_pool_domain_names = avg_pool_domain_names
        self.soft_embed_range = soft_embed_range
        self.soft_embed_max_scale = soft_embed_max_scale
        logger.info(
            f"SUB PARAM: domain_cls_scale: {domain_cls_scale}, use_soft_domain_embed:{use_soft_domain_embed}, use_avg_pool: {avg_pool_domain_names}"
        )
        if self.use_soft_domain_embed:
            logger.info(
                f"SOFT EMBED PARAM: soft_embed_range: {soft_embed_range}, soft_embed_max_scale:{soft_embed_max_scale}"
            )
        #  Dataclass is not supported on torch jit, so cast to python string
        self.other_domain_name = str(md_const.OTHER.name)
        self.domain_name_to_id = dict(md_const.domain_name_to_id)
        self.domain_cls_mappings = domain_cls_mappings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get embed
        if self.is_clip_backbone:
            student_feat_map, x = self.feature_extractor(x)
            if self.num_head_layers > 0:
                student_feat_map = student_feat_map.permute(1, 0, 2)  # NLD -> LND
                student_feat_map = self.head(student_feat_map)
                student_feat_map = student_feat_map.permute(1, 0, 2)  # LND -> NLD
                x = torch.concat([x, student_feat_map[:, 1:].mean(dim=1)], dim=1)

        # else:
        #     x = self.feature_extractor(x)

        x = x.view(x.shape[0], -1)
        if self.use_original_embedding:
            return F.normalize(x, dim=-1)

        # (bs, num_domains, embed_dim)
        each_domain_projections = self.projection(x)
        if self.embed_mode == "unified":
            _, embed = each_domain_projections.popitem()
            assert len(each_domain_projections) == 0
            return F.normalize(embed, dim=-1)
        elif self.embed_mode == "tile":
            embeds = []
            domain_cls_logits = self.domain_classifacation(x)
            domain_cls_proba = F.softmax(domain_cls_logits, dim=-1)
            for domain_name, each_domain_embed in each_domain_projections.items():
                each_domain_embed = F.normalize(each_domain_embed, dim=-1)
                if self.use_soft_domain_embed:
                    input_domain_id = self.domain_name_to_id[domain_name]
                    domain_cls_weight = domain_cls_proba[:, input_domain_id].unsqueeze(-1)
                    domain_cls_weight = (domain_cls_weight * self.soft_embed_range) + (
                        self.soft_embed_max_scale - self.soft_embed_range
                    )
                    each_domain_embed = each_domain_embed * domain_cls_weight

                embeds.append(each_domain_embed)
            return torch.concat(embeds, dim=-1)

        # (bs, 1, embed_dim)
        global_embed = self.avg_pool(x)
        # (bs, num_domains, embed_dim)
        # image_embeds_compressed[:, -1, :] = global_embed
        each_domain_projections[self.other_domain_name] = global_embed
        if self.avg_pool_domain_names is not None:
            for domain_name in each_domain_projections.keys():
                if domain_name in self.avg_pool_domain_names:
                    each_domain_projections[domain_name] = global_embed

        # (bs, num_domains, embed_dim)
        image_embeds_compressed = torch.stack(list(each_domain_projections.values()), dim=1)

        # target domain selection
        # (bs, num_domains)
        domain_cls_logits = self.domain_classifacation(x)
        target_domain_ids = domain_cls_logits.argmax(dim=-1)
        if self.domain_cls_mappings is not None:
            for input_domain_name, mapped_domain_name in self.domain_cls_mappings.items():
                input_domain_id = self.domain_name_to_id[input_domain_name]
                mapped_domain_id = self.domain_name_to_id[mapped_domain_name]
                target_domain_ids = torch.where(
                    target_domain_ids == input_domain_id, mapped_domain_id, target_domain_ids
                )

        domain_embed = F.one_hot(target_domain_ids, num_classes=self.num_domains)

        # slice target domain embed (bs, embed_dim)
        image_embeds_compressed = (image_embeds_compressed * domain_embed.unsqueeze(-1)).sum(1)
        image_embeds_compressed = F.normalize(image_embeds_compressed, dim=-1)

        # concat [global, each_domain]
        if self.use_soft_domain_embed:
            domain_embed = F.softmax(domain_cls_logits, dim=-1)
        domain_embed = domain_embed * self.domain_cls_scale
        image_embeds_compressed = torch.concat([domain_embed, image_embeds_compressed], dim=1)

        return image_embeds_compressed


def main() -> None:
    from src.guie.submission.conf.multi_eval_clip import (
        unified_dsample5,
        unified_dsample5_concat_2,
        unified_dsample5_concat_2_sma,
        unified_dsample5_concat_2_sma_center_crop,
        unified_dsample5_one_layer,
        unified_dsample10_middle_ckpt,
        unified_dsample20_class_less_further_concat_sma,
        unified_dsample20_class_less_further_head_sma,
        unified_dsample20_class_less_further_sma,
        unified_dsample20_class_less_sma,
        unified_dsample40_class_less_mild_sma,
        unified_final_second_vit_h,
        unified_final_third_vit_h,
        unified_final_vit_h,
        unified_final_vit_l,
    )
    from src.guie.submission.conf.tile import (
        tile_2domains_2projections,
        tile_3domains_2projections,
        tile_3domains_2projections_soft_embed,
        tile_3domains_2projections_soft_embed_weak,
    )
    from src.guie.submission.conf.unified import (
        unified_2domains_2projections,
        unified_3domains_2projections,
        unified_3domains_heavy_10k_2projections,
        unified_convN_t_3domains_heavy_10k,
        unified_convN_t_clip_teach_3domains_heavy_10k_strong_aug,
    )
    from src.guie.submission.conf.unified_clip import (
        unified_clip_1layer_head_3domains_strong_aug,
        unified_clip_2layer_head_3domains_strong_aug,
        unified_clip_4layer_head_3domains_strong_aug,
    )

    set_seed(TrainingArguments.seed)
    set_logger(logger=logging.getLogger())
    sub_conf = unified_final_third_vit_h()
    random_model = False
    sub_input = parse_submission_conf(sub_conf=sub_conf, random_model=random_model)

    image_size = sub_input.model_conf.input_size[0]
    input_mean, input_std = get_model_input_factor(
        is_clip=sub_input.model_conf.is_clip_backbone,
        model_name=sub_input.model_conf.open_clip_model_name,
        pretrained=sub_input.model_conf.open_clip_pretrained,
    )

    logger.info(f"sub conf, {sub_conf}")

    baseline_transform = Transform(
        image_size=tuple(sub_input.model_conf.input_size), mean=input_mean, std=input_std
    )
    baseline_encoder = MultiDomainArcfaceEncoder(
        model_config=sub_input.model_conf,
        ckpt_path=sub_input.ckpt_path,
        domain_to_num_labels=sub_input.domain_to_num_labels,
        **(sub_input.sub_conf),
    )

    batch_size = 8
    num_channels = 3
    pixel_values = torch.randn(
        (batch_size, num_channels, image_size, image_size), dtype=torch.float32
    )
    x = baseline_encoder(x=pixel_values)

    model_with_wrapper = Embed64Wrapper(
        image_encoder=baseline_encoder, transform=baseline_transform
    )
    save_model_as_jit(model=model_with_wrapper, saved_model_path=sub_input.saved_model_path)

    # test saved jit model
    embedding_fn_with_wrapper = load_jit_model(saved_model_path=sub_input.saved_model_path)
    embedding_with_wrapper = extract_embedding(
        embedding_fn=embedding_fn_with_wrapper, image_path=SAMPLE_IMG_PATH
    )
    logger.info(
        f"Output embedding: {embedding_with_wrapper.shape}, {embedding_with_wrapper.dtype}, \n {embedding_with_wrapper}"
    )
    test_embedding_format(embedding=embedding_with_wrapper)


if __name__ == "__main__":
    main()
