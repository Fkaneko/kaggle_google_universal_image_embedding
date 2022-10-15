import json
import logging
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import TrainingArguments, set_seed

from src.common.utils import set_logger
from src.guie.dataset.food_101.evaluation_runner import eval_food_101
from src.guie.dataset.glr.evaluation_runner import eval_glr
from src.guie.dataset.in_shop.evaluation_runner import eval_in_shop
from src.guie.dataset.met.evaluation_runner import eval_met
from src.guie.dataset.multi_domain.constants import md_const
from src.guie.dataset.objectnet.evaluation_runner import eval_objectnet
from src.guie.dataset.products_10k.evaluation_runner import eval_products_10k
from src.guie.dataset.sop.evaluation_runner import eval_sop
from src.guie.model.model_factory import get_model_input_factor
from src.guie.submission.conf.default import SubmissionInput, parse_submission_conf
from src.guie.submission.interface import Transform
from src.guie.submission.multi_domain_model import MultiDomainArcfaceEncoder

logger = logging.getLogger(__name__)


def run_evals_with_sub_conf(
    sub_input: SubmissionInput,
    batch_size: int = 256,
    num_workers: int = 16,
) -> dict:
    input_mean, input_std = get_model_input_factor(
        is_clip=sub_input.model_conf.is_clip_backbone,
        model_name=sub_input.model_conf.open_clip_model_name,
        pretrained=sub_input.model_conf.open_clip_pretrained,
    )

    image_transformations = Transform(
        image_size=tuple(sub_input.model_conf.input_size), mean=input_mean, std=input_std
    )

    model = MultiDomainArcfaceEncoder(
        model_config=sub_input.model_conf,
        ckpt_path=sub_input.ckpt_path,
        domain_to_num_labels=sub_input.domain_to_num_labels,
        # use_original_embedding=True,
        **(sub_input.sub_conf),
    )
    eval_results = {}

    model.cuda()
    model.eval()

    dataset_conf = sub_input.conf.dataset

    dataset_conf = OmegaConf.load("./src/guie/conf/dataset/downsampling_5.yaml")

    # prepare datalaoder
    image_transformations = torch.jit.script(image_transformations)
    eval_results["sop"] = eval_sop(
        # data_dir=Path(sub_input.conf.dataset.sop_dir),
        data_dir=Path("../input/Stanford_Online_Products"),
        image_transformations=image_transformations,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        # num_query_data=dataset_conf.sop_num_query_data,
        num_query_data=1000,
        num_queries_per_class=1,
        # num_queries_per_class=dataset_conf.sop_num_queries_per_class,
        num_index_data=None,
        knn_samples=md_const.KNN_SAMPLES,
        metric_mode=md_const.METRIC_MODE,
        # num_index_data=dataset_conf.sop_num_index_data,
        # num_query_data=dataset_conf.sop_num_query_data,
        # arrow_dir=arrow_dir,
        arrow_dir=Path(dataset_conf.sop_arrow_dir),
        update_eval_dataset=False,
        # update_eval_dataset=dataset_conf.up
    )
    eval_results["in_shop"] = eval_in_shop(
        # data_dir=Path(sub_input.conf.dataset.in_shop_dir),
        data_dir=Path("../input/DeepFashion/in_shop"),
        image_transformations=image_transformations,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        num_query_data=dataset_conf.in_shop_num_query_data,
        num_index_data=None,
        knn_samples=md_const.KNN_SAMPLES,
        metric_mode=md_const.METRIC_MODE,
        arrow_dir=Path(dataset_conf.in_shop_arrow_dir),
        update_eval_dataset=False,
    )

    eval_results["food_101"] = eval_food_101(
        # data_dir=Path(sub_input.conf.dataset.food_101_dir),
        data_dir=Path("../input/food-101"),
        image_transformations=image_transformations,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        num_index_data=None,
        num_query_data=500,
        num_queries_per_class=2,
        knn_samples=md_const.KNN_SAMPLES,
        metric_mode=md_const.METRIC_MODE,
        # num_index_data=dataset_conf.food_101_num_index_data,
        # num_query_data=dataset_conf.food_101_num_query_data,
        # num_queries_per_class=dataset_conf.food_101_num_queries_per_class,
        arrow_dir=Path(dataset_conf.food_101_arrow_dir),
        # update_eval_dataset=dataset_conf.food_101_update_eval_dataset
        update_eval_dataset=False,
    )

    eval_results["products_10k"] = eval_products_10k(
        # data_dir=Path(sub_input.conf.dataset.products_10k_dir),
        data_dir=Path("../input/products_10k"),
        image_transformations=image_transformations,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        num_index_data=dataset_conf.products_10k_num_index_data,
        num_query_data=dataset_conf.products_10k_num_query_data,
        num_queries_per_class=2,
        knn_samples=md_const.KNN_SAMPLES,
        metric_mode=md_const.METRIC_MODE,
        arrow_dir=Path(dataset_conf.products_10k_arrow_dir),
        update_eval_dataset=False,
        # num_queries_per_class=dataset_conf.products_10k_num_queries_per_class,
    )

    eval_results["met"] = eval_met(
        data_dir=Path(sub_input.conf.dataset.met_dir),
        image_transformations=image_transformations,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        num_index_data=None,
        num_query_data=None,
        num_queries_per_class=None,
        knn_samples=md_const.KNN_SAMPLES,
        metric_mode=md_const.METRIC_MODE,
        # num_index_data=dataset_conf.met_num_index_data,
        # num_query_data=dataset_conf.met_num_query_data,
        # num_queries_per_class=dataset_conf.met_num_queries_per_class,
        arrow_dir=Path(dataset_conf.met_arrow_dir),
        # update_eval_dataset=dataset_conf.met_update_eval_dataset
        update_eval_dataset=False,
    )

    eval_results["objectnet"] = eval_objectnet(
        # data_dir=Path(sub_input.conf.dataset.objectnet_dir),
        data_dir=Path("../input/objectnet/objectnet-1.0"),
        image_transformations=image_transformations,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        num_index_data=None,
        num_query_data=500,
        num_queries_per_class=2,
        knn_samples=md_const.KNN_SAMPLES,
        metric_mode=md_const.METRIC_MODE,
        # num_index_data=dataset_conf.objectnet_num_index_data,
        # num_query_data=dataset_conf.objectnet_num_query_data,
        # num_queries_per_class=dataset_conf.objectnet_num_queries_per_class,
        # arrow_dir=arrow_dir,
        arrow_dir=Path(dataset_conf.objectnet_arrow_dir),
        update_eval_dataset=False,
        # update_eval_dataset=dataset_conf.up
    )

    num_index_data = 60000
    mean_ap_public, mean_ap_private, mean_ap = eval_glr(
        data_dir=Path(dataset_conf.glr_data_dir),
        image_transformations=image_transformations,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        num_index_data=num_index_data,
        arrow_dir=Path(dataset_conf.glr_eval_arrow_dir),
        update_eval_dataset=dataset_conf.update_glr_eval_dataset,
        knn_samples=md_const.KNN_SAMPLES,
        metric_mode=md_const.METRIC_MODE,
    )
    eval_results["glr"] = mean_ap
    return eval_results


def main() -> None:

    set_seed(TrainingArguments.seed)
    set_logger(logging.getLogger())
    batch_size = 160
    num_workers = 6
    random_model = False

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

    eval_targets = {
        "vit_base": unified_dsample20_class_less_further_sma(),
        "vit_base_less": unified_dsample20_class_less_sma(),
        "vit_l": unified_final_vit_l(),
        "first": unified_final_vit_h(),
        "second": unified_final_second_vit_h(),
        "third": unified_final_third_vit_h(),
    }

    metrics_res = {}
    for eval_name, sub_conf in eval_targets.items():
        sub_input = parse_submission_conf(sub_conf=sub_conf, random_model=random_model)
        num_classes = sub_input.domain_to_num_labels[md_const.OTHER.name]
        eval_results = run_evals_with_sub_conf(
            sub_input=sub_input, batch_size=batch_size, num_workers=num_workers
        )
        eval_results["num_classes"] = int(num_classes)
        metrics_res[eval_name] = eval_results
        torch.cuda.empty_cache()

    with open("./eval_results.json", "w") as fp:
        json.dump(metrics_res, fp, indent=4)
    df = pd.DataFrame(metrics_res).T
    print(df.head())
    df.to_csv("./eval_results.csv", index=False)


if __name__ == "__main__":
    main()
