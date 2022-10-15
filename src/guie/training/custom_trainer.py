import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import evaluate
import numpy as np
import wandb
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix
from torch import nn
from transformers import Trainer

from src.guie.dataset.food_101.constants import food_101_const
from src.guie.dataset.food_101.evaluation_runner import prepare_food_101_eval_input
from src.guie.dataset.glr.build_dataset import get_dataloader, get_glr_test_dataset
from src.guie.dataset.glr.constants import glr_const
from src.guie.dataset.glr.glr_metrics.evaluation import (
    EmbedOutputs,
    calc_metric,
    generate_retrieval_predictions,
)
from src.guie.dataset.in_shop.constants import in_shop_const
from src.guie.dataset.in_shop.evaluation_runner import prepare_in_shop_eval_input
from src.guie.dataset.met.constants import met_const
from src.guie.dataset.met.evaluation_runner import prepare_met_eval_input
from src.guie.dataset.multi_domain.constants import md_const
from src.guie.dataset.objectnet.constants import objectnet_const
from src.guie.dataset.objectnet.evaluation_runner import prepare_objectnet_eval_input
from src.guie.dataset.products_10k.constants import products_10k_const
from src.guie.dataset.products_10k.evaluation_runner import prepare_products_10k_eval_input
from src.guie.dataset.sop.constants import sop_const
from src.guie.dataset.sop.evaluation_runner import prepare_sop_eval_input
from src.guie.submission.conf.default import CheckPointHandler

logger = logging.getLogger(__name__)


@dataclass
class DatasetMeta:
    key: str
    const: Any
    metric_weight: float


@dataclass(frozen=True)
class RetrievalEvalTargets:
    GLR: DatasetMeta = DatasetMeta(key="glr", const=glr_const, metric_weight=1.2)
    PRODUCTS_10K: DatasetMeta = DatasetMeta(
        key="products_10k", const=products_10k_const, metric_weight=1.0
    )
    IN_SHOP: DatasetMeta = DatasetMeta(key="in_shop", const=in_shop_const, metric_weight=1.0)
    SOP: DatasetMeta = DatasetMeta(key="sop", const=sop_const, metric_weight=0.8)
    MET: DatasetMeta = DatasetMeta(key="met", const=met_const, metric_weight=1.0)
    FOOD_101: DatasetMeta = DatasetMeta(key="food_101", const=food_101_const, metric_weight=0.35)
    OBJECT_NET: DatasetMeta = DatasetMeta(key="objectnet", const=objectnet_const, metric_weight=1.0)


class MultiDomainEvalTrainer(Trainer):
    def __init__(
        self,
        dataset_conf: DictConfig,
        data_dir: Path,
        image_transformations: nn.Module,
        batch_size: int = 256,
        arrow_dir: Optional[Path] = None,
        update_glr_eval_dataset: bool = False,
        num_workers: int = 6,
        num_index_data: int = 60000,
        dim_reduction_mapper: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        *args,
        **kwargs,
    ) -> None:
        kwargs["compute_metrics"] = self.compute_metrics_glr
        super().__init__(*args, **kwargs)

        # glr eval
        self.do_eval_glr = dataset_conf.do_eval_glr

        # manually save ckpt for simple moving average weight
        self.ckpt_handler = CheckPointHandler(
            save_dir=Path(kwargs["args"].output_dir), keep_only_latest=True
        )

        glr_dataset, glr_solution_index = get_glr_test_dataset(
            data_dir=data_dir,
            image_transformations=image_transformations,
            num_index_data=num_index_data,
            arrow_dir=arrow_dir,
            update_eval_dataset=update_glr_eval_dataset,
        )
        logger.info(f"GLR evaluation dataset: {glr_dataset}")
        self.glr_index_image_dataloader = get_dataloader(
            ds=glr_dataset[glr_const.INDEX], batch_size=batch_size, num_workers=num_workers
        )
        self.glr_test_image_dataloader = get_dataloader(
            ds=glr_dataset[glr_const.TEST], batch_size=batch_size, num_workers=num_workers
        )
        self.dim_reduction_mapper = dim_reduction_mapper
        self.glr_solution_index = glr_solution_index
        # glr objectnet

        # Load the accuracy metric from the datasets package
        self.metric_accuracy = evaluate.load("accuracy")
        common_kwargs_for_dataloader = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "image_transformations": image_transformations,
        }

        (
            self.in_shop_index_image_dataloader,
            self.in_shop_test_image_dataloader,
            self.in_shop_solution_index,
        ) = prepare_in_shop_eval_input(
            data_dir=Path(dataset_conf.in_shop_dir),
            arrow_dir=Path(dataset_conf.in_shop_arrow_dir),
            num_index_data=None,
            num_query_data=dataset_conf.in_shop_num_query_data,
            update_eval_dataset=False,
            **common_kwargs_for_dataloader,
        )
        (
            self.objectnet_index_image_dataloader,
            self.objectnet_test_image_dataloader,
            self.objectnet_solution_index,
        ) = prepare_objectnet_eval_input(
            data_dir=Path(dataset_conf.objectnet_dir),
            num_index_data=dataset_conf.objectnet_num_index_data,
            num_query_data=dataset_conf.objectnet_num_query_data,
            num_queries_per_class=dataset_conf.objectnet_num_queries_per_class,
            update_eval_dataset=dataset_conf.objectnet_update_eval_dataset,
            arrow_dir=Path(dataset_conf.objectnet_arrow_dir),
            **common_kwargs_for_dataloader,
        )
        (
            self.sop_index_image_dataloader,
            self.sop_test_image_dataloader,
            self.sop_solution_index,
        ) = prepare_sop_eval_input(
            data_dir=Path(dataset_conf.sop_dir),
            num_index_data=dataset_conf.sop_num_index_data,
            num_query_data=dataset_conf.sop_num_query_data,
            num_queries_per_class=dataset_conf.sop_num_queries_per_class,
            update_eval_dataset=dataset_conf.sop_update_eval_dataset,
            arrow_dir=Path(dataset_conf.sop_arrow_dir),
            **common_kwargs_for_dataloader,
        )
        (
            self.products_10k_index_image_dataloader,
            self.products_10k_test_image_dataloader,
            self.products_10k_solution_index,
        ) = prepare_products_10k_eval_input(
            data_dir=Path(dataset_conf.products_10k_dir),
            num_index_data=dataset_conf.products_10k_num_index_data,
            num_query_data=dataset_conf.products_10k_num_query_data,
            num_queries_per_class=dataset_conf.products_10k_num_queries_per_class,
            update_eval_dataset=dataset_conf.products_10k_update_eval_dataset,
            arrow_dir=Path(dataset_conf.products_10k_arrow_dir),
            **common_kwargs_for_dataloader,
        )
        (
            self.food_101_index_image_dataloader,
            self.food_101_test_image_dataloader,
            self.food_101_solution_index,
        ) = prepare_food_101_eval_input(
            data_dir=Path(dataset_conf.food_101_dir),
            num_index_data=dataset_conf.food_101_num_index_data,
            num_query_data=dataset_conf.food_101_num_query_data,
            num_queries_per_class=dataset_conf.food_101_num_queries_per_class,
            update_eval_dataset=dataset_conf.food_101_update_eval_dataset,
            arrow_dir=Path(dataset_conf.food_101_arrow_dir),
            **common_kwargs_for_dataloader,
        )
        (
            self.met_index_image_dataloader,
            self.met_test_image_dataloader,
            self.met_solution_index,
        ) = prepare_met_eval_input(
            data_dir=Path(dataset_conf.met_dir),
            num_index_data=dataset_conf.met_num_index_data,
            num_query_data=dataset_conf.met_num_query_data,
            num_queries_per_class=dataset_conf.met_num_queries_per_class,
            update_eval_dataset=dataset_conf.met_update_eval_dataset,
            arrow_dir=Path(dataset_conf.met_arrow_dir),
            **common_kwargs_for_dataloader,
        )
        self.return_embeddings = True
        self.retrieval_eval_targets = dataset_conf.retreival_eval_targets
        self.calc_distance_with_cpu = True

    def compute_metrics_glr(self, p) -> Dict[str, float]:

        # usual metrics
        # metric = self.compute_metrics_accuracy(p)
        metric = {}
        labels, label_domain_ids = p.label_ids
        logits, domain_cls_logits, top_5_accs_for_each_domain, loss_for_each_domain = p.predictions
        assert (labels.ndim == 1) & (label_domain_ids.ndim == 1)
        assert domain_cls_logits.ndim == 2
        assert top_5_accs_for_each_domain.shape[-1] == self.model.num_domains
        assert loss_for_each_domain.shape[-1] == self.model.num_domains
        wandb.log(
            {
                "conf_mat": wandb.plot.confusion_matrix(
                    y_true=label_domain_ids,
                    preds=np.argmax(domain_cls_logits, axis=1),
                    class_names=list(self.model.domain_to_num_labels.keys()),
                )
            }
        )
        conf_mat = confusion_matrix(
            y_true=label_domain_ids,
            y_pred=np.argmax(domain_cls_logits, axis=1),
            # labels=list(self.model.domain_to_num_labels.keys()),
        )
        logger.info(f"confusion matrix for domain cls \n {conf_mat}")

        for domain_name, _ in self.model.domain_to_num_labels.items():
            domain_id = md_const.domain_name_to_id[domain_name]
            # (bs, num_domains) slice
            top_5_acc = top_5_accs_for_each_domain[:, domain_id]
            loss = loss_for_each_domain[:, domain_id]
            # at some domain contain 0.0 element if there is no sample within a corresponding batch
            num_valid_batches = (loss > 0.0).sum()
            if num_valid_batches > 0:
                metric[f"top_5_acc_{domain_name}"] = top_5_acc.sum() / num_valid_batches
                metric[f"loss_{domain_name}"] = loss.sum() / num_valid_batches
            else:
                metric[f"top_5_acc_{domain_name}"] = 0.0
                metric[f"loss_{domain_name}"] = 0.0

        metric["domain_cls_acc"] = self.metric_accuracy.compute(
            predictions=np.argmax(domain_cls_logits, axis=1), references=label_domain_ids
        )["accuracy"]

        logger.info(f"use_weight_averaging for eval: {self.model.use_weight_averaging}")
        if self.model.use_weight_averaging:
            embedding_model = self.model.weight_averaging.network_sma.forward_embed
        else:
            embedding_model = self.model.forward_embed
        _generate_retrieval_predictions = partial(
            generate_retrieval_predictions,
            model=embedding_model,
            knn_samples=md_const.KNN_SAMPLES,
            return_embeddings=self.return_embeddings,
            calc_distance_with_cpu=self.calc_distance_with_cpu,
        )
        _calc_metric = partial(
            calc_metric, max_predictions=md_const.KNN_SAMPLES, metric_mode=md_const.METRIC_MODE
        )

        eval_target = RetrievalEvalTargets()
        each_embedding: Dict[str, EmbedOutputs] = {}
        each_solution: Dict[str, Dict[str, Dict[str, str]]] = {}
        for _, eval_meta in eval_target.__dict__.items():
            metric[f"mean_precision/{eval_meta.key}"] = 0.0
            if eval_meta.key not in self.retrieval_eval_targets:
                continue
            logger.info(f">> -- >> Start eval for: {eval_meta.key}")
            predictions, embed_outputs = _generate_retrieval_predictions(
                index_image_dataloader=getattr(self, eval_meta.key + "_index_image_dataloader"),
                test_image_dataloader=getattr(self, eval_meta.key + "_test_image_dataloader"),
                image_id_key=eval_meta.const.IMAGE_ID,
            )
            solution = getattr(self, eval_meta.key + "_solution_index")
            mean_precision = _calc_metric(predictions=predictions, solution=solution)
            each_embedding[eval_meta.key] = embed_outputs
            each_solution[eval_meta.key] = solution
            metric[f"mean_precision/{eval_meta.key}"] = mean_precision
            logger.info(f"<< -- << {eval_meta.key}:mean_precision: {mean_precision:>5.4f}")

        metric[f"mean_precision/{eval_target.GLR.key}"] = 0.0
        if self.do_eval_glr:
            # additional metrics
            # start prediction & evaluation
            predictions, embed_outputs = _generate_retrieval_predictions(
                index_image_dataloader=self.glr_index_image_dataloader,
                test_image_dataloader=self.glr_test_image_dataloader,
                image_id_key=glr_const.ID,
            )
            mean_ap_public = _calc_metric(
                predictions=predictions, solution=self.glr_solution_index.public_solution
            )
            mean_ap_private = _calc_metric(
                predictions=predictions, solution=self.glr_solution_index.private_solution
            )
            glr_solution_index = self.glr_solution_index.public_solution
            glr_solution_index.update(self.glr_solution_index.private_solution)

            each_embedding[eval_target.GLR.key] = embed_outputs
            each_solution[eval_target.GLR.key] = glr_solution_index
            mean_precision = (mean_ap_public + mean_ap_private) / 2.0
            metric[f"mean_precision/{eval_target.GLR.key}"] = mean_precision

        # # make index embed pool
        # index_embeds_pool = []
        # index_ids_pool = []
        # each_size = {}
        # for _eval_key in each_embedding.keys():
        #     index_embeds_pool.append(each_embedding[_eval_key].index_embeds)
        #     index_ids_pool.append(each_embedding[_eval_key].index_ids)
        #     each_size[_eval_key] = each_embedding[_eval_key].index_ids.shape[0]
        #     logger.debug(
        #         f"Num index data: {_eval_key:<15s}: {each_size[_eval_key]:<10d} {each_size[_eval_key]/248190:>4.3f}"
        #     )
        # index_embeds_pool = torch.concat(index_embeds_pool, dim=0)
        # index_ids_pool = np.concatenate(index_ids_pool, axis=0)

        # # calc distances each dataset test vs index pool
        # for _eval_key in each_embedding.keys():
        #     test_embeds = each_embedding[_eval_key].test_embeds
        #     test_ids = each_embedding[_eval_key].test_ids
        #     distances = torch.cdist(test_embeds, index_embeds_pool, p=2.0)
        #     if not self.calc_distance_with_cpu:
        #         distances = distances.cpu()

        #     distances = distances.numpy()
        #     # dist calc
        #     predicted_positions = np.argpartition(distances, md_const.KNN_SAMPLES, axis=1)[
        #         :, : md_const.KNN_SAMPLES
        #     ]
        #     predictions = {}
        #     for test_idx, query_id in enumerate(test_ids):
        #         nearest = [
        #             (index_ids_pool[j], distances[test_idx, j])
        #             for j in predicted_positions[test_idx]
        #         ]
        #         nearest.sort(key=lambda x: x[1])
        #         prediction = [str(index_id) for index_id, d in nearest]
        #         predictions[query_id] = prediction

        #     # metric calc
        #     mean_precision = calc_metric(
        #         predictions=predictions,
        #         solution=each_solution[_eval_key],
        #         max_predictions=md_const.KNN_SAMPLES,
        #         metric_mode=md_const.METRIC_MODE,
        #     )
        #     orig_mp = metric[f"mean_precision/{_eval_key}"]
        #     logger.info(
        #         f"Unified index mean ap {_eval_key}: {mean_precision:>5.4f} drop {orig_mp - mean_precision:>5.4f}"
        #     )
        #     metric[f"mean_precision/unified_{_eval_key}"] = mean_precision

        # final metric calc
        metric_final = 0.0
        num_domains = 0

        for _, eval_meta in eval_target.__dict__.items():
            if eval_meta.key not in list(each_embedding.keys()):
                continue
            metric_weight = eval_meta.metric_weight
            metric_final += metric[f"mean_precision/{eval_meta.key}"] * metric_weight
            num_domains += 1

        metric["mean_precision/unified_final"] = metric_final / num_domains
        logger.info(f"Unified final metric {metric_final / num_domains:>5.4f}")
        target_metric = "mean_precision/unified_final"
        if self.model.use_weight_averaging:
            state_dict = self.model.weight_averaging.last_state_dict
            training_steps = self.model.weight_averaging.global_iter
            self.ckpt_handler.save_ckpt_with_best_metric(
                current_metric=metric[target_metric],
                state_dict=state_dict,
                training_steps=training_steps,
            )

        return metric
