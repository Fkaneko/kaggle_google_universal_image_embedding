import json
import logging
import shutil
from pathlib import Path
from typing import Dict, NamedTuple

import torch
from omegaconf import DictConfig, OmegaConf

from src.guie.dataset.multi_domain.constants import md_const
from src.guie.dataset.multi_domain.preprocess import get_domain_to_num_labels, load_cached_train_csv

logger = logging.getLogger(__name__)

NORMAL_CKPT = "checkpoint"
WEIGHT_AVERAGING_CKPT = "weight_averaging"
CKPT_NAME = "pytorch_model.bin"
DOMAIN_TO_NUM_LABELS_NAME = "domain_to_num_labels.json"


def default_submission() -> dict:
    return {
        "save_name": "saved_model.pt",
        "ckpt_path": None,
        "avg_pool_domain_names": None,
        "domain_cls_mappings": None,
        "use_soft_domain_embed": False,
        "soft_embed_max_scale": 1.0,
        "soft_embed_range": 1.0,
        "delete_teacher": True,
    }


def create_ckpt_path_from_dir(
    ckpt_dir: Path, ckpt_steps: int, is_weight_averaging: bool = False
) -> Path:
    if is_weight_averaging:
        return Path(ckpt_dir, f"{WEIGHT_AVERAGING_CKPT}-{str(ckpt_steps)}", CKPT_NAME)
    else:
        return Path(ckpt_dir, f"{NORMAL_CKPT}-{str(ckpt_steps)}", CKPT_NAME)


class CheckPointHandler:
    def __init__(self, save_dir: Path, keep_only_latest: bool = True) -> None:
        self.save_dir = save_dir
        self.keep_only_latest = keep_only_latest

        self.last_steps = None
        self.best_metric = -1.0

    def save_ckpt_with_best_metric(
        self, current_metric: float, state_dict: dict, training_steps: int
    ) -> None:
        if current_metric > self.best_metric:
            self.save_ckpt(state_dict=state_dict, training_steps=training_steps)
            self.best_metric = current_metric

    def save_ckpt(self, state_dict: dict, training_steps: int) -> None:

        # save weight
        ckpt_path = create_ckpt_path_from_dir(
            ckpt_dir=self.save_dir, ckpt_steps=training_steps, is_weight_averaging=True
        )
        ckpt_path.parent.mkdir(exist_ok=True)

        logger.info(f"save weights on: {str(ckpt_path)}")
        torch.save(state_dict, str(ckpt_path))

        if (self.last_steps is not None) & self.keep_only_latest:
            old_ckpt_path = create_ckpt_path_from_dir(
                ckpt_dir=self.save_dir, ckpt_steps=self.last_steps, is_weight_averaging=True
            )
            if old_ckpt_path.parent.exists():
                logger.info(f"delete old ckpt, {str(old_ckpt_path)}")
                shutil.rmtree(old_ckpt_path.parent)

        # update internal state
        self.last_steps = training_steps


class SubmissionInput(NamedTuple):
    model_conf: DictConfig
    sub_conf: dict
    ckpt_path: Path
    domain_to_num_labels: Dict[str, int]
    saved_model_path: Path
    conf: DictConfig


def parse_submission_conf(sub_conf: dict, random_model: bool = False) -> SubmissionInput:
    if random_model:
        # random sub
        saved_model_path = Path("./saved_model.pt")
        model_conf = OmegaConf.load("./src/guie/conf/model/baseline.yaml")
        ckpt_path = None
        domain_to_num_labels = {"glr": 1, "products_10k": 1, "other": 1000}
        sub_conf = {}
    # trained sub
    else:

        save_name = sub_conf.pop("save_name")
        ckpt_path = sub_conf.pop("ckpt_path")

        saved_model_path = Path(f"../input/guit_submission/multi_domain_{save_name}.pt")
        ckpt_dir = Path(*ckpt_path.parts[:-2])
        conf = OmegaConf.load(ckpt_dir / "top_config.yaml")
        domain_json_path = Path(ckpt_dir, DOMAIN_TO_NUM_LABELS_NAME)
        if domain_json_path.exists():
            with domain_json_path.open("r") as fp:
                domain_to_num_labels = json.load(fp)
        else:
            cache_train_csv_path = Path(conf.dataset.cache_train_csv_path)
            train_df = load_cached_train_csv(cache_path=cache_train_csv_path)
            domain_to_num_labels = get_domain_to_num_labels(train_df=train_df)
            domain_to_num_labels[md_const.OMNI_BENCH.name] = md_const.OMNI_BENCH.default_num_classes

        model_conf = conf.model

    return SubmissionInput(
        model_conf=model_conf,
        sub_conf=sub_conf,
        ckpt_path=ckpt_path,
        domain_to_num_labels=domain_to_num_labels,
        saved_model_path=saved_model_path,
        conf=conf,
    )
