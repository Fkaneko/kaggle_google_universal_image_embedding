# adopted from https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text
import json
import logging
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
import wandb
from hydra import compose, initialize
from omegaconf import OmegaConf
from timm.optim import create_optimizer_v2
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from src.common.dataset.image_transform import InverseTransform, TrainTransform, ValTransform
from src.common.training.layer_wise_lr_decay import (
    NUM_LAYERS,
    LayerDecayValueAssigner,
    get_parameter_groups,
)
from src.common.utils import set_logger
from src.guie.dataset.multi_domain.build_dataset import (
    collate_fn,
    dataset_pipeline,
    load_multi_domain_dataset,
)
from src.guie.dataset.multi_domain.constants import md_const
from src.guie.model.model_factory import create_model_for_arcface, get_model_input_factor
from src.guie.submission.conf.default import DOMAIN_TO_NUM_LABELS_NAME
from src.guie.training.custom_trainer import MultiDomainEvalTrainer

logger = logging.getLogger(__name__)


def main() -> None:

    # 1. Parse input arguments
    with initialize(version_base="1.2", config_path="./src/guie/conf/"):
        conf = compose(config_name="baseline", overrides=sys.argv[1:])

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    parser = HfArgumentParser((TrainingArguments,))
    train_conf = OmegaConf.to_container(conf["train"])
    output_dir = Path(train_conf["output_dir"], timestamp)
    output_dir.mkdir(exist_ok=True)
    train_conf["output_dir"] = str(output_dir)
    training_args = parser.parse_dict(train_conf)[0]

    with (Path(training_args.output_dir) / "top_config.yaml").open("w") as fp:
        OmegaConf.save(config=conf, f=fp.name)
    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # 2. Setup logging

    set_logger(logger=logging.getLogger())
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    wandb.init(
        project=conf.ml_ops.project,
        name=conf.ml_ops.run_name,
        tags=conf.ml_ops.tags,
        group=conf.ml_ops.group,
        dir=training_args.output_dir,
    )
    wandb.config.update(OmegaConf.to_container(conf))

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    multi_domain_dataset, domain_to_num_labels = load_multi_domain_dataset(
        dataset_conf=conf.dataset
    )
    with Path(training_args.output_dir, DOMAIN_TO_NUM_LABELS_NAME).open("w") as fp:
        json.dump(domain_to_num_labels, fp)

    # 5. Load pretrained model, tokenizer, and feature extractor
    model = create_model_for_arcface(
        model_config=conf.model, domain_to_num_labels=domain_to_num_labels
    )
    input_mean, input_std = get_model_input_factor(
        is_clip=conf.model.is_clip_backbone,
        model_name=conf.model.open_clip_model_name,
        pretrained=conf.model.open_clip_pretrained,
    )

    image_transformations = {
        md_const.TRAIN: TrainTransform(
            image_size=tuple(conf.model.input_size),
            mean=input_mean,
            std=input_std,
            crop_scale_range=tuple(conf.dataset.crop_scale_range),
            rand_augment_magnitude=conf.dataset.rand_augment_magnitude,
        ),
        md_const.VAL: ValTransform(
            image_size=tuple(conf.model.input_size), mean=input_mean, std=input_std, crop_pct=1.0
        ),
    }
    multi_domain_dataset = dataset_pipeline(
        dataset=multi_domain_dataset,
        image_transformations=image_transformations,
        use_torch_jit=True,
        use_in1k_dataset=md_const.OTHER.name in conf.dataset.train_target_domain_names,
    )
    # 5.5 split dataset
    train_dataset = multi_domain_dataset[md_const.TRAIN]
    eval_dataset = multi_domain_dataset[md_const.VAL]
    # 5.6 visualize train_dataset
    image_inverse_transformations = InverseTransform(mean=input_mean, std=input_std)
    for idx, data in enumerate(train_dataset):
        plt.imshow(image_inverse_transformations(data["pixel_values"]))
        label_id = data[md_const.LABEL_ID]
        domain_id = data[md_const.DOMAIN_ID]
        title = f"{domain_id}:{label_id}"
        plt.title(title)
        plt.show()
        if idx > 5:
            break

    if list(conf.ml_ops.tags) == ["debug"]:
        logger.info("slice dataset for debug setting")
        train_dataset = train_dataset.select(range(0, 10000))
        eval_dataset = eval_dataset.shuffle()
        eval_dataset = eval_dataset.select(range(0, 512))

    # optim setting
    layer_decay = conf.model.layer_wise_lr_decay
    layer_decay_backbone_factor = conf.model.layer_decay_backbone_factor
    num_layers = NUM_LAYERS
    values = [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
    values = np.array(values)
    values[:-1] = values[:-1] * layer_decay_backbone_factor
    values = values.tolist()
    logger.info(f"layer_wise_lr_decay: {values}")

    assigner = LayerDecayValueAssigner(
        values=values, num_max_layer=len(values), is_clip_model=conf.model.is_clip_backbone
    )
    param_group_fn = partial(
        get_parameter_groups,
        weight_decay=training_args.weight_decay,
        skip_list=(),
        get_num_layer=assigner.get_layer_id,
        get_layer_scale=assigner.get_scale,
    )
    optim_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optim = create_optimizer_v2(
        model,
        opt="adamw",
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        param_group_fn=param_group_fn,
        **optim_kwargs,
    )
    scheduler = None  # use scheduler embed in optim

    # 8. Initialize our trainer
    glr_image_transformations = ValTransform(
        image_size=tuple(conf.model.input_size), mean=input_mean, std=input_std, crop_pct=1.0
    )
    glr_image_transformations = torch.jit.script(glr_image_transformations)
    trainer = MultiDomainEvalTrainer(
        dataset_conf=conf.dataset,
        data_dir=Path(conf.dataset.glr_data_dir),
        image_transformations=glr_image_transformations,
        batch_size=conf.dataset.glr_eval_batch_size,
        arrow_dir=Path(conf.dataset.glr_eval_arrow_dir),
        update_glr_eval_dataset=conf.dataset.update_glr_eval_dataset,
        num_workers=conf.dataset.preprocessing_num_workers,
        num_index_data=conf.dataset.glr_num_index_data,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        # compute_metrics=compute_metrics,
        data_collator=collate_fn,
        optimizers=(optim, scheduler),
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
