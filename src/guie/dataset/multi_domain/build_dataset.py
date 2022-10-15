import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Image, concatenate_datasets
from omegaconf import DictConfig
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import pil_to_tensor

from src.common.dataset.read_image import convert_to_rgb
from src.guie.dataset.imagenet_1k.build_dataset import load_in_1k_dataset
from src.guie.dataset.imagenet_1k.constants import in_1k_const
from src.guie.dataset.imagenet_1k.parse_in1k_class_with_wordnet import find_ood_class_from_in1k

from .constants import md_const
from .preprocess import (
    cache_train_csv,
    get_domain_to_num_labels,
    keep_only_target_domain,
    load_cached_train_csv,
    load_train_csv,
    make_label_id_unified,
)

logger = logging.getLogger(__name__)


OMNI_BENCH_TRAIN_CSV = "omni_bench_df_train.csv"
OMNI_BENCH_VAL_CSV = "omni_bench_df_val.csv"


def transform_images(
    examples: dict,
    image_transformations: Callable[[torch.Tensor], torch.Tensor],
    use_in1k_dataset: bool = False,
) -> dict:
    if use_in1k_dataset:
        # here image_file is decoded as PIL.Image
        pixel_values = [
            pil_to_tensor(convert_to_rgb(pil_image)) for pil_image in examples[md_const.FILEPATH]
        ]
    else:
        pixel_values = [
            read_image(image_file, mode=ImageReadMode.RGB)
            for image_file in examples[md_const.FILEPATH]
        ]
    examples["pixel_values"] = [image_transformations(pixel_value) for pixel_value in pixel_values]
    return examples


def load_multi_domain_dataset(dataset_conf: DictConfig) -> Tuple[DatasetDict, Dict[str, int]]:
    # csv load
    cache_train_csv_path = Path(dataset_conf.cache_train_csv_path)
    cache_omni_bench_train_path = cache_train_csv_path.with_name(OMNI_BENCH_TRAIN_CSV)
    cache_omni_bench_val_path = cache_train_csv_path.with_name(OMNI_BENCH_VAL_CSV)
    if (not dataset_conf.update_cache_train_csv) & cache_train_csv_path.exists():
        train_df = load_cached_train_csv(cache_path=cache_train_csv_path)
        omni_bench_df_train = load_cached_train_csv(cache_path=cache_omni_bench_train_path)
        omni_bench_df_val = load_cached_train_csv(cache_path=cache_omni_bench_val_path)
    else:
        train_df, omni_bench_df_train, omni_bench_df_val = load_train_csv(dataset_conf=dataset_conf)
        cache_train_csv(df=train_df, cache_path=cache_train_csv_path)
        cache_train_csv(df=omni_bench_df_train, cache_path=cache_omni_bench_train_path)
        cache_train_csv(df=omni_bench_df_val, cache_path=cache_omni_bench_val_path)

    val_df = train_df.loc[train_df["fold"] == dataset_conf.val_fold, :]
    train_df = train_df.loc[train_df["fold"] != dataset_conf.val_fold, :]

    # add omni verse
    train_df = (
        pd.concat([train_df, omni_bench_df_train], axis=0).sample(frac=1.0).reset_index(drop=True)
    )
    val_df = pd.concat([val_df, omni_bench_df_val], axis=0).sample(frac=1.0).reset_index(drop=True)
    if dataset_conf.get("val_data_num", None) is not None:
        logger.warn(f"validation data only use : {dataset_conf.val_data_num}")
        val_df = val_df.sample(n=dataset_conf.val_data_num)

    logger.info(f"domain count train \n {train_df[md_const.DOMAIN_ID].value_counts().sort_index()}")
    logger.info(f"domain count val \n {val_df[md_const.DOMAIN_ID].value_counts().sort_index()}")

    # class info for each domain, this dict will be used for model creation
    domain_to_num_labels = get_domain_to_num_labels(train_df)

    # select training target domain
    train_df, val_df, domain_to_num_labels = keep_only_target_domain(
        train_df=train_df,
        val_df=val_df,
        target_domain_names=list(dataset_conf.train_target_domain_names),
        domain_to_num_labels=domain_to_num_labels,
    )

    # merge domain
    if dataset_conf.make_unified_label:
        assert md_const.OTHER.name not in dataset_conf.train_target_domain_names
        train_df, val_df, domain_to_num_labels = make_label_id_unified(
            train_df=train_df, val_df=val_df, domain_to_num_labels=domain_to_num_labels
        )

    train_dataset = Dataset.from_pandas(df=train_df)
    val_dataset = Dataset.from_pandas(df=val_df)
    dataset = DatasetDict(
        {
            md_const.TRAIN: train_dataset,
            md_const.VAL: val_dataset,
        }
    )
    # pandas dataframe index
    dataset = dataset.remove_columns(["fold"])

    if md_const.OTHER.name in dataset_conf.train_target_domain_names:
        dataset = dataset.cast_column(md_const.FILEPATH, Image(decode=True))

        logger.info("start loading other domain dataset...")
        # imagenet 1k
        in_1k_dataset = load_in_1k_dataset(
            cache_dir=Path(dataset_conf.imagenet_1k_dir),
            arrow_dir=Path(dataset_conf.imagenet_1k_arrow_dir),
            data_files=None,
        )

        # class filtering
        wordnet_df = find_ood_class_from_in1k()
        ood_labels = list(
            wordnet_df.loc[wordnet_df[in_1k_const.IS_OOD], in_1k_const.LABEL].to_numpy()
        )

        logger.info(f"before ood label sampling {in_1k_dataset}")
        in_1k_dataset = in_1k_dataset.filter(
            lambda x: x in ood_labels,
            input_columns=[in_1k_const.LABEL],
            # batched=True,
            num_proc=int(dataset_conf.preprocessing_num_workers),
        )
        logger.info(f"after ood label sampling {in_1k_dataset}")

        # rename col, label_id, filepath
        in_1k_dataset = in_1k_dataset.rename_column(in_1k_const.LABEL, md_const.LABEL_ID)
        in_1k_dataset = in_1k_dataset.rename_column(in_1k_const.IMAGE, md_const.FILEPATH)

        for data_split in [in_1k_const.TRAIN, in_1k_const.VAL]:
            # sampling data
            in_1k_dataset[data_split] = in_1k_dataset[data_split].shuffle()
            in_1k_dataset[data_split] = in_1k_dataset[data_split].select(
                range(0, in_1k_dataset[data_split].num_rows // 10)
            )
            # add domain id
            in_1k_domain = [md_const.OTHER.id] * in_1k_dataset[data_split].num_rows
            in_1k_dataset[data_split] = in_1k_dataset[data_split].add_column(
                name=md_const.DOMAIN_ID, column=in_1k_domain
            )

        # fix datatype
        for data_split in [in_1k_const.TRAIN, in_1k_const.VAL]:
            in_1k_dataset[data_split] = in_1k_dataset[data_split].cast(
                features=dataset[md_const.TRAIN].features,
                num_proc=int(dataset_conf.preprocessing_num_workers),
            )

        # concat
        for md_data_split, in_1k_data_split in zip(
            [md_const.TRAIN, md_const.VAL], [in_1k_const.TRAIN, in_1k_const.VAL]
        ):
            dataset[md_data_split] = concatenate_datasets(
                [dataset[md_data_split], in_1k_dataset[in_1k_data_split]], axis=0
            )
        # shuffle for val. metric evaluated average of batch-average
        dataset[md_const.VAL] = dataset[md_const.VAL].shuffle()

    return dataset, domain_to_num_labels


def dataset_pipeline(
    dataset: DatasetDict,
    image_transformations: Dict[str, torch.nn.Module],
    use_torch_jit: bool = True,
    use_in1k_dataset: bool = False,
) -> DatasetDict:

    logger.info(f"use_in1k_dataset: {use_in1k_dataset}")
    for data_split in [md_const.TRAIN, md_const.VAL]:
        this_split_transform = image_transformations[data_split]
        if use_torch_jit:
            this_split_transform = torch.jit.script(this_split_transform)
        _transform_images = partial(
            transform_images,
            image_transformations=this_split_transform,
        )
        dataset[data_split].set_transform(_transform_images)
    return dataset


def collate_fn(
    examples: List[Dict[str, Any]], image_only: bool = False, input_ids_as_tensor: bool = False
) -> Dict[str, Any]:
    # image is already torch.tensor during transform_images with torch.jit
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    if image_only:
        return {"pixel_values": pixel_values}

    labels = torch.tensor([example[md_const.LABEL_ID] for example in examples], dtype=torch.long)
    domain_ids = torch.tensor(
        [example[md_const.DOMAIN_ID] for example in examples], dtype=torch.long
    )
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "label_domain_ids": domain_ids,
        "return_dict": True,
    }
