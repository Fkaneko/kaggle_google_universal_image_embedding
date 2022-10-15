import logging
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, NewType, Optional, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict, DatasetInfo, concatenate_datasets, load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from src.common.dataset.read_image import convert_to_rgb

from .constants import in_1k_const

logger = logging.getLogger(__name__)

IN1KExample = NewType("IN1KExample", dict)

IN_1K_ARROW_FILES = {
    in_1k_const.TRAIN: "imagenet-1k-train.arrow",
    in_1k_const.VAL: "imagenet-1k-validation.arrow",
    in_1k_const.TEST: "imagenet-1k-test.arrow",
}

HUGGING_FACE_IN_1K = "imagenet-1k"


def transform_images(
    examples: IN1KExample, image_column: str, image_transformations: torch.nn.Module
):
    # There are Grayscale images on WIT, need convert("RGB")
    # images = [pil_to_tensor(image_pil.convert("RGBA")) for image_pil in examples[image_column]]
    images = [pil_to_tensor(convert_to_rgb(image_pil)) for image_pil in examples[image_column]]
    examples["pixel_values"] = [image_transformations(image) for image in images]
    return examples


def load_in_1k_dataset(
    cache_dir: Path,
    data_files: Optional[Union[Dict[str, str], List[str]]] = None,
    arrow_dir: Optional[Path] = None,
    drop_test_split: bool = True,
    target_indices: Optional[List[int]] = None,
) -> DatasetDict:
    if arrow_dir is not None:
        logger.info("load from cached arrow file for dataset generation")
        # load from cached arrow_file
        dataset_dict = {}
        for data_split in [in_1k_const.TRAIN, in_1k_const.VAL, in_1k_const.TEST]:
            arrow_file = arrow_dir / IN_1K_ARROW_FILES[data_split]
            dataset_info = DatasetInfo.from_directory(str(arrow_dir))
            dataset_dict[data_split] = Dataset.from_file(
                filename=str(arrow_file),
                info=dataset_info,
            )
        dataset = DatasetDict(dataset_dict)
    else:
        # load from original
        dataset = load_dataset(HUGGING_FACE_IN_1K, cache_dir=str(cache_dir), use_auth_token=True)
    if drop_test_split:
        _ = dataset.pop(in_1k_const.TEST)

    if target_indices is not None:
        orig_num_example = dataset.num_rows
        dataset = dataset.select(target_indices)
        logger.info(f"slice dataset, original num: {orig_num_example} -> {dataset.num_rows}")
    return dataset


def in_1k_dataset_pipeline(
    dataset: DatasetDict,
    image_transformations: Dict[str, torch.nn.Module],
    num_proc: int = 6,
    use_torch_jit: bool = True,
) -> DatasetDict:
    for data_split in [in_1k_const.TRAIN, in_1k_const.VAL]:
        this_split_transform = image_transformations[data_split]
        if use_torch_jit:
            this_split_transform = torch.jit.script(this_split_transform)
        _transform_images = partial(
            transform_images,
            image_column=in_1k_const.IMAGE,
            image_transformations=this_split_transform,
        )
        dataset[data_split].set_transform(_transform_images)
    return dataset


def collate_fn(
    examples: IN1KExample, image_only: bool = False, input_ids_as_tensor: bool = False
) -> dict:
    # image is already torch.tensor during transform_images with torch.jit
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    if image_only:
        return {"pixel_values": pixel_values}

    label = torch.tensor([example[in_1k_const.LABEL] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "labels": label,
        "return_dict": True,
    }


def get_dataloader(
    ds: Dataset,
    batch_size: int,
    num_workers: int = 6,
    pin_memory: bool = True,
    is_train: bool = False,
    image_only: bool = True,
) -> DataLoader:
    if is_train:
        sampler = RandomSampler(ds)
    else:
        sampler = SequentialSampler(ds)
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
