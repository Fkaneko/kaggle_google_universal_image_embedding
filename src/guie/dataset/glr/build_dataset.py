import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, DatasetInfo, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler

from .constants import glr_const
from .glr_metrics.evaluation import SolutionIndex, get_solution_related_indices
from .preprocess import (
    GLRExample,
    add_filepath,
    filter_not_found_file,
    get_file_path,
    transform_images,
)

logger = logging.getLogger(__name__)


def load_glr_dataset(
    data_dir: Path,
    arrow_dir: Optional[Path] = None,
    num_index_data: Optional[int] = None,
    update_eval_dataset: bool = False,
) -> DatasetDict:

    if arrow_dir is not None:
        if (not update_eval_dataset) & arrow_dir.exists():
            logger.info(f"load glr eval dataset from {str(arrow_dir)}")
            return DatasetDict.load_from_disk(str(arrow_dir))

    index_csv_path = data_dir / (glr_const.INDEX + ".csv")
    test_csv_path = data_dir / (glr_const.TEST + ".csv")

    # Read solution.
    solution_path = data_dir / glr_const.RETRIEVAL_SOLUTION_CSV
    solution_index = get_solution_related_indices(solution_path=solution_path)

    index_df = pd.read_csv(index_csv_path)
    test_df = pd.read_csv(test_csv_path)
    index_df["sort_index"] = np.random.random_integers(low=1, high=10, size=len(index_df))

    # select related data
    test_df = test_df[test_df[glr_const.ID].isin(solution_index.target_indices[glr_const.TEST])]

    # set sort_index = 0 on target_indices and then sort
    target_mask = index_df[glr_const.ID].isin(solution_index.target_indices[glr_const.INDEX])
    index_df.loc[target_mask, "sort_index"] = 0
    index_df = index_df.sort_values(["sort_index"])

    if num_index_data is not None:
        orig_num_example = len(index_df)
        index_df = index_df.head(n=num_index_data)
        logger.info(f"slice dataset, original num: {orig_num_example} -> {len(index_df)}")

    # file path settings
    index_df[glr_const.FILEPATH] = index_df[glr_const.ID].apply(
        lambda x: str(get_file_path(data_dir=data_dir, data_split=glr_const.INDEX, id=x))
    )
    test_df[glr_const.FILEPATH] = test_df[glr_const.ID].apply(
        lambda x: str(get_file_path(data_dir=data_dir, data_split=glr_const.TEST, id=x))
    )

    index_dataset = Dataset.from_pandas(df=index_df)
    test_dataset = Dataset.from_pandas(df=test_df)

    dataset = DatasetDict(
        {
            glr_const.TEST: test_dataset,
            glr_const.INDEX: index_dataset,
        }
    )
    if arrow_dir is not None:
        logger.info(f"save glr eval dataset on {str(arrow_dir)}")
        dataset.save_to_disk(str(arrow_dir))

    return dataset


def dataset_pipeline(
    glr_dataset: Dataset,
    data_dir: Path,
    data_split: str,
    image_transformations: Callable[[torch.Tensor], torch.Tensor],
    num_proc: int = 6,
    is_check_file_exists: bool = False,
) -> Dataset:
    def _pipeline(examples: GLRExample) -> GLRExample:
        # text preprocess
        _add_filepath = partial(add_filepath, data_dir=data_dir, data_split=data_split)
        # compose
        examples = _add_filepath(examples)
        return examples

    if is_check_file_exists:
        logger.info("before filtering", glr_dataset)
        glr_dataset = glr_dataset.filter(
            filter_not_found_file,
            batched=True,
            num_proc=num_proc,
            desc="Filtering not found files",
        )
        logger.info("after filtering", glr_dataset)

    _transform_images = partial(transform_images, image_transformations=image_transformations)
    glr_dataset.set_transform(transform=_transform_images)
    return glr_dataset


def get_glr_test_dataset(
    data_dir: Path,
    image_transformations: Callable[[torch.Tensor], torch.Tensor],
    num_index_data: int = 30000,
    arrow_dir: Optional[Path] = None,
    update_eval_dataset: bool = False,
) -> Tuple[DatasetDict, SolutionIndex]:
    glr_dataset = load_glr_dataset(
        data_dir=data_dir,
        num_index_data=num_index_data,
        arrow_dir=arrow_dir,
        update_eval_dataset=update_eval_dataset,
    )

    # Read solution.
    solution_path = data_dir / glr_const.RETRIEVAL_SOLUTION_CSV
    solution_index = get_solution_related_indices(solution_path=solution_path)

    for data_split in glr_dataset.keys():
        glr_dataset[data_split] = dataset_pipeline(
            glr_dataset=glr_dataset[data_split],
            data_dir=data_dir,
            data_split=data_split,
            image_transformations=image_transformations,
        )

    return glr_dataset, solution_index


def collate_fn(examples: GLRExample) -> Dict[str, Any]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    ids = np.stack([example[glr_const.ID] for example in examples], axis=0)
    return {
        "pixel_values": pixel_values,
        glr_const.ID: ids,
        "return_loss": False,
    }


def get_dataloader(
    ds: Dataset,
    batch_size: int,
    num_workers: int = 6,
    pin_memory: bool = True,
    is_train: bool = False,
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
