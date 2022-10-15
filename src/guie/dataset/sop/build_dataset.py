import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from src.common.dataset.retrieval import make_query_index_id_pair, split_query_with_label_id
from src.guie.dataset.dataset_utils import transform_images

from .constants import sop_const
from .preprocess import load_eval_dataframe, make_query_id

logger = logging.getLogger(__name__)


def load_sop_dataset(
    data_dir: Path,
    arrow_dir: Optional[Path] = None,
    num_index_data: Optional[int] = None,
    num_query_data: Optional[int] = None,
    num_queries_per_class: Optional[int] = None,
    update_eval_dataset: bool = False,
    is_jpeg_dataset: bool = True,
) -> Tuple[DatasetDict, Dict[str, List[str]]]:

    sop_df = load_eval_dataframe(data_dir=data_dir, use_furniture_only=True)
    sop_df = make_query_id(sop_df=sop_df)
    if arrow_dir is not None:
        if (not update_eval_dataset) & arrow_dir.exists():
            logger.info(f"load sop eval dataset from {str(arrow_dir)}")
            dataset = DatasetDict.load_from_disk(str(arrow_dir))
            query_ids = dataset[sop_const.QUERY][sop_const.IMAGE_ID]
            sop_df[sop_const.DATA_SPLIT] = sop_const.INDEX
            sop_df.loc[
                sop_df[sop_const.IMAGE_ID].isin(query_ids), sop_const.DATA_SPLIT
            ] = sop_const.QUERY
            solution_index = make_query_index_id_pair(df=sop_df, dataset_const=sop_const)
            return dataset, solution_index

    sop_df = split_query_with_label_id(
        df=sop_df,
        dataset_const=sop_const,
        num_samples_per_class=num_queries_per_class,
        num_max_query_data=num_query_data,
    )
    # Set retrieval solution.
    solution_index = make_query_index_id_pair(df=sop_df, dataset_const=sop_const)

    index_df = sop_df[sop_df[sop_const.DATA_SPLIT] == sop_const.INDEX]
    test_df = sop_df[sop_df[sop_const.DATA_SPLIT] == sop_const.QUERY]

    if num_index_data is not None:
        index_df = index_df.sample(frac=num_index_data / len(index_df))

    index_dataset = Dataset.from_pandas(df=index_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(df=test_df.reset_index(drop=True))

    dataset = DatasetDict(
        {
            sop_const.QUERY: test_dataset,
            sop_const.INDEX: index_dataset,
        }
    )
    if arrow_dir is not None:
        logger.info(f"save sop eval dataset on {str(arrow_dir)}")
        dataset.save_to_disk(str(arrow_dir))

    return dataset, solution_index


def dataset_pipeline(
    sop_dataset: DatasetDict,
    data_dir: Path,
    image_transformations: Callable[[torch.Tensor], torch.Tensor],
) -> DatasetDict:

    _transform_images = partial(transform_images, image_transformations=image_transformations)
    sop_dataset.set_transform(transform=_transform_images)
    return sop_dataset


def collate_fn(examples: dict) -> Dict[str, Any]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    ids = np.stack([example[sop_const.IMAGE_ID] for example in examples], axis=0)
    return {
        "pixel_values": pixel_values,
        sop_const.IMAGE_ID: ids,
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
