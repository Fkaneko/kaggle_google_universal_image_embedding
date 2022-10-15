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

from .constants import objectnet_const
from .preprocess import load_train_csv, make_query_id

logger = logging.getLogger(__name__)


def load_objectnet_dataset(
    data_dir: Path,
    arrow_dir: Optional[Path] = None,
    num_index_data: Optional[int] = None,
    num_query_data: int = 500,
    num_queries_per_class: int = 2,
    update_eval_dataset: bool = False,
    is_jpeg_dataset: bool = True,
) -> Tuple[DatasetDict, Dict[str, List[str]]]:

    objectnet_df = load_train_csv(data_dir=data_dir, is_jpeg_dataset=is_jpeg_dataset)
    objectnet_df = make_query_id(objectnet_df=objectnet_df)
    if arrow_dir is not None:
        if (not update_eval_dataset) & arrow_dir.exists():
            logger.info(f"load objectnet eval dataset from {str(arrow_dir)}")
            dataset = DatasetDict.load_from_disk(str(arrow_dir))
            query_ids = dataset[objectnet_const.QUERY][objectnet_const.IMAGE_ID]
            objectnet_df[objectnet_const.DATA_SPLIT] = objectnet_const.INDEX
            objectnet_df.loc[
                objectnet_df[objectnet_const.IMAGE_ID].isin(query_ids), objectnet_const.DATA_SPLIT
            ] = objectnet_const.QUERY
            solution_index = make_query_index_id_pair(
                df=objectnet_df, dataset_const=objectnet_const
            )
            return dataset, solution_index

    # Set retrieval solution.
    objectnet_df = split_query_with_label_id(
        df=objectnet_df,
        dataset_const=objectnet_const,
        num_samples_per_class=num_queries_per_class,
        num_max_query_data=num_query_data,
    )
    # Set retrieval solution.
    solution_index = make_query_index_id_pair(df=objectnet_df, dataset_const=objectnet_const)

    index_df = objectnet_df[objectnet_df[objectnet_const.DATA_SPLIT] == objectnet_const.INDEX]
    test_df = objectnet_df[objectnet_df[objectnet_const.DATA_SPLIT] == objectnet_const.QUERY]

    if num_index_data is not None:
        index_df = index_df.sample(frac=num_index_data / len(index_df))

    index_dataset = Dataset.from_pandas(df=index_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(df=test_df.reset_index(drop=True))

    dataset = DatasetDict(
        {
            objectnet_const.QUERY: test_dataset,
            objectnet_const.INDEX: index_dataset,
        }
    )
    if arrow_dir is not None:
        logger.info(f"save objectnet eval dataset on {str(arrow_dir)}")
        dataset.save_to_disk(str(arrow_dir))

    return dataset, solution_index


def dataset_pipeline(
    objectnet_dataset: DatasetDict,
    data_dir: Path,
    image_transformations: Callable[[torch.Tensor], torch.Tensor],
) -> DatasetDict:

    _transform_images = partial(transform_images, image_transformations=image_transformations)
    objectnet_dataset.set_transform(transform=_transform_images)
    return objectnet_dataset


def collate_fn(examples: dict) -> Dict[str, Any]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    ids = np.stack([example[objectnet_const.IMAGE_ID] for example in examples], axis=0)
    return {
        "pixel_values": pixel_values,
        objectnet_const.IMAGE_ID: ids,
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
