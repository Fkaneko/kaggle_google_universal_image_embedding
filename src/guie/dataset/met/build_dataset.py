import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchvision.io import ImageReadMode, read_image

from src.common.dataset.retrieval import make_query_index_id_pair, split_query_with_label_id
from src.guie.dataset.dataset_utils import transform_images

from .constants import met_const
from .preprocess import load_eval_dataframe, make_query_id

logger = logging.getLogger(__name__)


def load_met_dataset(
    data_dir: Path,
    arrow_dir: Optional[Path] = None,
    num_index_data: Optional[int] = None,
    num_query_data: Optional[int] = None,
    num_queries_per_class: Optional[int] = None,
    update_eval_dataset: bool = False,
    is_jpeg_dataset: bool = True,
) -> Tuple[DatasetDict, Dict[str, List[str]]]:

    met_df = load_eval_dataframe(data_dir=data_dir)
    met_df = make_query_id(met_df=met_df)
    if arrow_dir is not None:
        if (not update_eval_dataset) & arrow_dir.exists():
            logger.info(f"load met eval dataset from {str(arrow_dir)}")
            dataset = DatasetDict.load_from_disk(str(arrow_dir))
            query_ids = dataset[met_const.QUERY][met_const.IMAGE_ID]
            met_df[met_const.DATA_SPLIT] = met_const.INDEX
            met_df.loc[
                met_df[met_const.IMAGE_ID].isin(query_ids), met_const.DATA_SPLIT
            ] = met_const.QUERY
            solution_index = make_query_index_id_pair(df=met_df, dataset_const=met_const)
            return dataset, solution_index

    # met_df = split_query_with_label_id(
    #     df=met_df,
    #     dataset_const=met_const,
    #     num_samples_per_class=num_queries_per_class,
    #     num_max_query_data=num_query_data,
    # )
    # Set retrieval solution.
    solution_index = make_query_index_id_pair(df=met_df, dataset_const=met_const)

    index_df = met_df[met_df[met_const.DATA_SPLIT] == met_const.INDEX]
    test_df = met_df[met_df[met_const.DATA_SPLIT] == met_const.QUERY]

    if num_index_data is not None:
        index_df = index_df.sample(frac=num_index_data / len(index_df))

    index_dataset = Dataset.from_pandas(df=index_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(df=test_df.reset_index(drop=True))

    dataset = DatasetDict(
        {
            met_const.QUERY: test_dataset,
            met_const.INDEX: index_dataset,
        }
    )
    if arrow_dir is not None:
        logger.info(f"save met eval dataset on {str(arrow_dir)}")
        dataset.save_to_disk(str(arrow_dir))

    return dataset, solution_index


def dataset_pipeline(
    met_dataset: DatasetDict,
    data_dir: Path,
    image_transformations: Callable[[torch.Tensor], torch.Tensor],
) -> DatasetDict:

    _transform_images = partial(transform_images, image_transformations=image_transformations)
    met_dataset.set_transform(transform=_transform_images)
    return met_dataset


def collate_fn(examples: dict) -> Dict[str, Any]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    ids = np.stack([example[met_const.IMAGE_ID] for example in examples], axis=0)
    return {
        "pixel_values": pixel_values,
        met_const.IMAGE_ID: ids,
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
