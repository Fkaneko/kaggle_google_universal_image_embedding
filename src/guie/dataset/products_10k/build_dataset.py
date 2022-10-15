import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from src.common.dataset.retrieval import make_query_index_id_pair
from src.guie.dataset.dataset_utils import transform_images

from .constants import products_10k_const
from .preprocess import load_data_csv, load_test_csv, make_query_id, split_query_with_group_id

logger = logging.getLogger(__name__)


def load_products_10k_dataset(
    data_dir: Path,
    arrow_dir: Optional[Path] = None,
    num_index_data: Optional[int] = None,
    num_query_data: Optional[int] = None,
    num_queries_per_class: Optional[int] = None,
    update_eval_dataset: bool = False,
    is_jpeg_dataset: bool = True,
) -> Tuple[DatasetDict, Dict[str, List[str]]]:

    products_10k_df = load_test_csv(data_dir=data_dir)
    products_10k_df = make_query_id(products_10k_df=products_10k_df)
    if arrow_dir is not None:
        if (not update_eval_dataset) & arrow_dir.exists():
            logger.info(f"load products_10k eval dataset from {str(arrow_dir)}")
            dataset = DatasetDict.load_from_disk(str(arrow_dir))
            query_ids = dataset[products_10k_const.QUERY][products_10k_const.IMAGE_ID]
            index_ids = dataset[products_10k_const.INDEX][products_10k_const.IMAGE_ID]

            # need some training data for index data
            train_df = load_data_csv(data_dir=data_dir, data_split=products_10k_const.TRAIN)
            train_df = make_query_id(products_10k_df=train_df)
            query_train_df = train_df[train_df[products_10k_const.IMAGE_ID].isin(index_ids)]

            products_10k_df = pd.concat([products_10k_df, query_train_df], axis=0).reset_index(
                drop=True
            )
            products_10k_df[products_10k_const.DATA_SPLIT] = products_10k_const.INDEX
            products_10k_df.loc[
                products_10k_df[products_10k_const.IMAGE_ID].isin(query_ids),
                products_10k_const.DATA_SPLIT,
            ] = products_10k_const.QUERY
            solution_index_new = make_query_index_id_pair(
                df=products_10k_df, dataset_const=products_10k_const
            )
            # for key in solution_index_new.keys():
            #     assert set(solution_index_new[key]) == set(
            #         solution_index[key]
            #     ), f"{solution_index[key]}, {solution_index_new[key]}"
            return dataset, solution_index_new

    products_10k_df = split_query_with_group_id(
        test_df=products_10k_df,
        data_dir=data_dir,
        update_query_list=True,
        num_samples_per_group=num_queries_per_class,
    )
    # Set retrieval solution.
    solution_index = make_query_index_id_pair(df=products_10k_df, dataset_const=products_10k_const)

    index_df = products_10k_df[
        products_10k_df[products_10k_const.DATA_SPLIT] == products_10k_const.INDEX
    ]
    test_df = products_10k_df[
        products_10k_df[products_10k_const.DATA_SPLIT] == products_10k_const.QUERY
    ]

    if num_index_data is not None:
        index_df = index_df.sample(frac=num_index_data / len(index_df))

    index_dataset = Dataset.from_pandas(df=index_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(df=test_df.reset_index(drop=True))

    dataset = DatasetDict(
        {
            products_10k_const.QUERY: test_dataset,
            products_10k_const.INDEX: index_dataset,
        }
    )
    if arrow_dir is not None:
        logger.info(f"save products_10k eval dataset on {str(arrow_dir)}")
        dataset.save_to_disk(str(arrow_dir))

    return dataset, solution_index


def dataset_pipeline(
    products_10k_dataset: DatasetDict,
    data_dir: Path,
    image_transformations: Callable[[torch.Tensor], torch.Tensor],
) -> DatasetDict:

    _transform_images = partial(transform_images, image_transformations=image_transformations)
    products_10k_dataset.set_transform(transform=_transform_images)
    return products_10k_dataset


def collate_fn(examples: dict) -> Dict[str, Any]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    ids = np.stack([example[products_10k_const.IMAGE_ID] for example in examples], axis=0)
    return {
        "pixel_values": pixel_values,
        products_10k_const.IMAGE_ID: ids,
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
