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

from src.common.dataset.downsampling_data import sampling_per_class
from src.common.dataset.retrieval import make_query_index_id_pair
from src.guie.dataset.dataset_utils import transform_images

from .constants import in_shop_const
from .preprocess import load_eval_dataframe, make_query_id

logger = logging.getLogger(__name__)


def load_in_shop_dataset(
    data_dir: Path,
    arrow_dir: Optional[Path] = None,
    num_index_data: Optional[int] = None,
    num_query_data: Optional[int] = None,
    update_eval_dataset: bool = False,
    is_jpeg_dataset: bool = True,
) -> Tuple[DatasetDict, Dict[str, List[str]]]:

    in_shop_df = load_eval_dataframe(data_dir=data_dir)
    in_shop_df = make_query_id(in_shop_df=in_shop_df)
    if arrow_dir is not None:
        if (not update_eval_dataset) & arrow_dir.exists():
            logger.info(f"load in_shop eval dataset from {str(arrow_dir)}")
            dataset = DatasetDict.load_from_disk(str(arrow_dir))
            query_ids = dataset[in_shop_const.QUERY][in_shop_const.IMAGE_ID]
            index_ids = dataset[in_shop_const.INDEX][in_shop_const.IMAGE_ID]
            all_ids = list(query_ids) + list(index_ids)
            in_shop_df = in_shop_df[in_shop_df[in_shop_const.IMAGE_ID].isin(all_ids)]

            in_shop_df[in_shop_const.DATA_SPLIT] = in_shop_const.INDEX
            in_shop_df.loc[
                in_shop_df[in_shop_const.IMAGE_ID].isin(query_ids), in_shop_const.DATA_SPLIT
            ] = in_shop_const.QUERY
            solution_index = make_query_index_id_pair(df=in_shop_df, dataset_const=in_shop_const)
            return dataset, solution_index

    # Set retrieval solution.
    solution_index = make_query_index_id_pair(df=in_shop_df, dataset_const=in_shop_const)

    index_df = in_shop_df[in_shop_df[in_shop_const.DATA_SPLIT] == in_shop_const.INDEX]
    test_df = in_shop_df[in_shop_df[in_shop_const.DATA_SPLIT] == in_shop_const.QUERY]
    test_df = sampling_per_class(
        test_df, class_label_name=in_shop_const.LABEL_ID, num_samples_per_class=1
    )
    test_df = test_df.sample(num_query_data)
    # also downsampling solution_index
    solution_index = {
        query_id: index_ids
        for query_id, index_ids in solution_index.items()
        if query_id in test_df[in_shop_const.IMAGE_ID].to_list()
    }
    if num_index_data is not None:
        index_df = index_df.sample(frac=num_index_data / len(index_df))

    index_dataset = Dataset.from_pandas(df=index_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(df=test_df.reset_index(drop=True))

    dataset = DatasetDict(
        {
            in_shop_const.QUERY: test_dataset,
            in_shop_const.INDEX: index_dataset,
        }
    )
    if arrow_dir is not None:
        logger.info(f"save in_shop eval dataset on {str(arrow_dir)}")
        dataset.save_to_disk(str(arrow_dir))

    return dataset, solution_index


def dataset_pipeline(
    in_shop_dataset: DatasetDict,
    data_dir: Path,
    image_transformations: Callable[[torch.Tensor], torch.Tensor],
) -> DatasetDict:

    _transform_images = partial(transform_images, image_transformations=image_transformations)
    in_shop_dataset.set_transform(transform=_transform_images)
    return in_shop_dataset


def collate_fn(examples: dict) -> Dict[str, Any]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    ids = np.stack([example[in_shop_const.IMAGE_ID] for example in examples], axis=0)
    return {
        "pixel_values": pixel_values,
        in_shop_const.IMAGE_ID: ids,
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
