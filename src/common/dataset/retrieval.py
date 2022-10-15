import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .downsampling_data import sampling_per_class

logger = logging.getLogger(__name__)


def split_query_with_label_id(
    df: pd.DataFrame,
    dataset_const: Any,
    num_samples_per_class: int,
    num_max_query_data: Optional[int] = None,
) -> pd.DataFrame:
    assert num_samples_per_class is not None

    query_df = sampling_per_class(
        train_df=df,
        class_label_name=dataset_const.LABEL_ID,
        num_samples_per_class=num_samples_per_class,
    )
    if num_max_query_data is not None:
        if len(query_df) > num_max_query_data:
            query_df = query_df.sample(num_max_query_data)

    query_ids = query_df[dataset_const.IMAGE_ID]

    df[dataset_const.DATA_SPLIT] = dataset_const.INDEX
    df.loc[
        df[dataset_const.IMAGE_ID].isin(query_ids), dataset_const.DATA_SPLIT
    ] = dataset_const.QUERY

    logger.info(
        f"Split query/index for dataset: {dataset_const.__class__} \n{df[dataset_const.DATA_SPLIT].value_counts()}"
    )
    return df


def make_query_index_id_pair(df: pd.DataFrame, dataset_const: Any) -> Dict[str, List[str]]:
    solution_index = {}
    query_df = df[df[dataset_const.DATA_SPLIT] == dataset_const.QUERY]
    index_df = df[df[dataset_const.DATA_SPLIT] == dataset_const.INDEX]

    query_ids = query_df[dataset_const.IMAGE_ID]
    query_label_ids = query_df[dataset_const.LABEL_ID]

    label_id_to_image_id = {
        label_id: list(set(group_df.to_list()))
        for label_id, group_df in index_df.groupby(dataset_const.LABEL_ID)[dataset_const.IMAGE_ID]
    }
    solution_index = {
        query_id: label_id_to_image_id[query_label_id]
        for query_id, query_label_id in zip(query_ids, query_label_ids)
    }
    return solution_index
