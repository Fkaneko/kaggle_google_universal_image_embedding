import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.dataset.downsampling_data import sampling_per_class

from .constants import products_10k_const

logger = logging.getLogger(__name__)


def make_query_id(products_10k_df: pd.DataFrame) -> pd.DataFrame:

    products_10k_df[products_10k_const.IMAGE_ID] = (
        products_10k_df[products_10k_const.TRAIN_OR_TEST].astype(str)
        + "_"
        + products_10k_df[products_10k_const.LABEL_ID].astype(str)
        + "_"
        + products_10k_df[products_10k_const.NAME].astype(str)
    )
    return products_10k_df


def get_file_path(data_dir: Path, data_split: str, filename: str) -> Path:
    return Path(data_dir, data_split, filename)


def rename_label_col(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df[products_10k_const.LABEL_ID] = train_df[products_10k_const.LABEL]
    del train_df[products_10k_const.LABEL]
    return train_df


def load_data_csv(data_dir: Path, data_split: str) -> pd.DataFrame:

    if data_split == products_10k_const.TRAIN:
        csv_path = data_dir / products_10k_const.TRAIN_CSV
    elif data_split == products_10k_const.TEST:
        csv_path = data_dir / products_10k_const.TEST_CSV

    train_df = pd.read_csv(csv_path)

    train_df[products_10k_const.FILEPATH] = train_df[products_10k_const.NAME].apply(
        lambda x: str(get_file_path(data_dir=data_dir, data_split=data_split, filename=x))
    )
    train_df = rename_label_col(train_df=train_df)
    train_df[products_10k_const.TRAIN_OR_TEST] = data_split

    logger.info(
        f"load products_10k num_classes {train_df[products_10k_const.LABEL_ID].nunique()}, num_samples: {len(train_df)}"
    )
    return train_df


def split_query_with_group_id(
    test_df: pd.DataFrame,
    data_dir: Path,
    update_query_list: bool = True,
    num_samples_per_group: int = 2,
    num_samples_per_class: int = 3,
    num_samples_per_class_from_train: int = 5,
) -> pd.DataFrame:

    cache_query_list_path = data_dir / products_10k_const.QUERY_LABEL_IDS_LIST
    if cache_query_list_path.exists() & (not update_query_list):
        logger.info(f"loading pre computed query list: {str(cache_query_list_path)}")
        query_df = pd.read_csv(cache_query_list_path)
    else:
        # use group info for query label id selection
        # group info generation from train data
        train_df = load_data_csv(data_dir=data_dir, data_split=products_10k_const.TRAIN)
        train_df = make_query_id(products_10k_df=train_df)
        label_id_to_group_id = {
            label_id: group_ids.iloc[0]
            for label_id, group_ids in train_df.groupby(products_10k_const.LABEL_ID)[
                products_10k_const.GROUP
            ]
        }
        group_id_to_label_ids = {
            group_id: list(set(label_ids))
            for group_id, label_ids in train_df.groupby(products_10k_const.GROUP)[
                products_10k_const.LABEL_ID
            ]
        }
        small_group_ids = [
            group_id
            for group_id, label_ids in group_id_to_label_ids.items()
            if len(label_ids) <= num_samples_per_group
        ]

        # map group info test and then select query label id
        test_df[products_10k_const.GROUP] = test_df[products_10k_const.LABEL_ID].apply(
            lambda x: label_id_to_group_id[x]
        )
        grouped_sampled_df = sampling_per_class(
            test_df,
            class_label_name=products_10k_const.GROUP,
            num_samples_per_class=num_samples_per_group,
        )
        # do not allow small group into query, because query data will be removed from train data
        grouped_sampled_df = grouped_sampled_df[
            ~grouped_sampled_df[products_10k_const.GROUP].isin(small_group_ids)
        ]
        query_label_ids = np.unique(grouped_sampled_df[products_10k_const.LABEL_ID])
        query_df = sampling_per_class(
            test_df[test_df[products_10k_const.LABEL_ID].isin(query_label_ids)],
            class_label_name=products_10k_const.GROUP,
            num_samples_per_class=num_samples_per_class,
        )
        if update_query_list:
            logger.info(f"Update products_10k query label ids: {str(cache_query_list_path)}")
            query_df.to_csv(cache_query_list_path, index=False)

    logger.info(f"Split query classes for products_10k: {len(query_label_ids)}")
    test_df[products_10k_const.DATA_SPLIT] = products_10k_const.INDEX
    test_df.loc[
        test_df[products_10k_const.IMAGE_ID].isin(query_df[products_10k_const.IMAGE_ID]),
        products_10k_const.DATA_SPLIT,
    ] = products_10k_const.QUERY
    # additional query label id data from train and add into index
    query_train_df = train_df.loc[train_df[products_10k_const.LABEL_ID].isin(query_label_ids)]
    query_train_df = sampling_per_class(
        query_train_df[query_train_df[products_10k_const.LABEL_ID].isin(query_label_ids)],
        class_label_name=products_10k_const.GROUP,
        num_samples_per_class=num_samples_per_class_from_train,
    )
    query_train_df[products_10k_const.DATA_SPLIT] = products_10k_const.INDEX
    test_df = pd.concat([test_df, query_train_df], axis=0).reset_index(drop=True)

    logger.info(
        f"Split query/index for dataset: products_10k \n{test_df[products_10k_const.DATA_SPLIT].value_counts()}"
    )
    return test_df


def load_test_csv(data_dir: Path) -> pd.DataFrame:
    test_df = load_data_csv(data_dir=data_dir, data_split=products_10k_const.TEST)
    return test_df


def load_train_csv(data_dir: Path, drop_query_samples: bool = True) -> pd.DataFrame:
    train_df = load_data_csv(data_dir=data_dir, data_split=products_10k_const.TRAIN)
    if drop_query_samples:
        cache_query_list_path = data_dir / products_10k_const.QUERY_LABEL_IDS_LIST
        logger.info(f"loading pre computed query list: {str(cache_query_list_path)}")
        query_df = pd.read_csv(cache_query_list_path)
        query_label_ids = list(set(query_df[products_10k_const.LABEL_ID]))
        train_df = train_df.loc[~train_df[products_10k_const.LABEL_ID].isin(query_label_ids)]
        logger.info(
            f"After query class removal products_10k num_classes {train_df[products_10k_const.LABEL_ID].nunique()}, num_samples: {len(train_df)}"
        )
    return train_df


def add_domain_name(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df[products_10k_const.DOMAIN_NAME] = products_10k_const.OTHER
    pack_mask = train_df[products_10k_const.GROUP].isin(products_10k_const.PACKAGE_GROUPS)
    toy_mask = train_df[products_10k_const.GROUP].isin(products_10k_const.TOY_GROUPS)

    train_df.loc[pack_mask, products_10k_const.DOMAIN_NAME] = products_10k_const.PACKAGE
    train_df.loc[toy_mask, products_10k_const.DOMAIN_NAME] = products_10k_const.TOY
    return train_df
