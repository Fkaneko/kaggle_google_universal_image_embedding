import json
import logging
from pathlib import Path

import pandas as pd

from .constants import met_const

logger = logging.getLogger(__name__)


def get_file_path(data_dir: Path, filename: str) -> Path:
    return Path(data_dir, filename)


def make_query_id(met_df: pd.DataFrame) -> pd.DataFrame:
    met_df[met_const.IMAGE_ID] = (
        met_df[met_const.LABEL_ID].astype(str) + "_" + met_df[met_const.FILENAME].astype(str)
    )
    return met_df


def load_eval_dataframe(data_dir: Path, num_train_index_data: int = 15000) -> pd.DataFrame:
    test_df = load_test_json(data_dir=data_dir, drop_ood_data=False)

    # making query/index split
    # from test dataset
    query_df = test_df[test_df[met_const.LABEL_ID].notna()]
    query_df[met_const.DATA_SPLIT] = met_const.QUERY
    other_art_df = test_df[test_df[met_const.FILENAME].str.startswith(met_const.TEST_OTHER)]
    other_art_df[met_const.LABEL_ID] = -1
    other_art_df[met_const.DATA_SPLIT] = met_const.INDEX

    # from train dataset
    train_df = load_train_json(data_dir=data_dir, drop_query_class=False, add_filepath=False)
    train_df[met_const.DATA_SPLIT] = met_const.INDEX
    train_query_class_mask = train_df[met_const.LABEL_ID].isin(query_df[met_const.LABEL_ID])
    train_query_class_df = train_df[train_query_class_mask]
    train_index_class_df = train_df[~train_query_class_mask]
    train_index_class_df = train_index_class_df.sample(n=num_train_index_data)

    test_df = pd.concat(
        # [query_df, other_art_df, train_query_class_df, train_index_class_df], axis=0
        [query_df, train_query_class_df, train_index_class_df],
        axis=0,
    ).reset_index(drop=True)

    test_df[met_const.FILEPATH] = test_df[met_const.FILENAME].apply(
        lambda x: str(get_file_path(data_dir=data_dir, filename=x))
    )
    logger.info(
        f"Using met_dataset as eval: num_classes {test_df[met_const.LABEL_ID].nunique()}, num_samples: {len(test_df)}"
    )
    logger.info(f"met_dataset as eval: data_split \n{test_df[met_const.DATA_SPLIT].value_counts()}")

    return test_df


def load_test_json(data_dir: Path, drop_ood_data: bool = False) -> pd.DataFrame:
    test_json_path = data_dir / met_const.TEST_JSON
    with test_json_path.open("r") as f:
        loaded = json.load(f)
    test_df = pd.DataFrame(loaded)

    # use only train_df cols
    test_df[met_const.LABEL_ID] = test_df[met_const.MET_ID]
    test_df = test_df.loc[:, [met_const.LABEL_ID, met_const.FILENAME]]
    if drop_ood_data:
        test_df = test_df[test_df[met_const.LABEL_ID].notna()]

    return test_df


def load_train_json(
    data_dir: Path, drop_query_class: bool = True, add_filepath: bool = True
) -> pd.DataFrame:
    train_json_path = data_dir / met_const.TRAIN_JSON
    with train_json_path.open("r") as f:
        loaded = json.load(f)
    train_df = pd.DataFrame(loaded)

    if drop_query_class:
        test_df = load_test_json(data_dir=data_dir, drop_ood_data=True)
        logger.info(
            f"before dropping train met label idl: num_classes {train_df[met_const.LABEL_ID].nunique()}, num_samples: {len(train_df)}"
        )
        train_df = train_df[~train_df[met_const.LABEL_ID].isin(test_df[met_const.LABEL_ID])]
        logger.info(
            f"after dropping train met label idl: num_classes {train_df[met_const.LABEL_ID].nunique()}, num_samples: {len(train_df)}"
        )
    else:
        logger.info(
            f"train met label idl: num_classes {train_df[met_const.LABEL_ID].nunique()}, num_samples: {len(train_df)}"
        )
    if add_filepath:
        train_df[met_const.FILEPATH] = train_df[met_const.FILENAME].apply(
            lambda x: str(get_file_path(data_dir=data_dir, filename=x))
        )
    return train_df
