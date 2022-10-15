import logging
from pathlib import Path

import pandas as pd

from src.common.dataset.downsampling_data import make_label_id_successive

from .constants import in_shop_const

logger = logging.getLogger(__name__)


def get_file_path(data_dir: Path, image_name: str) -> Path:
    return Path(data_dir, image_name)


def make_query_id(in_shop_df: pd.DataFrame) -> pd.DataFrame:
    in_shop_df[in_shop_const.IMAGE_ID] = in_shop_df[in_shop_const.IMAGE_NAME].apply(
        lambda x: "/".join(x.split("/")[1:])  # delete common "img/" part
    )
    return in_shop_df


def load_label_txt(data_dir: Path) -> pd.DataFrame:
    train_txt_path = data_dir / in_shop_const.TRAIN_TXT

    with train_txt_path.open("r") as fp:
        lines = fp.readlines()
        data_num = lines.pop(0)
        column_names = lines.pop(0).split()
        lines = [line.strip().split() for line in lines]
        in_shop_df = pd.DataFrame(lines, columns=column_names)

    in_shop_df[in_shop_const.FILEPATH] = in_shop_df[in_shop_const.IMAGE_NAME].apply(
        lambda x: str(get_file_path(data_dir=data_dir, image_name=x))
    )
    logger.info(f"Load in_shop_dataset: \n {in_shop_df[in_shop_const.DATA_SPLIT].value_counts()}")
    return in_shop_df


def load_eval_dataframe(data_dir: Path, use_train_data_only: bool = True) -> pd.DataFrame:

    in_shop_df = load_label_txt(data_dir=data_dir)
    in_shop_df = in_shop_df[~(in_shop_df[in_shop_const.DATA_SPLIT] == in_shop_const.TRAIN)]
    logger.info(
        f"Using in_shop_dataset as eval: \n {in_shop_df[in_shop_const.LABEL_ID].value_counts()}"
    )
    return in_shop_df


def load_train_txt(data_dir: Path, use_train_data_only: bool = True) -> pd.DataFrame:

    in_shop_df = load_label_txt(data_dir=data_dir)
    if use_train_data_only:
        in_shop_df = in_shop_df[in_shop_df[in_shop_const.DATA_SPLIT] == in_shop_const.TRAIN]
        in_shop_df = in_shop_df.reset_index(drop=True)

    in_shop_df = make_label_id_successive(
        train_df=in_shop_df, class_label_name=in_shop_const.LABEL_ID
    )
    logger.info(
        f"Using in_shop_dataset as train: num_classes {in_shop_df[in_shop_const.LABEL_ID].nunique()}, num_samples: {len(in_shop_df)}"
    )
    return in_shop_df
