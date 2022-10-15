import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.common.dataset.downsampling_data import make_label_id_successive, sampling_per_class

from .constants import food_101_const

logger = logging.getLogger(__name__)


def get_file_path(data_dir: Path, image_name: str, extension: str = ".jpg") -> Path:
    return Path(data_dir, food_101_const.IMAGE_DIR, image_name + extension)


def make_query_id(food_101_df: pd.DataFrame) -> pd.DataFrame:
    food_101_df[food_101_const.IMAGE_ID] = (
        food_101_df[food_101_const.LABEL_NAME].astype(str)
        + "_"
        + food_101_df[food_101_const.IMAGE_ID].astype(str)
    )
    return food_101_df


def add_super_class_name(food_101_df: pd.DataFrame) -> pd.DataFrame:
    food_101_df[food_101_const.SUPER_CLASS_NAME] = food_101_df[food_101_const.IMAGE_NAME].apply(
        lambda x: x.split("/")[0].replace("_final", "")
    )
    return food_101_df


def load_label_txt(txt_path: Path) -> pd.DataFrame:
    with txt_path.open("r") as fp:
        lines = fp.readlines()
        lines = [
            [int(elem) if elem.isdecimal() else elem for elem in line.strip().split("/")]
            for line in lines
        ]
        food_101_df = pd.DataFrame(
            lines, columns=[food_101_const.LABEL_NAME, food_101_const.IMAGE_ID]
        )
        food_101_df[food_101_const.LABEL_ID] = food_101_df[food_101_const.LABEL_NAME]
        food_101_df[food_101_const.IMAGE_NAME] = (
            food_101_df[food_101_const.LABEL_NAME].astype(str)
            + "/"
            + food_101_df[food_101_const.IMAGE_ID].astype(str)
        )

    return food_101_df


def load_eval_dataframe(data_dir: Path) -> pd.DataFrame:
    test_txt_path = data_dir / food_101_const.TEST_TXT
    food_101_df = load_label_txt(txt_path=test_txt_path)
    food_101_df[food_101_const.FILEPATH] = food_101_df[food_101_const.IMAGE_NAME].apply(
        lambda x: str(get_file_path(data_dir=data_dir, image_name=x))
    )
    logger.info(
        f"Using food_101_dataset as eval: num_classes {food_101_df[food_101_const.LABEL_ID].nunique()}, num_samples: {len(food_101_df)}"
    )
    return food_101_df


def load_train_txt(data_dir: Path, use_furniture_only: bool = True) -> pd.DataFrame:

    train_txt_path = data_dir / food_101_const.TRAIN_TXT
    food_101_df = load_label_txt(txt_path=train_txt_path)
    food_101_df[food_101_const.FILEPATH] = food_101_df[food_101_const.IMAGE_NAME].apply(
        lambda x: str(get_file_path(data_dir=data_dir, image_name=x))
    )
    logger.info(
        f"Using food_101_dataset as train: num_classes {food_101_df[food_101_const.LABEL_ID].nunique()}, num_samples: {len(food_101_df)}"
    )
    return food_101_df
