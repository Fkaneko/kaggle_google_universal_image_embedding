import logging
from pathlib import Path

import pandas as pd

from .constants import sop_const

logger = logging.getLogger(__name__)

FURNITURES = (
    "lamp",
    "cabinet",
    "chair",
    "table",
    "sofa",
)

HOME_PRODUCTS = (
    "coffee_maker",
    "kettle",
    "stapler",
    "toaster",
    "bicycle",
    "fan",
    "mug",
)


def get_file_path(data_dir: Path, image_name: str) -> Path:
    return Path(data_dir, image_name)


def make_query_id(sop_df: pd.DataFrame) -> pd.DataFrame:
    sop_df[sop_const.IMAGE_ID] = (
        sop_df[sop_const.SUPER_CLASS_NAME].astype(str)
        + "_"
        + sop_df[sop_const.IMAGE_ID].astype(str)
    )
    return sop_df


def add_super_class_name(sop_df: pd.DataFrame) -> pd.DataFrame:
    sop_df[sop_const.SUPER_CLASS_NAME] = sop_df[sop_const.IMAGE_NAME].apply(
        lambda x: x.split("/")[0].replace("_final", "")
    )
    return sop_df


def load_label_txt(txt_path: Path, use_furniture_only: bool = True) -> pd.DataFrame:
    with txt_path.open("r") as fp:
        lines = fp.readlines()
        column_names = lines.pop(0).strip().split()
        lines = [
            [int(elem) if elem.isdecimal() else elem for elem in line.strip().split()]
            for line in lines
        ]
        sop_df = pd.DataFrame(lines, columns=column_names)
    sop_df = add_super_class_name(sop_df=sop_df)

    if use_furniture_only:
        logger.info("drop not Furniture super classes")
        sop_df = sop_df.loc[sop_df[sop_const.SUPER_CLASS_NAME].isin(FURNITURES)]
        sop_df = sop_df.reset_index(drop=True)

    return sop_df


def load_eval_dataframe(data_dir: Path, use_furniture_only: bool = True) -> pd.DataFrame:

    test_txt_path = data_dir / sop_const.TEST_TXT
    sop_df = load_label_txt(txt_path=test_txt_path, use_furniture_only=use_furniture_only)
    sop_df[sop_const.FILEPATH] = sop_df[sop_const.IMAGE_NAME].apply(
        lambda x: str(get_file_path(data_dir=data_dir, image_name=x))
    )
    logger.info(
        f"Using sop_dataset as eval: num_classes {sop_df[sop_const.LABEL_ID].nunique()}, num_samples: {len(sop_df)}"
    )
    logger.info(
        f"sop dataset num classes per super class:\n{sop_df.groupby(sop_const.SUPER_CLASS_NAME)[sop_const.LABEL_ID].nunique().sort_values()}"
    )
    return sop_df


def load_train_txt(data_dir: Path, use_furniture_only: bool = True) -> pd.DataFrame:

    train_txt_path = data_dir / sop_const.TRAIN_TXT
    sop_df = load_label_txt(txt_path=train_txt_path, use_furniture_only=use_furniture_only)
    sop_df[sop_const.FILEPATH] = sop_df[sop_const.IMAGE_NAME].apply(
        lambda x: str(get_file_path(data_dir=data_dir, image_name=x))
    )
    logger.info(
        f"Using sop_dataset as train: num_classes {sop_df[sop_const.LABEL_ID].nunique()}, num_samples: {len(sop_df)}"
    )
    logger.info(f"sop dataset super class:\n{sop_df[sop_const.SUPER_CLASS_NAME].value_counts()}")
    logger.info(
        f"sop dataset num classes per super class:\n{sop_df.groupby(sop_const.SUPER_CLASS_NAME)[sop_const.LABEL_ID].nunique().sort_values()}"
    )
    return sop_df
