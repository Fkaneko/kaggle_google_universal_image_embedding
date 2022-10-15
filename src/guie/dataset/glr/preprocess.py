import logging
from pathlib import Path
from typing import Callable, List, NewType, Optional

import pandas as pd
import torch
from torchvision.io import ImageReadMode, read_image

from .constants import glr_const

logger = logging.getLogger(__name__)

GLRExample = NewType("GLRExample", dict)

from src.guie.dataset.dataset_utils import transform_images as common_transform


def transform_images(
    examples: GLRExample, image_transformations: Callable[[torch.Tensor], torch.Tensor]
) -> GLRExample:
    return common_transform(examples=examples, image_transformations=image_transformations)


def get_directory_tree_from_id(id: str) -> Path:
    return Path(id[0], id[1], id[2], f"{id}.jpg")


def filter_not_found_file(examples: GLRExample) -> List[bool]:
    is_founds = []
    for filepath in examples[glr_const.FILEPATH]:
        is_founds.append(Path(filepath).exists())
    return is_founds


def get_file_path(data_dir: Path, data_split: str, id: str) -> Path:
    assert data_split in [glr_const.TRAIN, glr_const.INDEX, glr_const.TEST]
    relative_filepath = get_directory_tree_from_id(id=id)
    return Path(data_dir, data_split, relative_filepath)


def add_filepath(examples: GLRExample, data_dir: Path, data_split: str) -> GLRExample:
    filepaths = [
        str(get_file_path(data_dir=data_dir, data_split=data_split, id=id_str))
        for id_str in examples[glr_const.ID]
    ]
    examples[glr_const.FILEPATH] = filepaths
    return examples


def load_train_csv(data_dir: Path, add_filepath: bool = False) -> pd.DataFrame:

    train_csv_path = data_dir / (glr_const.TRAIN + ".csv")
    train_df = pd.read_csv(train_csv_path)

    if add_filepath:
        train_df[glr_const.FILEPATH] = train_df[glr_const.ID].apply(
            lambda x: str(get_file_path(data_dir=data_dir, data_split=glr_const.TRAIN, id=x))
        )
    return train_df


if __name__ == "__main__":
    from src.common.dataset.downsampling_data import downsample_with_class_label
    from src.common.utils import set_logger

    set_logger(logger=logger)
    glr_cleaned_dir = Path("../input/glr_cleaned")
    glr_df = load_train_csv(data_dir=glr_cleaned_dir)
    glr_df = downsample_with_class_label(train_df=glr_df, class_label_name=glr_const.LABEL_ID)
    print(glr_df.landmark_id)
    # print(glr_df.head())
    # print(glr_df[glr_const.LABEL_ID].value_counts())
    # print((glr_df[glr_const.LABEL_ID].value_counts() > 50).index)
