import logging
from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd

from src.common.dataset.downsampling_data import make_label_id_successive

from .constants import hotel_id_const

logger = logging.getLogger(__name__)


def make_image_info_from_path(all_image_paths: Iterator[Path]) -> List[Dict[str, str]]:
    image_infos = []
    for path in all_image_paths:
        image_infos.append(
            {
                hotel_id_const.IMAGE_ID: path.stem,
                hotel_id_const.FOLDER_NAME: path.parent.name,
                hotel_id_const.FILEPATH: str(path),
            }
        )
    return image_infos


def load_train_csv(data_dir: Path) -> pd.DataFrame:

    image_dir = data_dir / hotel_id_const.TRAIN_DIR_NAME
    all_image_paths = image_dir.glob("**/*.jpg")
    image_infos = make_image_info_from_path(all_image_paths=all_image_paths)

    hotel_id_df = pd.DataFrame(image_infos)
    hotel_id_df[hotel_id_const.LABEL_ID] = hotel_id_df[hotel_id_const.FOLDER_NAME]
    hotel_id_df = make_label_id_successive(
        train_df=hotel_id_df, class_label_name=hotel_id_const.LABEL_ID
    )
    logger.info(
        f"found {len(hotel_id_df)} images and {hotel_id_df[hotel_id_const.LABEL_ID].nunique()} instances"
    )
    return hotel_id_df
