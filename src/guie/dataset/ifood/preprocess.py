import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from .constants import ifood_const

logger = logging.getLogger(__name__)


def get_file_path(data_dir: Path, data_split: str, filename: str) -> Path:
    return Path(data_dir, data_split, filename)


def load_label_id_to_name_mapping(data_dir: Path) -> Dict[int, str]:
    mapping_path = Path(data_dir / ifood_const.LABEL_ID_TO_NAME_FILE)
    label_id_to_name = {}
    with mapping_path.open("r") as fp:
        for line in fp.readlines():
            label_id, label_name = line.strip().split(" ")
            label_id_to_name[int(label_id)] = label_name
    return label_id_to_name


def load_train_csv(data_dir: Path) -> pd.DataFrame:
    logger.warn("for clean data, use only validation data for iFood")
    train_json_path = data_dir / ifood_const.VAL_CSV
    train_df = pd.read_csv(train_json_path)
    logger.info(f"load {len(train_df)} samples as iFood dataset")
    train_df[ifood_const.FILEPATH] = train_df[ifood_const.FILENAME].apply(
        lambda x: str(
            get_file_path(data_dir=data_dir, data_split=ifood_const.VAL_DIR_NAME, filename=x)
        )
    )
    label_id_to_name = load_label_id_to_name_mapping(data_dir=data_dir)
    train_df[ifood_const.LABEL_NAME] = train_df[ifood_const.LABEL_ID].apply(
        lambda x: label_id_to_name[x]
    )
    return train_df
