import json
import logging
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, List, NewType, Optional, Tuple

import pandas as pd
import torch

from .constants import omni_const

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OmniDataFrameColumn:
    DATA_SPLIT: str = "data_split"
    LABEL_ID: str = "label_id"
    FILENAME: str = "filename"
    REALM: str = "realm"

    FILEPATH: str = "filepath"
    LABEL_NAME: str = "label_name"


omni_col = OmniDataFrameColumn()


TARGET_REALMS = (
    "aircraft",
    "consumer_goods",
    "creation",
    "decoration",
    "device",
    "food",
    "geological_formation",
    "instrumentality",
    "locomotive",
    "material",
    "process",
    "region",
    "structure",
)
TARGET_DATA_SPLIT = (omni_const.TRAIN, omni_const.VAL)


def get_file_path(data_dir: Path, realm: str, data_split: str, filename: str) -> Path:
    assert data_split in TARGET_DATA_SPLIT
    assert realm in TARGET_REALMS
    return Path(data_dir, realm, data_split, filename)


def iterate_target_dirs() -> Generator[Tuple[str, str], None, None]:
    for realm in TARGET_REALMS:
        for data_split in TARGET_DATA_SPLIT:
            yield realm, data_split


Record = namedtuple("Record", [str(omni_col.FILENAME), str(omni_col.LABEL_ID)])


def load_record_txt(record_txt_path: Path) -> List[Record]:
    records = []
    for line in record_txt_path.read_text().splitlines():
        filename, label = line.split()
        records.append(Record(filename=filename, label_id=int(label)))
    return records


def generate_dataset_list_as_pandas(
    data_dir: Path, target_realms: Optional[List[str]] = None
) -> pd.DataFrame:
    dfs = []
    for realm, data_split in iterate_target_dirs():
        if target_realms is not None:
            if realm not in target_realms:
                logger.debug(f"Ignoring this realm: {realm}")
                continue
        logger.info(f"Loading this realm: {realm}")
        record_txt_path = Path(data_dir, realm, data_split, omni_const.RECORD_TXT)
        records = load_record_txt(record_txt_path=record_txt_path)
        df = pd.DataFrame(records)
        df[omni_col.REALM] = realm
        df[omni_col.DATA_SPLIT] = data_split
        dfs.append(df)
    omni_df = pd.concat(dfs, axis=0)

    def _get_file_path(x: pd.Series) -> str:
        return str(
            get_file_path(data_dir, x[omni_col.REALM], x[omni_col.DATA_SPLIT], x[omni_col.FILENAME])
        )

    omni_df[omni_col.FILEPATH] = omni_df.apply(_get_file_path, axis=1)
    return omni_df


def align_label_name(
    omni_df: pd.DataFrame, trainid2name: Dict[str, Dict[str, str]]
) -> pd.DataFrame:
    def _align(x: pd.Series) -> str:
        return trainid2name[x[omni_col.REALM]][str(x[omni_col.LABEL_ID])]

    omni_df[omni_col.LABEL_NAME] = omni_df.apply(_align, axis=1)
    return omni_df


def load_train_csv(
    data_dir: Path,
    load_target_realms: Optional[List[str]] = None,
) -> pd.DataFrame:
    trainid2name_path = data_dir / omni_const.TRAIN_ID_2_NAME
    omni_df = generate_dataset_list_as_pandas(data_dir=data_dir, target_realms=load_target_realms)
    trainid2name = json.loads(trainid2name_path.read_text())
    omni_df = align_label_name(omni_df=omni_df, trainid2name=trainid2name)
    return omni_df
