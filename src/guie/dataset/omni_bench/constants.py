from dataclasses import dataclass


@dataclass(frozen=True)
class Omni:
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"
    LABEL: str = "str"

    ID: str = "id"

    RECORD_TXT = "records.txt"
    TRAIN_ID_2_NAME = "trainid2name.json"
    IMAGE_DIR_NAME = "images"
    FILEPATH: str = "filepath"


omni_const = Omni()
