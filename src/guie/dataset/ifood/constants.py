from dataclasses import dataclass


@dataclass(frozen=True)
class iFood:
    # data split
    TRAIN: str = "train"
    VAL: str = "val"

    TRAIN_DIR_NAME: str = "train_set"
    VAL_DIR_NAME: str = "val_set"

    # annotation files
    TRAIN_CSV: str = "train_labels.csv"
    VAL_CSV: str = "val_labels.csv"

    LABEL_ID_TO_NAME_FILE: str = "class_list.txt"

    # csv field
    FILENAME: str = "img_name"
    LABEL_ID: str = "label"

    # additional field
    FILEPATH: str = "filepath"  #
    LABEL_NAME: str = "label_name"  #


ifood_const = iFood()
