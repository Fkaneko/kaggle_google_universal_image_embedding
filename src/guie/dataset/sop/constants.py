from dataclasses import dataclass


@dataclass(frozen=True)
class SOP:
    # data split
    TRAIN: str = "train"
    TEST: str = "test"

    # added split
    QUERY: str = "query"
    INDEX: str = "index"

    # annotation
    TRAIN_TXT: str = "Ebay_train.txt"
    TEST_TXT: str = "Ebay_test.txt"

    # train/test txt field
    IMAGE_ID: str = "image_id"
    LABEL_ID: str = "class_id"
    SUPER_CLASS_ID: str = "super_class_id"
    SUPER_CLASS_NAME: str = "super_class_name"
    IMAGE_NAME: str = "path"

    # additional field
    FILEPATH: str = "filepath"
    DATA_SPLIT: str = "data_split"


sop_const = SOP()
