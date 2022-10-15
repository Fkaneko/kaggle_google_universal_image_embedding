from dataclasses import dataclass


@dataclass(frozen=True)
class FOOD101:
    # data split
    TRAIN: str = "train"
    TEST: str = "test"

    # added split
    QUERY: str = "query"
    INDEX: str = "index"

    # annotation
    TRAIN_TXT: str = "meta/train.txt"
    TEST_TXT: str = "meta/test.txt"
    IMAGE_DIR: str = "images"

    # train/test txt field
    LABEL_NAME: str = "label_name"
    IMAGE_ID: str = "image_id"

    # additional field
    LABEL_ID: str = "label_id"
    IMAGE_NAME: str = "image_name"

    FILEPATH: str = "filepath"
    DATA_SPLIT: str = "data_split"


food_101_const = FOOD101()
