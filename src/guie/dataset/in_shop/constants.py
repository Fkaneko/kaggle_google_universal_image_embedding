from dataclasses import dataclass


@dataclass(frozen=True)
class InShop:
    # data split
    TRAIN: str = "train"
    QUERY: str = "query"
    INDEX: str = "gallery"

    # annotation
    TRAIN_TXT: str = "Eval/list_eval_partition.txt"
    IMAGE_DIR_NAME: str = "img"

    # train txt field
    IMAGE_NAME: str = "image_name"
    LABEL_ID: str = "item_id"
    DATA_SPLIT: str = "evaluation_status"

    # additional field
    FILEPATH: str = "filepath"  # class group
    IMAGE_ID: str = "image_id"  # class group


in_shop_const = InShop()
