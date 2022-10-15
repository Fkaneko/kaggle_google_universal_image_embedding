from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Products10K:
    # data split
    TRAIN: str = "train"
    TEST: str = "test"

    # added split
    QUERY: str = "query"
    INDEX: str = "index"

    # annotation
    TRAIN_CSV: str = "train.csv"
    TEST_CSV: str = "test_kaggletest.csv"

    # csv field
    NAME: str = "name"  # filename
    LABEL: str = "class"
    GROUP: str = "group"  # class group

    # additional field
    FILEPATH: str = "filepath"  #
    LABEL_ID: str = "label_id"  # class renamed field
    DATA_SPLIT: str = "data_split"
    IMAGE_ID: str = "image_id"
    TRAIN_OR_TEST: str = "train_or_test"

    QUERY_LABEL_IDS_LIST: str = "query_label_ids.csv"

    # group info
    # ?, food, home
    PACKAGE_GROUPS: Tuple[int, ...] = tuple(
        list(range(128, 130)) + list(range(149, 194)) + list(range(234, 316))
    )
    TOY_GROUPS: Tuple[int, ...] = tuple(range(194, 200))
    DOMAIN_NAME: str = "domain_name"
    PACKAGE: str = "package"
    TOY: str = "toy"
    OTHER: str = "other"


products_10k_const = Products10K()
