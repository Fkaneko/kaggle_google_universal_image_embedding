from dataclasses import dataclass


@dataclass(frozen=True)
class MET:
    # data split
    TRAIN: str = "train"
    TEST: str = "test"
    TEST_OTHER: str = "test_other"

    # added split
    QUERY: str = "query"
    INDEX: str = "index"

    # annotation
    TRAIN_JSON: str = "MET_database.json"
    TEST_JSON: str = "testset.json"

    # json field
    LABEL_ID: str = "id"
    FILENAME: str = "path"  # class group
    # test json field
    MET_ID: str = "MET_id"

    # additional field
    FILEPATH: str = "filepath"  #
    IMAGE_ID: str = "image_id"
    DATA_SPLIT: str = "data_split"


met_const = MET()
