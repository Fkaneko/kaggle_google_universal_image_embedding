from dataclasses import dataclass


@dataclass
class GLR:
    INDEX: str = "index"
    TEST: str = "test"
    TRAIN: str = "train"
    ID: str = "id"

    RETRIEVAL_SOLUTION_CSV: str = "retrieval_solution_v2.1.csv"
    EVAL_KNN_5: int = 5

    FILEPATH: str = "filepath"
    LABEL_ID: str = "landmark_id"


glr_const = GLR()
