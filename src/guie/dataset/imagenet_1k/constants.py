from dataclasses import dataclass


@dataclass(frozen=True)
class ImageNet1K:
    TRAIN: str = "train"
    VAL: str = "validation"
    TEST: str = "test"

    IMAGE: str = "image"
    LABEL: str = "label"

    NUM_CLASSES: int = 1000
    NUM_TRAIN_SAMPLES: int = 1281167
    NUM_VAL_SAMPLES: int = 50000

    # this competition specific constants
    IS_OOD: str = "is_out_of_domain"


in_1k_const = ImageNet1K()
