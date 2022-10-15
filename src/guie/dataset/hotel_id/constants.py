from dataclasses import dataclass


@dataclass(frozen=True)
class HotelID:
    # data split
    TRAIN: str = "train"

    TRAIN_DIR_NAME: str = "train_images"
    # additional field
    LABEL_ID: str = "label_id"
    IMAGE_ID: str = "image_id"  #
    FOLDER_NAME: str = "folder_name"  #
    FILEPATH: str = "filepath"  #


hotel_id_const = HotelID()
