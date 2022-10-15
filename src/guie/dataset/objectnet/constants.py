from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectNet:
    # data split
    TRAIN: str = "train"

    DATA_SPLIT: str = "data_split"
    INDEX: str = "index"
    QUERY: str = "query"

    # annotation
    TRAIN_JSON: str = "MET_database.json"
    IMAGE_DIR_NAME: str = "images"
    LABEL_NAME_JSON: str = "mappings/folder_to_objectnet_label.json"
    TO_IN_1K_LABEL_JSON: str = "mappings/objectnet_to_imagenet_1k.json"
    IMAGE_JPEG_DIR_NAME: str = "images_jpeg"

    # json field
    LABEL_ID: str = "label_id"

    # additional field
    IMAGE_ID: str = "image_id"
    ORIGINAL_PNG_FILEPATH: str = "filepath_png"
    FOLDER_NAME: str = "folder_name"
    FILEPATH: str = "filepath"  #
    IMAGE_WIDTH: str = "width"  #
    IMAGE_HEIGHT: str = "height"  #

    JPEG_DATA_CSV: str = "images_jpeg/train_jpeg.csv"


objectnet_const = ObjectNet()
