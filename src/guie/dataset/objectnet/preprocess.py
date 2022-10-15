import logging
from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .constants import objectnet_const

logger = logging.getLogger(__name__)


def get_file_path(data_dir: Path, filename: str) -> Path:
    return Path(data_dir, filename)


def make_image_info_from_path(all_image_paths: Iterator[Path]) -> List[Dict[str, str]]:
    image_infos = []
    for path in all_image_paths:
        image_infos.append(
            {
                objectnet_const.IMAGE_ID: path.stem,
                objectnet_const.FOLDER_NAME: path.parent.name,
                objectnet_const.FILEPATH: str(path),
            }
        )
    return image_infos


def make_query_id(objectnet_df: pd.DataFrame) -> pd.DataFrame:
    objectnet_df[objectnet_const.IMAGE_ID] = (
        objectnet_df[objectnet_const.FOLDER_NAME].astype(str)
        + "_"
        + objectnet_df[objectnet_const.IMAGE_ID].astype(str)
    )
    return objectnet_df


def fix_jpeg_filepath(objectnet_df: pd.DataFrame) -> pd.DataFrame:
    logger.warn("fix ObjectNet jpeg filepath")
    objectnet_df[objectnet_const.FILEPATH] = objectnet_df[objectnet_const.FILEPATH].apply(
        lambda x: str(Path(x).parent).replace("/images/", "/images_jpeg/")
        + "/"
        + (Path(x).stem + ".jpg")
    )
    return objectnet_df


def load_train_csv(data_dir: Path, is_jpeg_dataset: bool = True) -> pd.DataFrame:
    if is_jpeg_dataset:
        csv_path = data_dir / objectnet_const.JPEG_DATA_CSV
        objectnet_df = pd.read_csv(csv_path)
        logger.info(f"load objectnet JPEG dataset, num_samples: {len(objectnet_df)}")
        objectnet_df = fix_jpeg_filepath(objectnet_df=objectnet_df)
    else:
        image_dir = data_dir / objectnet_const.IMAGE_DIR_NAME
        all_image_paths = image_dir.glob("**/*.png")
        image_infos = make_image_info_from_path(all_image_paths=all_image_paths)
        logger.info(f"found {len(image_infos)}, images")

        objectnet_df = pd.DataFrame(image_infos)

    objectnet_df[objectnet_const.LABEL_ID] = objectnet_df[objectnet_const.FOLDER_NAME]
    return objectnet_df


def convert_into_jpeg_dataset(
    objectnet_df: pd.DataFrame,
    data_dir: Path,
    upper_limit_image_size: int = 512,
    jpeg_quality: int = 75,
) -> pd.DataFrame:
    save_root_dir = data_dir / objectnet_const.IMAGE_JPEG_DIR_NAME
    if not save_root_dir.exists():
        save_root_dir.mkdir()

    new_paths = []
    image_widths = []
    image_heights = []
    for row in tqdm(objectnet_df.itertuples(), total=len(objectnet_df)):
        filepath = Path(getattr(row, objectnet_const.FILEPATH))
        save_class_dir = save_root_dir / getattr(row, objectnet_const.FOLDER_NAME)
        if not save_class_dir.exists():
            save_class_dir.mkdir()

        with filepath.open("rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            min_image_size = min(list(img.size))
            if min_image_size > upper_limit_image_size:
                scale = upper_limit_image_size / min_image_size
                new_image_size = int(scale * img.size[0]), int(scale * img.size[1])
                logger.debug(f"{img.size} -> {new_image_size}")
                img = img.resize(new_image_size, Image.Resampling.BICUBIC)

            save_path = save_class_dir / (getattr(row, objectnet_const.IMAGE_ID) + ".jpg")
            img.save(save_path, quality=jpeg_quality)
            image_heights.append(img.size[1])
            image_widths.append(img.size[0])
        new_paths.append(str(save_path))

    # save csv as jpeg dataset
    objectnet_df[objectnet_const.ORIGINAL_PNG_FILEPATH] = objectnet_df[
        objectnet_const.FILEPATH
    ].copy()
    objectnet_df[objectnet_const.FILEPATH] = new_paths
    objectnet_df[objectnet_const.IMAGE_WIDTH] = image_widths
    objectnet_df[objectnet_const.IMAGE_HEIGHT] = image_heights
    csv_save_path = data_dir / objectnet_const.JPEG_DATA_CSV
    logger.info(f"save jpeg data csv at: {str(csv_save_path)}")
    objectnet_df.to_csv(csv_save_path, index=False)
    return objectnet_df
