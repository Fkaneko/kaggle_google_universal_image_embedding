import io
import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import datasets
import torch
from datasets import Dataset, DatasetDict
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import pil_to_tensor

logger = logging.getLogger(__name__)
RESIZED_IMG_KEY = "resized_pil_image"
DEFAULT_SIZE = (224, 224)


def transform_images(
    examples: dict,
    image_transformations: Callable[[torch.Tensor], torch.Tensor],
    filepath_key: str = "filepath",
) -> dict:
    if RESIZED_IMG_KEY in examples.keys():
        pixel_values = [pil_to_tensor(image_file) for image_file in examples[RESIZED_IMG_KEY]]
    else:
        pixel_values = [
            read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[filepath_key]
        ]
    examples["pixel_values"] = [image_transformations(pixel_value) for pixel_value in pixel_values]
    return examples


def read_and_resize_image(img_filepath: str, new_image_size: tuple = (224, 224)) -> Any:
    with open(img_filepath, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize(new_image_size, Image.Resampling.BICUBIC)
    return img


def read_image_file(
    img_filepath: str, new_image_size: tuple = (224, 224), as_pillow: bool = True
) -> Union[Any, bytes]:
    img = read_and_resize_image(img_filepath, new_image_size)
    if as_pillow:
        return img

    with io.BytesIO() as temp_img:
        img.save(temp_img, format="JPEG")
        image_bytes = temp_img.getvalue()
    return image_bytes


def resize_images(
    examples: dict,
    filepath_key: str = "filepath",
    resized_size: Tuple[int, int] = (224, 224),
) -> dict:
    pil_image = [
        read_and_resize_image(image_file, new_image_size=resized_size)
        for image_file in examples[filepath_key]
    ]
    examples[RESIZED_IMG_KEY] = pil_image
    return examples


def resize_and_save_dataset_as_arrow(
    ds: Union[Dataset, DatasetDict],
    save_path: Path,
    filepath_key: str = "filepath",
    resized_size: Optional[Tuple[int, int]] = None,
    num_proc: int = 6,
    keep_in_memory: bool = True,
) -> None:
    if resized_size is not None:
        _transform_images = partial(
            resize_images, filepath_key=filepath_key, resized_size=resized_size
        )
        ds = ds.map(
            _transform_images,
            batched=True,
            num_proc=num_proc,
            desc="resize image",
            keep_in_memory=keep_in_memory,
        )
    logger.info(f"save dataset as arrow: {str(save_path)}")
    ds = ds.cast_column(RESIZED_IMG_KEY, datasets.Image())
    ds.save_to_disk(str(save_path))


if __name__ == "__main__":
    sample_image_path = "./CLIP.png"
    image_bytes = read_image_file(img_filepath=sample_image_path, as_pillow=False)
    print(image_bytes)
    with io.BytesIO() as temp_img:
        temp_img.write(image_bytes)
        img = Image.open(temp_img)
        img.save("./clip_2_resize.jpg")

    img_pillow = read_image_file(img_filepath=sample_image_path, as_pillow=True)
    print(img_pillow)
    img_pillow.save("./clip_pill.jpg")
