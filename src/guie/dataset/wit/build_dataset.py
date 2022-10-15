import logging
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
from datasets import Dataset, DatasetInfo, concatenate_datasets, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from src.guie.dataset.wit.generate_embedding import normalize_image_embedding
from src.guie.dataset.wit.preprocess_text import (
    HUGGING_FACE_WIT,
    WIT_ATTR_CAPTION,
    WIT_FEATURE,
    WIT_IMAGE,
    WIT_IMAGE_URL,
    WITExample,
    concat_captions,
    delete_lang_prefix_for_en,
    filter_non_target_lang_caption,
)

logger = logging.getLogger(__name__)

WIT_ARROW_FILE = "parquet-train.arrow"
BACKGROUND_COLOR = (255, 255, 255)

CAN_NOT_OPEN_IMAGE_URLS = (
    "https://upload.wikimedia.org/wikipedia/commons/3/3e/AMDT-amblem.png",
    "https://upload.wikimedia.org/wikipedia/commons/d/d7/World_Scout_Jamboree_-_Punch_cartoon_-_Project_Gutenberg_eText_16628.png",
    "https://upload.wikimedia.org/wikipedia/commons/8/86/Bishop_%28chess%29_movements.gif",
)


def tokenize_captions(examples, caption_column: str, tokenizer: Callable[[str], torch.LongTensor]):
    captions = [tokenizer(caption) for caption in examples[caption_column]]
    examples["input_ids"] = captions
    # examples["attention_mask"] = text_inputs.attention_mask
    return examples


def convert_to_rgb(image: Image) -> Image:
    # color model: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
    if (image.getbands()[-1] in ["A", "a"]) | (image.mode == "P"):
        image = image.convert("RGBA")

    if image.mode == "RGB":
        return image
    elif image.mode == "RGBA":
        # from : http://stackoverflow.com/a/9459208/284318
        image.load()  # needed for split()
        new_rgb_image = Image.new("RGB", image.size, BACKGROUND_COLOR)
        new_rgb_image.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return new_rgb_image
    else:
        return image.convert("RGB")


def transform_images(examples, image_column: str, image_transformations: torch.nn.Module):
    # There are Grayscale images on WIT, need convert("RGB")
    # images = [pil_to_tensor(image_pil.convert("RGBA")) for image_pil in examples[image_column]]
    images = [pil_to_tensor(convert_to_rgb(image_pil)) for image_pil in examples[image_column]]
    examples["pixel_values"] = [image_transformations(image) for image in images]
    return examples


# def filter_corrupt_images(examples: WITExample, image_column: str):
#     """remove problematic images"""
#     valid_images = []
#     try:
#         image = examples[image_column]
#         valid_images.append(True)
#     except Exception:
#         logger.info("found corrupted images")
#         valid_images.append(False)
#     return valid_images


def filter_manual_corrupt_images(wit_dataset: Dataset) -> List[str]:
    """remove problematic images"""
    invalid_image_urls = []
    indices = []
    wit_dataset_wo_image = wit_dataset.remove_columns(WIT_IMAGE)
    for row_index in tqdm(range(wit_dataset.num_rows)):
        try:
            image = wit_dataset[row_index][WIT_IMAGE]
        except:
            url = wit_dataset_wo_image[row_index][WIT_IMAGE_URL]
            print(f"found corrupted images, row_index{row_index}, url:{url}")
            invalid_image_urls.append(url)
            indices.append(row_index)

    for idx, url in zip(indices, invalid_image_urls):
        print(f"{idx}, {url}")

    return invalid_image_urls


def load_wit_dataset(
    cache_dir: Path,
    data_files: Optional[Union[Dict[str, str], List[str]]] = None,
    arrow_dir: Optional[Path] = None,
    target_indices: Optional[List[int]] = None,
) -> Dataset:
    if arrow_dir is not None:
        logger.info("load from cached arrow file for dataset generation")
        # load from cached arrow_file
        arrow_file = arrow_dir / WIT_ARROW_FILE
        dataset_info = DatasetInfo.from_directory(str(arrow_dir))
        wit_dataset = Dataset.from_file(
            filename=str(arrow_file),
            info=dataset_info,
        )
    else:
        # load from original
        wit_dataset = load_dataset(
            HUGGING_FACE_WIT,
            data_files=data_files,
            split="train",
            cache_dir=str(cache_dir),
            ignore_verifications=True,  # skip size check for partial data use
        )
    if target_indices is not None:
        orig_num_example = wit_dataset.num_rows
        wit_dataset = wit_dataset.select(target_indices)
        logger.info(f"slice dataset, original num: {orig_num_example} -> {wit_dataset.num_rows}")
    return wit_dataset


def dataset_pipeline(
    wit_dataset: Dataset,
    new_caption_column: str,
    cache_dir: Path,
    tokenizer: Callable[[str], torch.LongTensor],
    image_transformations: torch.nn.Module,
    data_files: Optional[Union[Dict[str, str], List[str]]] = None,
    arrow_dir: Optional[Path] = None,
    num_proc: int = 6,
    preprocess_text_on_the_fly: bool = True,
    target_indices: Optional[List[int]] = None,
    skip_preprocess_image: bool = False,
) -> Dataset:
    def _pipeline_text(examples):
        # text preprocess
        _delete_lang_prefix = partial(
            delete_lang_prefix_for_en,
            new_caption_column=new_caption_column,
        )
        _concat_captions = partial(
            concat_captions,
            caption_column=new_caption_column,
        )
        _tokenize_captions = partial(
            tokenize_captions,
            tokenizer=tokenizer,
            caption_column=new_caption_column,
        )
        # compose
        examples = _delete_lang_prefix(examples)
        examples = _concat_captions(examples)
        examples = _tokenize_captions(examples)
        if not preprocess_text_on_the_fly:
            examples = normalize_image_embedding(examples)
        return examples

    def _pipeline_image(examples):
        # image preprocess
        _transform_images = partial(
            transform_images, image_column=WIT_IMAGE, image_transformations=image_transformations
        )
        examples = _transform_images(examples)
        return examples

    def _pipeline(examples):
        examples = _pipeline_text(examples)
        examples = _pipeline_image(examples)
        return examples

    # filter non targets caption samples
    wit_dataset = wit_dataset.filter(
        filter_non_target_lang_caption,
        batched=True,
        num_proc=num_proc,
        desc="Filtering invalid caption samples",
    )

    def filter_invalid_images(examples) -> List[bool]:
        return [url not in CAN_NOT_OPEN_IMAGE_URLS for url in examples[WIT_IMAGE_URL]]

    orig_num = wit_dataset.num_rows
    wit_dataset = wit_dataset.filter(
        filter_invalid_images,
        batched=True,
        num_proc=num_proc,
        desc="Filtering invalid image samples",
    )
    print(f"after filtering invalid image samples: {orig_num} -> {wit_dataset.num_rows}")
    # if invalid_image_urls is None:
    #     invalid_image_urls = filter_manual_corrupt_images(wit_dataset=wit_dataset)
    #     logging.info("save invalid_image_urls")
    #     np.save("./invalid_image_urls.npy", invalid_image_urls)

    # wit_dataset = wit_dataset.filter(lambda x: x[WIT_IMAGE_URL] in invalid_image_urls)

    if preprocess_text_on_the_fly:
        target_columns = [WIT_IMAGE, WIT_ATTR_CAPTION, WIT_FEATURE]
        wit_dataset = wit_dataset.remove_columns(
            [col for col in wit_dataset.column_names if col not in target_columns]
        )
        wit_dataset.set_transform(_pipeline)
        return wit_dataset

    wit_dataset = wit_dataset.map(
        _pipeline_text,
        remove_columns=[col for col in wit_dataset.column_names if col != new_caption_column],
        batched=True,
        num_proc=num_proc,
        desc="Fill caption with sub caption if it is empty",
    )
    orig_dataset = load_wit_dataset(
        cache_dir=cache_dir,
        arrow_dir=arrow_dir,
        data_files=data_files,
        target_indices=target_indices,
    )

    orig_dataset = orig_dataset.filter(
        filter_non_target_lang_caption,
        batched=True,
        num_proc=num_proc,
        desc="Filtering invalid caption samples",
    )
    orig_dataset = orig_dataset.remove_columns(
        [col for col in orig_dataset.column_names if col != WIT_IMAGE]
    )
    wit_dataset = concatenate_datasets([orig_dataset, wit_dataset], axis=1)

    if not skip_preprocess_image:
        # always process image on the fly
        wit_dataset.set_transform(_pipeline_image)
    return wit_dataset


def collate_fn_clip(
    examples: WITExample, image_only: bool = False, input_ids_as_tensor: bool = False
) -> dict:
    # image is already torch.tensor during transform_images with torch.jit
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    if image_only:
        return {"pixel_values": pixel_values}

    # input_ids is still python values
    if input_ids_as_tensor:
        input_ids = torch.stack([example["input_ids"] for example in examples])
    else:
        input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    # attention_mask = torch.tensor(
    #     [example["attention_mask"] for example in examples], dtype=torch.long
    # )
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        # "attention_mask": attention_mask,
        "return_loss": True,
    }


def get_dataloader(
    ds: Dataset,
    batch_size: int,
    num_workers: int = 6,
    pin_memory: bool = True,
    is_train: bool = False,
    image_only: bool = True,
) -> DataLoader:
    collate_fn = partial(collate_fn_clip, image_only=image_only)
    if is_train:
        sampler = RandomSampler(ds)
    else:
        sampler = SequentialSampler(ds)
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
