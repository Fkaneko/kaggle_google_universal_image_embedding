import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, NewType, Optional

import numpy as np

from src.guie.model.clip.constant import CLIP_MAX_LEN

logger = logging.getLogger(__name__)

WITExample = NewType("WITExample", dict)

HUGGING_FACE_WIT = "wikimedia/wit_base"

EN_TAG = "English:"
LANG_TAG_SEPARATOR = ":"

# wit dataset column_names
WIT_IMAGE = "image"
WIT_IMAGE_URL = "image_url"
WIT_EMBEDDING = "embedding"
WIT_ATTR_CAPTION = "caption_attribution_description"
WIT_FEATURE = "wit_features"
# feature under WIT_FEATURE
WIT_REF_CAPTION = "caption_reference_description"
WIT_LANG = "language"
WIT_SECTION_CONTEXT = "context_section_description"
WIT_PAGE_CONTEXT = "context_page_description"
WIT_SECTION_TITLE = "hierarchical_section_title"
WIT_PAGE_TITLE = "page_title"


HTML_COLON_SLASH = "://"

CAPTION_PRIORITIES = (
    WIT_REF_CAPTION,
    WIT_ATTR_CAPTION,
    WIT_SECTION_CONTEXT,
    WIT_PAGE_CONTEXT,
    WIT_SECTION_TITLE,
    WIT_PAGE_TITLE,
)

# insert this word if we use CONTEXT
CONTEXT_SEPARATOR = "Details:"

__PUNCTUATION = [".", "!", "?"]
SENTENCE_END = "."


@dataclass
class IsCaptionFound:
    caption_reference_description: bool = False
    caption_attribution_description: bool = False
    context_section_description: bool = False
    context_page_description: bool = False
    hierarchical_section_title: bool = False
    page_title: bool = False

    def __call__(self, wit_col_name: str) -> bool:
        return getattr(self, wit_col_name)

    def set_found(self, wit_col_name: str) -> None:
        return setattr(self, wit_col_name, True)

    def need_title(self) -> bool:
        for wit_col_name in CAPTION_PRIORITIES:
            if getattr(self, wit_col_name):
                return False
        return True


def filter_non_target_lang_caption(examples, target_lang: List[str] = ["en"]) -> List[bool]:
    valid_captions = []
    for caption, wit_feat in zip(examples[WIT_ATTR_CAPTION], examples[WIT_FEATURE]):
        if caption is not None:
            # find english tag, so it should be valid sample
            if caption.find(EN_TAG) > -1:
                valid_captions.append(True)
                continue

        if len(set(wit_feat[WIT_LANG]) & set(target_lang)) > 0:
            valid_captions.append(True)
        else:
            valid_captions.append(False)
    return valid_captions


def remove_non_en_lang_caption(caption: str) -> str:
    if caption.find(LANG_TAG_SEPARATOR) == -1:
        # there are no lang tag, so nothing to be removed
        return caption

    new_caption = []
    for word_idx, word in enumerate(caption.split(" ")):
        if word.endswith(LANG_TAG_SEPARATOR):
            # something like, English: Title: is acceptable
            if word_idx == 0:
                logger.info(f"Successive tag, {word}")
                pass
            else:
                break
        new_caption.append(word)
    caption = " ".join(new_caption)
    return caption


def delete_lang_prefix_for_en(examples, new_caption_column: str) -> List[str]:
    assert new_caption_column != WIT_ATTR_CAPTION, f"{new_caption_column}"

    new_captions = []
    for caption in examples[WIT_ATTR_CAPTION]:
        if caption is not None:
            en_cap_pos = caption.find(EN_TAG)
            if en_cap_pos > -1:
                # make en caption is always first
                caption = caption[en_cap_pos:]
                caption = caption.replace(EN_TAG, "").strip()
                caption = remove_non_en_lang_caption(caption)

            caption = caption.strip()

        new_captions.append(caption)
    examples[new_caption_column] = new_captions
    return examples


def should_ignore_attribute_caption(
    caption_orig: str, caption_cleaned: Optional[str] = None
) -> bool:
    if caption_orig is None:
        return True

    if caption_cleaned is not None:
        if len(caption_cleaned.strip()) == 0:
            logger.error(f"orig:{caption_orig}, ")
            return True

    # delete : with HTML for LANG_TAG_SEPARATOR identification
    caption_wo_http_colon = caption_orig.replace(HTML_COLON_SLASH, "")
    if caption_wo_http_colon.find(LANG_TAG_SEPARATOR) > -1:
        if caption_wo_http_colon.find(EN_TAG) > -1:
            return False
        else:
            return True
    else:
        # TODO: need check is_english
        return False


def get_target_lang_index(target_lang: List[str], wit_feat: Dict[str, str]) -> List[int]:
    return [feat_idx for feat_idx, lang in enumerate(wit_feat[WIT_LANG]) if lang in target_lang]


def find_alternative_captions(wit_feat: Dict[str, str], target_lang: List[str]) -> List[str]:
    target_lang_idxs = get_target_lang_index(target_lang, wit_feat)
    priorities = [WIT_REF_CAPTION, WIT_SECTION_CONTEXT, WIT_PAGE_CONTEXT]
    for target_column in priorities:
        target_sub_captions = [wit_feat[target_column][feat_idx] for feat_idx in target_lang_idxs]
        if any(target_sub_captions):
            break
    return target_sub_captions


def replace_attribute_caption_with_ref_caption(
    examples, caption_column: str, target_lang: List[str] = ["en"]
) -> List[str]:
    def _abs_diff(caption: str, tokenization_lag_ratio: float = 1.2) -> int:
        if caption is None:
            return np.inf

        words = caption.split(" ")
        return int(abs(CLIP_MAX_LEN - len(words) * tokenization_lag_ratio))

    new_captions = []
    for caption, caption_orig, wit_feat in zip(
        examples[caption_column], examples[WIT_ATTR_CAPTION], examples[WIT_FEATURE]
    ):
        target_sub_captions = find_alternative_captions(wit_feat=wit_feat, target_lang=target_lang)
        if should_ignore_attribute_caption(caption_orig=caption_orig):
            if any(target_sub_captions):
                if len(target_sub_captions) > 1:
                    target_sub_captions = sorted(target_sub_captions, key=_abs_diff)
                caption = target_sub_captions[0]
            else:
                logger.info(
                    f"fail to find alternative captions so use main caption for this sample, {caption_orig}, {wit_feat}"
                )

            if caption is None:
                raise ValueError(
                    f"No available captions for this sample, {caption_orig}, {wit_feat}"
                )

        new_captions.append(caption)
    examples[caption_column] = new_captions
    return examples


def _is_punctuation_for_en(character: str) -> bool:
    return character in __PUNCTUATION


def _append_punctuation_if_empty_for_en(caption: str) -> str:
    if not _is_punctuation_for_en(character=caption[-1]):
        caption += __PUNCTUATION[0]
    return caption


def _concat_caption(new_caption: str, caption: str, separator: Optional[str] = None) -> str:
    # normalize strings
    caption = caption.strip()
    caption = _append_punctuation_if_empty_for_en(caption=caption)

    if len(new_caption) == 0:
        return caption

    if separator is not None:
        caption = " ".join([separator, caption])

    new_caption = " ".join([new_caption, caption])
    return new_caption


def concat_captions(examples, caption_column: str, target_lang: List[str] = ["en"]) -> List[str]:
    new_captions = []
    _NEW_CAPTION_INIT = ""
    for caption, caption_orig, wit_feat in zip(
        examples[caption_column], examples[WIT_ATTR_CAPTION], examples[WIT_FEATURE]
    ):
        new_caption = copy.copy(_NEW_CAPTION_INIT)
        is_caption_found = IsCaptionFound()

        target_lang_idxs = get_target_lang_index(target_lang, wit_feat)
        for target_column in CAPTION_PRIORITIES:
            if target_column == WIT_ATTR_CAPTION:
                if (not should_ignore_attribute_caption(caption_orig, caption_cleaned=caption)) & (
                    not is_caption_found(WIT_REF_CAPTION)
                ):
                    new_caption = _concat_caption(new_caption=new_caption, caption=caption)
                    is_caption_found.set_found(target_column)
            else:
                target_sub_captions = [
                    wit_feat[target_column][feat_idx] for feat_idx in target_lang_idxs
                ]
                if any(target_sub_captions):
                    if (target_column == WIT_PAGE_CONTEXT) & is_caption_found(WIT_SECTION_CONTEXT):
                        continue
                    elif target_column in [WIT_SECTION_TITLE, WIT_PAGE_TITLE]:
                        if is_caption_found.need_title():
                            logger.info(
                                f"Use title, {target_sub_captions}, caption_orig: {caption_orig}, wit_feat: {wit_feat}"
                            )
                        else:
                            continue

                    target_sub_captions = list(filter(lambda x: x is not None, target_sub_captions))
                    target_sub_captions = list(
                        filter(lambda x: len(x.strip()) > 0, target_sub_captions)
                    )
                    if len(target_sub_captions) == 0:
                        continue
                    elif len(target_sub_captions) > 1:
                        target_sub_captions = sorted(target_sub_captions, key=len, reverse=True)

                    separator = (
                        CONTEXT_SEPARATOR
                        if target_column in [WIT_SECTION_CONTEXT, WIT_PAGE_CONTEXT]
                        else None
                    )
                    new_caption = _concat_caption(
                        new_caption=new_caption, caption=target_sub_captions[0], separator=separator
                    )
                    is_caption_found.set_found(target_column)

        if new_caption == _NEW_CAPTION_INIT:
            logger.info(
                f"fail to find alternative captions so use attribution caption for this sample, {caption_orig}, {wit_feat}"
            )
            new_caption = caption

        if new_caption is None:
            raise ValueError(f"No available captions for this sample, {caption_orig}, {wit_feat}")

        new_captions.append(new_caption)
    examples[caption_column] = new_captions
    return examples
