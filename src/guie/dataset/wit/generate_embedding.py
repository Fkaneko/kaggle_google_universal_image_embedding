import logging
from functools import partial
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

from src.guie.dataset.wit.preprocess_text import WIT_EMBEDDING

logger = logging.getLogger(__name__)


CAPTION_EMBEDDING = "caption_embedding"
IMAGE_NORM_EMBEDDING = "image_norm_embedding"
CONCAT_EMBEDDING = "concat_embedding"

ALL_EMBEDDINGS = (CONCAT_EMBEDDING, IMAGE_NORM_EMBEDDING, CAPTION_EMBEDDING)
NUM_PROCESS_FOR_SENTENCE_EMBED = 1


def encode_caption_with_transformer(
    examples, caption_column: str, transformer: SentenceTransformer, batch_size: int = 128
):
    sentences = examples[caption_column]
    embeddings = transformer.encode(sentences, batch_size=batch_size)
    examples[CAPTION_EMBEDDING] = embeddings
    return examples


def normalize_embedding(
    embeds: np.ndarray, return_as_list: bool = True
) -> Union[np.ndarray, List[np.ndarray]]:
    embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)
    if return_as_list:
        embeds = [embed.squeeze(0) for embed in np.split(embeds, embeds.shape[0], axis=0)]
    return embeds


def normalize_image_embedding(examples):
    embeds = np.stack(examples[WIT_EMBEDDING]).astype(np.float32)
    examples[IMAGE_NORM_EMBEDDING] = normalize_embedding(embeds=embeds, return_as_list=True)
    return examples


def concat_embedding(examples):
    image_embeds = np.stack(examples[IMAGE_NORM_EMBEDDING])
    caption_embeds = np.stack(examples[CAPTION_EMBEDDING])
    concat_embeds = np.concatenate([image_embeds, caption_embeds], axis=-1)
    # and then normalize
    examples[CONCAT_EMBEDDING] = normalize_embedding(embeds=concat_embeds, return_as_list=True)
    return examples


def caption_encode_pipeline(
    wit_dataset: Dataset,
    new_caption_column: str,
    transformer_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 128,
) -> Dataset:

    transformer = SentenceTransformer(transformer_name)

    def _pipeline(examples):
        _encode_caption_with_transformer = partial(
            encode_caption_with_transformer,
            caption_column=new_caption_column,
            batch_size=batch_size,
            transformer=transformer,
        )
        examples = _encode_caption_with_transformer(examples)
        examples = concat_embedding(examples)
        return examples

    wit_dataset = wit_dataset.map(
        _pipeline,
        batched=True,
        num_proc=NUM_PROCESS_FOR_SENTENCE_EMBED,
        desc="Encode caption with sentence_transformers",
    )
    return wit_dataset


def calc_cos_similarity(query: np.ndarray, samples: np.ndarray) -> np.ndarray:
    if query.ndim == 1:
        query = query[np.newaxis]
    return ((cdist(query, samples, "cosine") - 1) * -1).squeeze(0)


def visualize_knn_result_for_3_queries(
    k: int,
    wit_dataset: Dataset,
    query_range: range = range(0, 20),
    clustering: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> None:
    for query_index in query_range:
        for target_embedding in ALL_EMBEDDINGS:
            visualize_knn_result(
                k=k,
                knn_target_caption=target_embedding,
                wit_dataset=wit_dataset,
                query_range=[query_index],
                clustering=clustering,
            )


def visualize_knn_result(
    k: int,
    knn_target_caption: str,
    wit_dataset: Dataset,
    query_range: list = list(range(0, 20)),
    clustering: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> None:
    figsize = (12, 6)
    nrows = 2
    ncols = k // nrows

    for query_index in query_range:
        queries = {}
        for target_embedding in ALL_EMBEDDINGS:
            queries[target_embedding] = np.array(wit_dataset[query_index][target_embedding]).astype(
                np.float32
            )

        scores_image, retrieved_examples_image = wit_dataset.get_nearest_examples(
            knn_target_caption, queries[knn_target_caption], k=k
        )

        cos_similarity = calc_cos_similarity(
            query=queries[CONCAT_EMBEDDING], samples=retrieved_examples_image[CONCAT_EMBEDDING]
        )
        image_cos_similarity = calc_cos_similarity(
            query=queries[IMAGE_NORM_EMBEDDING],
            samples=retrieved_examples_image[IMAGE_NORM_EMBEDDING],
        )
        caption_cos_similarity = calc_cos_similarity(
            query=queries[CAPTION_EMBEDDING], samples=retrieved_examples_image[CAPTION_EMBEDDING]
        )
        if clustering is not None:
            print(np.stack(retrieved_examples_image[CONCAT_EMBEDDING]).shape)
            labels = clustering(np.stack(retrieved_examples_image[CONCAT_EMBEDDING]))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for neig_index in range(k):
            row_index = neig_index // ncols
            col_index = neig_index % ncols
            axes[row_index][col_index].imshow(retrieved_examples_image["image"][neig_index])
            title = f"{cos_similarity[neig_index]:>3.2f}, I:{image_cos_similarity[neig_index]:>3.2f}, C:{caption_cos_similarity[neig_index]:>3.2f}"
            if clustering is not None:
                title += " " + str(labels[neig_index])
            axes[row_index][col_index].set_title(title)
        fig.suptitle(knn_target_caption, fontsize=16)
        plt.show()
