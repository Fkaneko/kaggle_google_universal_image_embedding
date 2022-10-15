import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..constants import glr_const
from . import dataset_file_io, metrics

logger = logging.getLogger(__name__)


@dataclass
class SolutionIndex:
    public_solution: Dict[str, List[str]]
    private_solution: Dict[str, List[str]]
    target_indices: Dict[str, List[str]]


@dataclass
class EmbedOutputs:
    index_ids: np.ndarray
    index_embeds: torch.Tensor
    test_ids: np.ndarray
    test_embeds: torch.Tensor


def calc_embeddings(
    model: nn.Module, dataloader: DataLoader, image_id_key: str = "id", store_cpu: bool = True
) -> Tuple[np.ndarray, torch.Tensor]:
    embeds = []
    ids = []
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            images = batch["pixel_values"].cuda()
            embed = model(images)
            # if store_cpu:
            #     embed = embed.cpu()
            embeds.append(embed)
            ids.append(batch[image_id_key])

    embeds = torch.concat(embeds, dim=0)
    if store_cpu:
        embeds = embeds.cpu()
    ids = np.concatenate(ids, axis=0)
    return ids, embeds


# from https://www.kaggle.com/code/philculliton/landmark-retrieval-2020-shared-scoring-script/notebook?scriptVersionId=38406123to_
def calc_metric(
    predictions: Dict[str, List[str]],
    solution: Dict[str, List[str]],
    verbose: bool = False,
    max_predictions: int = 100,
    metric_mode: str = "mAP",
) -> float:
    relevant_predictions = {}

    for key in solution.keys():
        if key in predictions:
            relevant_predictions[key] = predictions[key]

    logger.info(f"Start calc metric: {metric_mode} @ {max_predictions}")
    # Mean average precision.
    if metric_mode == "mAP":
        metric = metrics.MeanAveragePrecision(
            relevant_predictions, solution, max_predictions=max_predictions
        )
    elif metric_mode == "mP":
        metric = metrics.MeanPrecisions(
            relevant_predictions, solution, max_predictions=max_predictions
        )
        logger.info(f"Mean Precisions until {max_predictions}:\n {metric}")
        metric = metric[max_predictions - 1]
    else:
        raise ValueError(f"unsupported metric, {metric_mode}")

    if verbose:
        print(f"{metric_mode}@{max_predictions}: {metric:.4f}")

    return metric


def get_solution_related_indices(
    solution_path: Path,
    num_index_data: Optional[int] = None,
    all_index_ids: Optional[List[str]] = None,
) -> SolutionIndex:
    print("Reading solution...")
    public_solution, private_solution, ignored_ids = dataset_file_io.ReadSolution(
        str(solution_path), dataset_file_io.RETRIEVAL_TASK_ID
    )
    used_test_images = list(set(public_solution.keys()) | set(private_solution.keys()))
    used_index_images = list(
        set([index for index_list in public_solution.values() for index in index_list])
        | set([index for index_list in private_solution.values() for index in index_list])
    )

    if num_index_data is not None:
        assert all_index_ids is not None
        # add index image
        used_index_images = list(
            set(used_index_images) | set(random.sample(all_index_ids, k=num_index_data))
        )
    target_indices = {glr_const.TEST: used_test_images, glr_const.INDEX: used_index_images}
    return SolutionIndex(
        public_solution=public_solution,
        private_solution=private_solution,
        target_indices=target_indices,
    )


def generate_retrieval_predictions(
    model: nn.Module,
    index_image_dataloader: DataLoader,
    test_image_dataloader: DataLoader,
    dim_reduction_mapper: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    is_euclid_distance: bool = True,
    image_id_key: str = "id",
    knn_samples: int = 100,
    return_embeddings: bool = False,
    calc_distance_with_cpu: bool = True,
) -> Dict[str, List[str]]:
    index_ids, index_embeds = calc_embeddings(
        model=model,
        dataloader=index_image_dataloader,
        image_id_key=image_id_key,
        store_cpu=calc_distance_with_cpu,
    )
    test_ids, test_embeds = calc_embeddings(
        model=model,
        dataloader=test_image_dataloader,
        image_id_key=image_id_key,
        store_cpu=calc_distance_with_cpu,
    )

    if is_euclid_distance:
        distances = torch.cdist(test_embeds, index_embeds, p=2.0)
    else:
        # -cos_sim as distances
        distances = -(test_embeds @ index_embeds.T)

    if not calc_distance_with_cpu:
        distances = distances.cpu()
    distances = distances.numpy()

    predicted_positions = np.argpartition(distances, knn_samples, axis=1)[:, :knn_samples]

    print("Converting to dict...", end="\t")
    predictions = {}
    for test_idx, query_id in enumerate(test_ids):
        nearest = [(index_ids[j], distances[test_idx, j]) for j in predicted_positions[test_idx]]
        nearest.sort(key=lambda x: x[1])
        prediction = [str(index_id) for index_id, d in nearest]
        predictions[query_id] = prediction

    if return_embeddings:
        embed_outputs = EmbedOutputs(
            index_ids=index_ids,
            index_embeds=index_embeds,
            test_ids=test_ids,
            test_embeds=test_embeds,
        )
        return predictions, embed_outputs
    else:
        return predictions
