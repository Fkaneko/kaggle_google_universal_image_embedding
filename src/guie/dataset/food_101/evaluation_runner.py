import logging
import pathlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from src.guie.dataset.glr.glr_metrics.evaluation import calc_metric, generate_retrieval_predictions

from .build_dataset import dataset_pipeline, get_dataloader, load_food_101_dataset
from .constants import food_101_const

logger = logging.getLogger(__name__)


def prepare_food_101_eval_input(
    arrow_dir: Optional[Path],
    batch_size: int,
    data_dir: Path,
    image_transformations: nn.Module,
    num_query_data: Optional[int],
    num_queries_per_class: Optional[int],
    num_index_data: Optional[int],
    num_workers: int,
    update_eval_dataset: bool,
) -> Tuple[DataLoader, DataLoader, Dict[str, List[str]]]:
    food_101_dataset, solution_index = load_food_101_dataset(
        data_dir=data_dir,
        num_index_data=num_index_data,
        num_query_data=num_query_data,
        num_queries_per_class=num_queries_per_class,
        arrow_dir=arrow_dir,
        update_eval_dataset=update_eval_dataset,
    )

    food_101_dataset = dataset_pipeline(
        food_101_dataset=food_101_dataset,
        data_dir=data_dir,
        image_transformations=image_transformations,
    )
    logger.info(f"food_101 evaluation dataset: {food_101_dataset}")
    index_image_dataloader = get_dataloader(
        ds=food_101_dataset[food_101_const.INDEX], batch_size=batch_size, num_workers=num_workers
    )
    test_image_dataloader = get_dataloader(
        ds=food_101_dataset[food_101_const.QUERY], batch_size=batch_size, num_workers=num_workers
    )

    # start prediction & evaluation
    return index_image_dataloader, test_image_dataloader, solution_index


def eval_food_101(
    data_dir: Path,
    image_transformations: nn.Module,
    model: nn.Module,
    batch_size: int = 256,
    arrow_dir: Optional[Path] = None,
    update_eval_dataset: bool = False,
    num_workers: int = 6,
    num_index_data: Optional[int] = None,
    num_query_data: Optional[int] = None,
    num_queries_per_class: Optional[int] = None,
    dim_reduction_mapper: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    knn_samples: int = 100,
    metric_mode: str = "mP",
) -> float:

    index_image_dataloader, test_image_dataloader, solution_index = prepare_food_101_eval_input(
        arrow_dir=arrow_dir,
        batch_size=batch_size,
        data_dir=data_dir,
        image_transformations=image_transformations,
        num_index_data=num_index_data,
        num_query_data=num_query_data,
        num_queries_per_class=num_queries_per_class,
        num_workers=num_workers,
        update_eval_dataset=update_eval_dataset,
    )
    predictions = generate_retrieval_predictions(
        model=model,
        index_image_dataloader=index_image_dataloader,
        test_image_dataloader=test_image_dataloader,
        dim_reduction_mapper=dim_reduction_mapper,
        image_id_key=food_101_const.IMAGE_ID,
        knn_samples=knn_samples,
    )
    mean_ap = calc_metric(
        predictions=predictions,
        solution=solution_index,
        max_predictions=knn_samples,
        metric_mode=metric_mode,
    )
    logger.info(f"food_101: mean ap: {mean_ap}")
    return mean_ap
