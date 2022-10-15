import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from torch import nn

from .build_dataset import get_dataloader, get_glr_test_dataset
from .constants import glr_const
from .glr_metrics.evaluation import calc_metric, generate_retrieval_predictions

logger = logging.getLogger(__name__)


def eval_glr(
    data_dir: Path,
    image_transformations: nn.Module,
    model: nn.Module,
    batch_size: int = 256,
    arrow_dir: Optional[Path] = None,
    update_eval_dataset: bool = False,
    num_workers: int = 6,
    num_index_data: int = 60000,
    dim_reduction_mapper: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_mode: str = "mP",
    knn_samples: int = 5,
) -> Tuple[float, float, float]:
    glr_dataset, solution_index = get_glr_test_dataset(
        data_dir=data_dir,
        image_transformations=image_transformations,
        num_index_data=num_index_data,
        arrow_dir=arrow_dir,
        update_eval_dataset=update_eval_dataset,
    )
    logger.info(f"GLR evaluation dataset: {glr_dataset}")
    index_image_dataloader = get_dataloader(
        ds=glr_dataset[glr_const.INDEX], batch_size=batch_size, num_workers=num_workers
    )
    test_image_dataloader = get_dataloader(
        ds=glr_dataset[glr_const.TEST], batch_size=batch_size, num_workers=num_workers
    )

    # start prediction & evaluation
    predictions = generate_retrieval_predictions(
        model=model,
        index_image_dataloader=index_image_dataloader,
        test_image_dataloader=test_image_dataloader,
        dim_reduction_mapper=dim_reduction_mapper,
        image_id_key=glr_const.ID,
        knn_samples=knn_samples,
    )
    mean_ap_public = calc_metric(
        predictions=predictions,
        solution=solution_index.public_solution,
        max_predictions=knn_samples,
        metric_mode=metric_mode,
    )
    mean_ap_private = calc_metric(
        predictions=predictions,
        solution=solution_index.private_solution,
        max_predictions=knn_samples,
        metric_mode=metric_mode,
    )
    mean_ap = (mean_ap_public + mean_ap_private) / 2.0
    logger.info(f"GLR: mean ap: {mean_ap}")
    return mean_ap_public, mean_ap_private, mean_ap
