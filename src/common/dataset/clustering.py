import logging
from pathlib import Path
from typing import Any, Callable

import hdbscan
import joblib
import numpy as np
import umap

logger = logging.getLogger(__name__)


def set_umap_dimension_reduction(n_components: int = 5, for_visualize: bool = True) -> umap.UMAP:
    if for_visualize:
        return umap.UMAP()
    # parameter from https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py
    # low min_dist is recommended at https://umap-learn.readthedocs.io/en/latest/clustering.html
    return umap.UMAP(
        n_neighbors=15, n_components=n_components, min_dist=0.0, metric="cosine", low_memory=True
    )


def set_clustering_model() -> Any:
    # parameter from https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py
    return hdbscan.HDBSCAN(
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,  # need for un seen data prediction
    )


def clustering_from_dim_reduction(
    mapper: umap.UMAP, hdbscan_model: Any
) -> Callable[[np.ndarray], np.ndarray]:
    def _pipeline(input_array: np.ndarray) -> np.ndarray:
        input_reduced_array = mapper.transform(input_array)
        predictions, _ = hdbscan.approximate_predict(hdbscan_model, input_reduced_array)
        return predictions

    return _pipeline


def build_clustering(input_array: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:

    logger.info("start dimension reduction before clustering")
    mapper = set_umap_dimension_reduction(for_visualize=False)
    hdbscan_model = set_clustering_model()
    mapper = mapper.fit(input_array)
    input_reduced_array = mapper.transform(input_array)

    logger.info("start clustering")
    hdbscan_model.fit(input_reduced_array)
    logger.info(f"num unlabeled data at hdbscan {(hdbscan_model.labels_ == -1).sum()}")

    vis_mapper = set_umap_dimension_reduction(for_visualize=True)
    vis_mapper = vis_mapper.fit(input_array)
    umap.plot.points(vis_mapper, labels=hdbscan_model.labels_)

    clustering = clustering_from_dim_reduction(mapper=mapper, hdbscan_model=hdbscan_model)
    return clustering


def save_umap(mapper: umap.UMAP, filepath: Path) -> None:
    logger.info(f"save umap model at {str(filepath)})")
    joblib.dump(mapper, str(filepath))


def load_umap(filepath: Path) -> umap.UMAP:
    logger.info(f"load umap model from {str(filepath)})")
    return joblib.load(str(filepath))
