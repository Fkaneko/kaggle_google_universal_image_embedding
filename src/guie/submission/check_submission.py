import logging

import numpy as np

logger = logging.getLogger(__name__)


def test_embedding_format(embedding: np.ndarray) -> None:
    assert embedding.shape == (64,), f"{embedding.shape}"
    assert embedding.dtype == np.float32, f"{embedding.dtype}"
    logger.info("pass embedding format test")
