import logging

import numpy as np
import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def convert_embed_size_with_average_pooling(embeddings: np.ndarray, target_size: int) -> np.ndarray:
    """convert embedding vectors last dim into given target_size with avg pooling"""
    need_squeeze = False
    if embeddings.ndim == 0:
        raise ValueError("embeddings.ndim should be > 0 but 0 is given")
    elif embeddings.ndim == 1:
        embeddings = embeddings[np.newaxis]
        need_squeeze = True

    embeddings = torch.tensor(embeddings)
    embeddings = F.adaptive_avg_pool1d(embeddings, target_size)

    if need_squeeze:
        embeddings = embeddings.squeeze(0)

    return embeddings.numpy()


if __name__ == "__main__":
    target_size = 64
    in_features = 512
    num_samples = 1000

    embeddings = np.random.random((num_samples, in_features)).astype(np.float32)
    output = convert_embed_size_with_average_pooling(embeddings=embeddings, target_size=target_size)
    print(output.shape)
    embeddings = np.random.random((in_features,)).astype(np.float32)
    output = convert_embed_size_with_average_pooling(embeddings=embeddings, target_size=target_size)
    print(output.shape)
