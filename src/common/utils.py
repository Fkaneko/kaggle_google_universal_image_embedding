import logging
import os
import random
import sys
from typing import Optional

import numpy as np
import requests

log = logging.getLogger(__name__)


def set_random_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)


def request_and_save(url: str, is_save: bool = True, save_path: Optional[str] = None) -> None:
    """
    download and save it on save_path
    if thre is cache, automatically use cache
    """
    if save_path is None:
        save_path = url.split("/")[-1]

    if os.path.exists(save_path):
        log.info(f"use cached file for {url}")
        return save_path

    log.info(f"download from {url}")
    log.info(f"and save it  {save_path}")
    r = requests.get(url=url)
    # for 404 error
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)


def set_logger(logger: logging.Logger, log_level: int = logging.INFO) -> None:
    logger.setLevel(log_level)

    # create console handler and set level to debug
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
