import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

log = logging.getLogger(__name__)


def get_ml_ops_logger(
    conf: DictConfig,
    work_dir: Path,
    logger_name: str = "wandb",
) -> Any:
    name = work_dir.name
    project = conf.ml_ops.project
    if logger_name == "tensorboard":
        logger = TensorBoardLogger("tb_logs", name="my_model")
    elif logger_name == "wandb":
        if log.isEnabledFor(level=logging.DEBUG):
            project += "_debug"
        logger = WandbLogger(
            project=project,
            name=name,
            log_model=conf.ml_ops.log_model,
            save_dir=str(work_dir),
            tags=conf.ml_ops.tags,
        )
        logger.experiment.config["env/work_dir"] = str(work_dir)

    elif logger_name == "neptune":
        raise ValueError

    return logger
