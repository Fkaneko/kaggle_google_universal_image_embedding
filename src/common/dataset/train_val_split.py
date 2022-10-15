import logging
from typing import Optional

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

logger = logging.getLogger(__name__)


def make_split(
    df: pd.DataFrame,
    n_splits: int = 3,
    target_key: str = "target",
    group_key: Optional[str] = None,
    is_reset_index: bool = True,
    shuffle: bool = True,
    how: str = "stratified",
) -> pd.DataFrame:

    if shuffle:
        df = df.sample(frac=1.0)

    if is_reset_index:
        df.reset_index(drop=True, inplace=True)
    df["fold"] = -1

    split_keys = {"X": df, "y": df[target_key]}
    if how == "stratified":
        cv = StratifiedKFold(n_splits=n_splits)
    elif how == "group":
        assert group_key is not None
        cv = GroupKFold(n_splits=n_splits)
        split_keys.update({"groups": df[group_key]})
    elif how == "stratified_group":
        assert group_key is not None
        cv = StratifiedGroupKFold(n_splits=n_splits)
        split_keys.update({"groups": df[group_key]})
    else:
        raise ValueError(f"how: {how}")

    for i, (train_idx, valid_idx) in enumerate(cv.split(**split_keys)):
        df.loc[valid_idx, "fold"] = i

    logger.debug(f">> check split with target\n {pd.crosstab(df.fold, df[target_key])}")
    if group_key is not None:
        logger.debug(f">> check split with target\n {pd.crosstab(df.fold, df[group_key])}")

    return df
