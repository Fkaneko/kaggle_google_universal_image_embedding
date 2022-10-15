import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def make_label_id_successive(
    train_df: pd.DataFrame, class_label_name: str = "landmark_id"
) -> pd.DataFrame:
    class_label_original = train_df[class_label_name].to_numpy()
    class_label_ids = train_df[class_label_name].value_counts().index
    new_class_label_name = "_new_class_label"
    old2new = {
        old_label_id: new_label_id for new_label_id, old_label_id in enumerate(class_label_ids)
    }

    def _set_old2new(df: pd.DataFrame) -> pd.DataFrame:
        old_label_id = df.iloc[0][class_label_name]
        df[new_class_label_name] = old2new[old_label_id]
        return df

    train_df = train_df.groupby(class_label_name).apply(_set_old2new)
    train_df[class_label_name] = train_df[new_class_label_name]
    del train_df[new_class_label_name]
    train_df[class_label_name + "_original"] = class_label_original

    logger.info("reset class label id for starting with 0")
    return train_df


def sampling_per_class(
    train_df: pd.DataFrame, class_label_name: str, num_samples_per_class: int
) -> pd.DataFrame:
    dfs = []
    train_df = train_df.sample(frac=1.0)
    orig_len = len(train_df)
    for label_id, group_df in train_df.groupby(class_label_name):
        dfs.append(group_df.head(num_samples_per_class))

    train_df = pd.concat(dfs, axis=0)
    train_df = train_df.reset_index(drop=True)
    logger.info(
        f"apply per class {num_samples_per_class} sampling, {orig_len} -> {len(train_df)}, num_classes {train_df[class_label_name].nunique()}"
    )
    return train_df


def downsample_with_class_label(
    train_df: pd.DataFrame,
    class_label_name: str = "landmark_id",
    num_samples_per_class: Optional[int] = 10,
    remove_tail_class: bool = True,
    tail_class_threshold: int = 50,
    reset_label_id: bool = True,
    check_label_id: bool = True,
) -> pd.DataFrame:

    if remove_tail_class:
        orig_len = len(train_df)
        orig_num_classes = train_df[class_label_name].nunique()
        per_label_id_count = train_df[class_label_name].value_counts()
        target_label_ids = per_label_id_count[per_label_id_count > tail_class_threshold].index
        train_df = train_df[train_df[class_label_name].isin(target_label_ids)]
        logger.info(
            f"remove_tail_class with threshold:{tail_class_threshold}, num_samples: {orig_len} -> {len(train_df)}"
        )
        logger.info(
            f"remove_tail_class with threshold:{tail_class_threshold}, num_classes: {orig_num_classes} -> {train_df[class_label_name].nunique()}"
        )
        if reset_label_id:
            train_df = make_label_id_successive(
                train_df=train_df, class_label_name=class_label_name
            )

    if num_samples_per_class is not None:
        train_df = sampling_per_class(
            train_df=train_df,
            class_label_name=class_label_name,
            num_samples_per_class=num_samples_per_class,
        )

    if check_label_id:
        # check is label consistent
        label_set = set(train_df[class_label_name].to_numpy())
        assert max(label_set) + 1 == len(label_set)
    return train_df
