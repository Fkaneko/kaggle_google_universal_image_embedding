import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from omegaconf import DictConfig

from src.common.dataset.downsampling_data import (
    downsample_with_class_label,
    make_label_id_successive,
)
from src.common.dataset.train_val_split import make_split
from src.guie.dataset.glr.constants import glr_const
from src.guie.dataset.glr.preprocess import get_file_path
from src.guie.dataset.glr.preprocess import load_train_csv as load_glr_train_csv
from src.guie.dataset.hotel_id.constants import hotel_id_const
from src.guie.dataset.hotel_id.preprocess import load_train_csv as load_hotel_id_train_csv
from src.guie.dataset.ifood.constants import ifood_const
from src.guie.dataset.ifood.preprocess import load_train_csv as load_ifood_train_csv
from src.guie.dataset.in_shop.constants import in_shop_const
from src.guie.dataset.in_shop.preprocess import load_train_txt as load_in_shop_train_csv
from src.guie.dataset.met.constants import met_const
from src.guie.dataset.met.preprocess import load_train_json as load_met_train_json
from src.guie.dataset.multi_domain.constants import md_const
from src.guie.dataset.omni_bench.constants import omni_const
from src.guie.dataset.omni_bench.preprocess import load_train_csv as load_omni_bench_train_csv
from src.guie.dataset.omni_bench.preprocess import omni_col
from src.guie.dataset.products_10k.constants import products_10k_const
from src.guie.dataset.products_10k.preprocess import add_domain_name
from src.guie.dataset.products_10k.preprocess import load_train_csv as load_products_10k_train_csv
from src.guie.dataset.sop.constants import sop_const
from src.guie.dataset.sop.preprocess import load_train_txt as load_sop_train_csv

logger = logging.getLogger(__name__)

OMNI_BENCH_REALMS = ["food"]


def cache_train_csv(df: pd.DataFrame, cache_path: Path) -> None:
    logger.info(f"cache data frame on {str(cache_path)}")
    df.to_csv(cache_path, index=False)


def load_cached_train_csv(cache_path: Path) -> pd.DataFrame:
    logger.info(f"load cached data frame from {str(cache_path)}")
    df = pd.read_csv(cache_path)
    return df


def load_train_csv(dataset_conf: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # sop dataset
    logger.info("start sop dataset loading...")
    sop_df = load_sop_train_csv(
        data_dir=Path(dataset_conf.sop_dir), use_furniture_only=dataset_conf.sop_use_furniture_only
    )
    sop_df = downsample_with_class_label(
        train_df=sop_df,
        class_label_name=sop_const.LABEL_ID,
        num_samples_per_class=dataset_conf.sop_num_samples_per_class,
        remove_tail_class=dataset_conf.sop_remove_tail_class,
        tail_class_threshold=dataset_conf.sop_tail_class_threshold,
    )

    # in_shop dataset
    logger.info("start in_shop dataset loading...")
    in_shop_df = load_in_shop_train_csv(data_dir=Path(dataset_conf.in_shop_dir))
    in_shop_df = downsample_with_class_label(
        train_df=in_shop_df,
        class_label_name=in_shop_const.LABEL_ID,
        num_samples_per_class=dataset_conf.in_shop_num_samples_per_class,
        remove_tail_class=dataset_conf.in_shop_remove_tail_class,
        tail_class_threshold=dataset_conf.in_shop_tail_class_threshold,
    )
    # products_10k
    logger.info("start products_10k dataset loading...")
    products_10k_df = load_products_10k_train_csv(data_dir=Path(dataset_conf.products_10k_dir))
    products_10k_df = add_domain_name(train_df=products_10k_df)
    products_10k_df = downsample_with_class_label(
        train_df=products_10k_df,
        class_label_name=products_10k_const.LABEL_ID,
        num_samples_per_class=dataset_conf.products_10k_num_samples_per_class,
        remove_tail_class=dataset_conf.products_10k_remove_tail_class,
        tail_class_threshold=dataset_conf.products_10k_tail_class_threshold,
    )
    logger.info(
        f"Products10K, Domain samples:\n{products_10k_df[products_10k_const.DOMAIN_NAME].value_counts()}"
    )
    logger.info(
        f"Products10K, Domain classes:\n{products_10k_df.groupby(products_10k_const.DOMAIN_NAME)[products_10k_const.LABEL_ID].nunique()}"
    )
    # met dataset
    logger.info("start met dataset loading...")
    met_df = load_met_train_json(data_dir=Path(dataset_conf.met_dir))
    met_df = downsample_with_class_label(
        train_df=met_df,
        class_label_name=met_const.LABEL_ID,
        num_samples_per_class=dataset_conf.met_num_samples_per_class,
        remove_tail_class=dataset_conf.met_remove_tail_class,
        tail_class_threshold=dataset_conf.met_tail_class_threshold,
    )
    # ifood dataset
    logger.info("start ifood dataset loading...")
    ifood_df = load_ifood_train_csv(data_dir=Path(dataset_conf.ifood_dir))
    ifood_df = downsample_with_class_label(
        train_df=ifood_df,
        class_label_name=ifood_const.LABEL_ID,
        num_samples_per_class=dataset_conf.ifood_num_samples_per_class,
        remove_tail_class=dataset_conf.ifood_remove_tail_class,
        tail_class_threshold=dataset_conf.ifood_tail_class_threshold,
    )
    # hotel_id dataset
    logger.info("start hotel_id dataset loading...")
    hotel_id_df = load_hotel_id_train_csv(data_dir=Path(dataset_conf.hotel_id_dir))
    hotel_id_df = downsample_with_class_label(
        train_df=hotel_id_df,
        class_label_name=hotel_id_const.LABEL_ID,
        num_samples_per_class=dataset_conf.hotel_id_num_samples_per_class,
        remove_tail_class=dataset_conf.hotel_id_remove_tail_class,
        tail_class_threshold=dataset_conf.hotel_id_tail_class_threshold,
    )

    # glr
    logger.info("start glr dataset loading...")
    glr_df = load_glr_train_csv(data_dir=Path(dataset_conf.glr_cleaned_dir))
    glr_df = downsample_with_class_label(
        train_df=glr_df,
        class_label_name=glr_const.LABEL_ID,
        num_samples_per_class=dataset_conf.glr_num_samples_per_class,
        remove_tail_class=dataset_conf.glr_remove_tail_class,
        tail_class_threshold=dataset_conf.glr_tail_class_threshold,
    )
    glr_df[glr_const.FILEPATH] = glr_df[glr_const.ID].apply(
        lambda x: str(
            get_file_path(
                data_dir=Path(dataset_conf.glr_cleaned_dir), data_split=glr_const.TRAIN, id=x
            )
        )
    )
    # omin
    logger.info("start omni benchmark dataset loading...")
    omni_bench_df = load_omni_bench_train_csv(
        data_dir=Path(dataset_conf.omni_bench_dir), load_target_realms=OMNI_BENCH_REALMS
    )
    # manually make successive dut to pre defined train/val split
    omni_bench_df = make_label_id_successive(
        train_df=omni_bench_df, class_label_name=omni_col.LABEL_ID
    )

    # split with pre defined key
    omni_bench_df_val = omni_bench_df.loc[omni_bench_df[omni_col.DATA_SPLIT] == omni_const.VAL]
    omni_bench_df_train = omni_bench_df.loc[omni_bench_df[omni_col.DATA_SPLIT] == omni_const.TRAIN]

    omni_bench_df_train = downsample_with_class_label(
        train_df=omni_bench_df_train,
        class_label_name=omni_col.LABEL_ID,
        num_samples_per_class=dataset_conf.omni_bench_num_samples_per_class,
        remove_tail_class=dataset_conf.omni_bench_remove_tail_class,
        tail_class_threshold=dataset_conf.omni_bench_tail_class_threshold,
        reset_label_id=False,
    )

    # normalize label_id name
    products_10k_df = products_10k_df.rename(
        columns={products_10k_const.LABEL_ID: md_const.LABEL_ID}
    )
    met_df = met_df.rename(columns={met_const.LABEL_ID: md_const.LABEL_ID})
    ifood_df = ifood_df.rename(columns={ifood_const.LABEL_ID: md_const.LABEL_ID})
    hotel_id_df = hotel_id_df.rename(columns={hotel_id_const.LABEL_ID: md_const.LABEL_ID})
    in_shop_df = in_shop_df.rename(columns={in_shop_const.LABEL_ID: md_const.LABEL_ID})
    sop_df = sop_df.rename(columns={sop_const.LABEL_ID: md_const.LABEL_ID})
    glr_df = glr_df.rename(columns={glr_const.LABEL_ID: md_const.LABEL_ID})

    # set domain id
    products_10k_df[md_const.DOMAIN_ID] = md_const.PRODUCT_10K.id
    met_df[md_const.DOMAIN_ID] = md_const.MET.id
    ifood_df[md_const.DOMAIN_ID] = md_const.IFOOD.id
    hotel_id_df[md_const.DOMAIN_ID] = md_const.HOTEL_ID.id
    in_shop_df[md_const.DOMAIN_ID] = md_const.IN_SHOP.id
    sop_df[md_const.DOMAIN_ID] = md_const.SOP.id
    glr_df[md_const.DOMAIN_ID] = md_const.GLR.id
    omni_bench_df_train[md_const.DOMAIN_ID] = md_const.OMNI_BENCH.id
    omni_bench_df_val[md_const.DOMAIN_ID] = md_const.OMNI_BENCH.id

    # concat over different data
    train_df = pd.concat(
        [
            products_10k_df.loc[:, md_const.TARGET_FIELDS],
            met_df.loc[:, md_const.TARGET_FIELDS],
            ifood_df.loc[:, md_const.TARGET_FIELDS],
            hotel_id_df.loc[:, md_const.TARGET_FIELDS],
            in_shop_df.loc[:, md_const.TARGET_FIELDS],
            sop_df.loc[:, md_const.TARGET_FIELDS],
            glr_df.loc[:, md_const.TARGET_FIELDS],
        ],
        axis=0,
    )
    train_df = train_df.reset_index(drop=True)

    logger.info(
        f"loading multi_domain dataframe, total sample: {len(train_df)} \n each domain \n{train_df[md_const.DOMAIN_ID].value_counts()})"
    )

    logger.info("start train/val split")
    _train_val_split_key = "split_key"
    train_df[_train_val_split_key] = (
        train_df[md_const.DOMAIN_ID].astype(str) + "_" + train_df[md_const.LABEL_ID].astype(str)
    )
    train_df = make_split(
        df=train_df, n_splits=dataset_conf.n_splits, target_key=_train_val_split_key
    )
    del train_df[_train_val_split_key]

    omni_bench_df_train = omni_bench_df_train.loc[:, md_const.TARGET_FIELDS]
    omni_bench_df_val = omni_bench_df_val.loc[:, md_const.TARGET_FIELDS]
    # pseudo field def
    omni_bench_df_val["fold"] = None
    omni_bench_df_train["fold"] = None

    return train_df, omni_bench_df_train, omni_bench_df_val


def get_domain_to_num_labels(train_df: pd.DataFrame) -> Dict[str, int]:
    domain_to_num_labels = {}
    for domain in md_const.all_domains:
        if domain.name == md_const.OTHER.name:
            domain_to_num_labels[md_const.OTHER.name] = 1
        else:
            domain_to_num_labels[domain.name] = len(
                set(train_df.loc[train_df[md_const.DOMAIN_ID] == domain.id, md_const.LABEL_ID])
            )
    return domain_to_num_labels


def keep_only_target_domain(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_domain_names: List[str],
    domain_to_num_labels: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:

    logger.info("start target domain selection")
    train_domain_ids = []
    non_target_domain_names = list(domain_to_num_labels.keys())
    for domain_name in target_domain_names:
        train_domain_ids.append(md_const.domain_name_to_id[domain_name])
        non_target_domain_names.remove(domain_name)

    for non_target_domain_name in non_target_domain_names:
        domain_to_num_labels[non_target_domain_name] = 1

    train_df = train_df[train_df[md_const.DOMAIN_ID].isin(train_domain_ids)].reset_index(drop=True)
    val_df = val_df[val_df[md_const.DOMAIN_ID].isin(train_domain_ids)].reset_index(drop=True)
    logger.info(f"target domain selection end: \n{train_df[md_const.DOMAIN_ID].value_counts()}")
    logger.info(f"target domain selection end: \n{val_df[md_const.DOMAIN_ID].value_counts()}")
    return train_df, val_df, domain_to_num_labels


def make_label_id_unified(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    domain_to_num_labels: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    logger.info("start generating unified label ids...")

    _unified_label_id_seed = "label_id_seed"
    train_df[_unified_label_id_seed] = (
        train_df[md_const.DOMAIN_ID].astype(str) + "_" + train_df[md_const.LABEL_ID].astype(str)
    )
    val_df[_unified_label_id_seed] = (
        val_df[md_const.DOMAIN_ID].astype(str) + "_" + val_df[md_const.LABEL_ID].astype(str)
    )
    unique_labels = list(set(train_df[_unified_label_id_seed]))
    label_seed_to_new_label_id = {
        str(label_id_seed): new_label_id for new_label_id, label_id_seed in enumerate(unique_labels)
    }
    train_df[md_const.LABEL_ID] = train_df[_unified_label_id_seed].apply(
        lambda x: label_seed_to_new_label_id[x]
    )
    val_df[md_const.LABEL_ID] = val_df[_unified_label_id_seed].apply(
        lambda x: label_seed_to_new_label_id[x]
    )
    del train_df[_unified_label_id_seed]
    del val_df[_unified_label_id_seed]

    # override domain id
    _UNIFIED_DOMAIN = md_const.OTHER
    # update domain_to_num_labels
    for domain in md_const.all_domains:
        if domain == _UNIFIED_DOMAIN:
            domain_to_num_labels[_UNIFIED_DOMAIN.name] = len(label_seed_to_new_label_id)
        else:
            domain_to_num_labels[domain.name] = 1
    logger.info(f"update domain after unified: {domain_to_num_labels}")
    return (
        train_df,
        val_df,
        domain_to_num_labels,
    )


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from src.common.utils import set_logger

    set_logger(logger=logging.getLogger())
    dataset_conf = OmegaConf.load("./src//guie//conf/dataset/baseline.yaml")
    train_df = load_train_csv(dataset_conf=dataset_conf)
    print(train_df.head)
    print(train_df.shape)
    # print(glr_df.head())
    # print(glr_df[glr_const.LABEL_ID].value_counts())
    # print((glr_df[glr_const.LABEL_ID].value_counts() > 50).index)
