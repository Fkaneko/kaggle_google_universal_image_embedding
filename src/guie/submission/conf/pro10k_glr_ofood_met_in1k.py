from pathlib import Path

from src.guie.dataset.multi_domain.constants import md_const

from .default import default_submission



def fix_lr() -> dict:
    conf = default_submission()
    save_name = "pro10k_glr_o-food_met_in1k_fix_lr"
    ckpt_path = Path(
        "../working/guie/train_log/27-09-2022_10-48-27/checkpoint-20200/pytorch_model.bin"
    )
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def remove_food() -> dict:
    conf = default_submission()
    save_name = "pro10k_glr_o-food_met_in1k_fix_lr_remove_food"
    ckpt_path = Path(
        "../working/guie/train_log/27-09-2022_10-48-27/checkpoint-20200/pytorch_model.bin"
    )
    avg_pool_domain_names = [
        str(md_const.OMNI_BENCH.name),
    ]
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
            "avg_pool_domain_names": avg_pool_domain_names,
        }
    )
    return conf


def domain_cls_only() -> dict:
    conf = default_submission()
    save_name = "pro10k_glr_o-food_met_in1k_fix_lr_domain_cls_only"
    ckpt_path = Path(
        "../working/guie/train_log/27-09-2022_10-48-27/checkpoint-20200/pytorch_model.bin"
    )
    avg_pool_domain_names = [
        str(md_const.PRODUCT_10K.name),
        str(md_const.GLR.name),
        str(md_const.MET.name),
        str(md_const.OMNI_BENCH.name),
    ]
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
            "avg_pool_domain_names": avg_pool_domain_names,
        }
    )
    return conf


def domain_cls_only_with_met_to_glr() -> dict:
    conf = default_submission()
    save_name = "pro10k_glr_o-food_met_in1k_fix_lr_domain_cls_only_with_met_to_glr"
    ckpt_path = Path(
        "../working/guie/train_log/27-09-2022_10-48-27/checkpoint-20200/pytorch_model.bin"
    )
    avg_pool_domain_names = [
        str(md_const.PRODUCT_10K.name),
        str(md_const.GLR.name),
        str(md_const.OMNI_BENCH.name),
        str(md_const.MET.name),
    ]
    domain_cls_mappings = {md_const.MET.name: md_const.GLR.name}
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
            "avg_pool_domain_names": avg_pool_domain_names,
            "domain_cls_mappings": domain_cls_mappings,
        }
    )
    return conf
