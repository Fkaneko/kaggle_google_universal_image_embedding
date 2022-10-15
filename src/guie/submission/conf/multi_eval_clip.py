from pathlib import Path

from src.guie.model.multi_domain.multi_domain_learning import md_embed_mode

from .default import create_ckpt_path_from_dir, default_submission


def unified_dsample5() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_dsample5"
    ckpt_dir = Path("../working/guie/train_log/07-10-2022_11-30-54")
    ckpt_steps = 7600
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_dsample10_middle_ckpt() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_dsample10_middle_ckpt"
    ckpt_dir = Path("../working/guie/train_log/07-10-2022_18-20-48")
    ckpt_steps = 18620
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_dsample5_one_layer() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_dsample5_one_layer"
    ckpt_dir = Path("../working/guie/train_log/08-10-2022_00-45-31")
    ckpt_steps = 5168
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_dsample5_concat_2() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_dsample5_concat_2"
    ckpt_dir = Path("../working/guie/train_log/08-10-2022_08-30-28")
    ckpt_steps = 7600
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_dsample5_concat_2_sma() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_dsample5_concat_2_sma"
    ckpt_dir = Path("../working/guie/train_log/08-10-2022_08-30-28")
    ckpt_steps = 7600
    ckpt_path = create_ckpt_path_from_dir(
        ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps, is_weight_averaging=True
    )
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_dsample5_concat_2_sma_center_crop() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_dsample5_concat_2_sma_center_crop"
    ckpt_dir = Path("../working/guie/train_log/08-10-2022_08-30-28")
    ckpt_steps = 7600
    ckpt_path = create_ckpt_path_from_dir(
        ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps, is_weight_averaging=True
    )
    # crop_pct = 0.8
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_dsample20_class_less_further_sma() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_dsample20_class_less_further_sma"
    ckpt_dir = Path("../working/guie/train_log/09-10-2022_08-45-39/")
    ckpt_steps = 6425
    ckpt_path = create_ckpt_path_from_dir(
        ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps, is_weight_averaging=True
    )
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_dsample20_class_less_sma() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_dsample20_class_less_sma"
    ckpt_dir = Path("../working/guie/train_log/09-10-2022_06-00-51")
    ckpt_steps = 9264
    ckpt_path = create_ckpt_path_from_dir(
        ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps, is_weight_averaging=True
    )
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_dsample20_class_less_further_head_sma() -> dict:
    # valiant-tree-400
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_dsample20_class_less_further_head_sma"
    ckpt_dir = Path("../working/guie/train_log/09-10-2022_14-21-58")
    ckpt_steps = 4369
    ckpt_path = create_ckpt_path_from_dir(
        ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps, is_weight_averaging=True
    )
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_dsample40_class_less_mild_sma() -> dict:
    # sweet-wildflower-405
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_dsample40_class_less_mild_sma"

    ckpt_dir = Path("../working/guie/train_log/09-10-2022_14-44-25")
    ckpt_steps = 4140
    ckpt_path = create_ckpt_path_from_dir(
        ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps, is_weight_averaging=True
    )
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_dsample20_class_less_further_concat_sma() -> dict:
    # atomic-dream-407
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_dsample20_class_less_further_concat_sma"

    ckpt_dir = Path("../working/guie/train_log/09-10-2022_18-17-09")
    ckpt_steps = 6425
    ckpt_path = create_ckpt_path_from_dir(
        ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps, is_weight_averaging=True
    )
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_final_vit_l() -> dict:
    # curious-tree-420
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_final_vit_l"

    ckpt_dir = Path("../working/guie/train_log/10-10-2022_00-43-49")
    ckpt_steps = 6264
    ckpt_path = create_ckpt_path_from_dir(
        ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps, is_weight_averaging=True
    )
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_final_vit_h() -> dict:
    # legendary-salad-422
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_final_vit_h_fixed"

    ckpt_dir = Path("../working/guie/train_log/10-10-2022_04-32-33")
    ckpt_steps = 5220
    ckpt_path = create_ckpt_path_from_dir(
        ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps, is_weight_averaging=True
    )
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_final_second_vit_h() -> dict:
    # feasible-monkey-427
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_final_second_vit_h_fixed"

    ckpt_dir = Path("../working/guie/train_log/10-10-2022_12-46-59")
    ckpt_steps = 4347  # epoch = 21
    ckpt_path = create_ckpt_path_from_dir(
        ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps, is_weight_averaging=True
    )
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_final_third_vit_h() -> dict:
    # wise-plant-430
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_final_third_vit_h_fixed"
    ckpt_dir = Path("../working/guie/train_log/10-10-2022_15-59-25")
    ckpt_steps = 3243  # epoch = 23
    ckpt_path = create_ckpt_path_from_dir(
        ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps, is_weight_averaging=True
    )
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf
