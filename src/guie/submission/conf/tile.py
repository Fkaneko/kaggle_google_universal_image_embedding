from pathlib import Path

from src.guie.model.multi_domain.multi_domain_learning import md_embed_mode

from .default import create_ckpt_path_from_dir, default_submission


def tile_2domains_2projections() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.TILE}_pro10k_glr_met_2_projections"
    ckpt_dir = Path("../working/guie/train_log/28-09-2022_20-29-04")
    ckpt_steps = 10750
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def tile_3domains_2projections() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.TILE}_3domains_2projections"
    ckpt_dir = Path("../working/guie/train_log/29-09-2022_01-52-59")
    ckpt_steps = 13825
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def tile_3domains_2projections_soft_embed() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.TILE}_3domains_2projections_soft_embed"
    ckpt_dir = Path("../working/guie/train_log/29-09-2022_01-52-59")
    ckpt_steps = 13825
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)

    # soft parameter history
    # 1575
    # actual_range = 0.56
    # max_range = 1.0
    # 1555
    # actual_range = 0.4
    # max_range = 1.0
    # 0.1455
    # actual_range = 0.2
    # max_range = 1.0
    # 1575
    # actual_range = 0.8
    # max_range = 1.4
    # 1555
    # actual_range = 1.0
    # max_range = 1.5

    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
            "use_soft_domain_embed": True,
            "soft_embed_max_scale": 1.0,
            "soft_embed_range": 0.4,
        }
    )
    return conf


def tile_3domains_2projections_soft_embed_weak() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.TILE}_3domains_2projections_soft_embed_weak"
    ckpt_dir = Path("../working/guie/train_log/29-09-2022_01-52-59")
    ckpt_steps = 13825
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)

    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
            "use_soft_domain_embed": True,
            "soft_embed_max_scale": 1.0,
            "soft_embed_range": 0.2,
        }
    )
    return conf
