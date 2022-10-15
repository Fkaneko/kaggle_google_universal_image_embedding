from pathlib import Path

from src.guie.model.multi_domain.multi_domain_learning import md_embed_mode

from .default import create_ckpt_path_from_dir, default_submission


def unified_2domains_2projections() -> dict:
    # Note: this was actually 3 domains
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_pro10k_glr_2_projections"
    ckpt_dir = Path("../working/guie/train_log/28-09-2022_15-06-22")
    ckpt_steps = 13825
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_3domains_2projections() -> dict:
    # Note: this was actually 2 domains
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_pro10k_glr_met_2_projections"
    ckpt_dir = Path("../working/guie/train_log/28-09-2022_17-54-14")
    ckpt_steps = 10750
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_3domains_heavy_10k_2projections() -> dict:
    # Note: this was actually 2 domains
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_3domains_heavy_10k_2projections"
    ckpt_dir = Path("../working/guie/train_log/29-09-2022_12-20-16")
    ckpt_steps = 17925
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_convN_t_3domains_heavy_10k() -> dict:
    # Note: this was actually 2 domains
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_convN_3domains_heavy_10k"
    ckpt_dir = Path("../working/guie/train_log/29-09-2022_21-59-24")
    ckpt_steps = 27075
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_convN_t_clip_teach_3domains_heavy_10k_strong_aug() -> dict:
    # Note: this was actually 2 domains
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_convN_t_clip_teach_3domains_heavy_10k_strong_aug"
    ckpt_dir = Path("../working/guie/train_log/30-09-2022_11-16-44")
    ckpt_steps = 28225
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf
