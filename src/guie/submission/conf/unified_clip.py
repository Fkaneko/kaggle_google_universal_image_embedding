from pathlib import Path

from src.guie.model.multi_domain.multi_domain_learning import md_embed_mode

from .default import create_ckpt_path_from_dir, default_submission


def unified_clip_1layer_head_3domains_strong_aug() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_clip_1layer_head_3domains_strong_aug"
    ckpt_dir = Path("../working/guie/train_log/01-10-2022_09-12-56")
    ckpt_steps = 13825
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_clip_2layer_head_3domains_strong_aug() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_clip_2layer_head_3domains_strong_aug"
    ckpt_dir = Path("../working/guie/train_log/01-10-2022_10-53-02")
    ckpt_steps = 13825
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf


def unified_clip_4layer_head_3domains_strong_aug() -> dict:
    conf = default_submission()
    save_name = f"{md_embed_mode.UNIFIED}_clip_4layer_head_3domains_strong_aug"
    ckpt_dir = Path("../working/guie/train_log/01-10-2022_12-33-11")
    ckpt_steps = 13825
    ckpt_path = create_ckpt_path_from_dir(ckpt_dir=ckpt_dir, ckpt_steps=ckpt_steps)
    conf.update(
        {
            "save_name": save_name,
            "ckpt_path": ckpt_path,
        }
    )
    return conf
