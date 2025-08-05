# observation_helpers.py
from isaaclab.managers import SceneEntityCfg
import torch

def object_pos(env, asset_cfg: SceneEntityCfg):
    """Return world position of the object's root body."""
    asset = env.scene[asset_cfg.name]
    # root_state_w shape: (num_envs, 13) -> first 3 entries are position
    return asset.data.root_pos_w  # (num_envs, 3)


def target_pos(env, asset_cfg: SceneEntityCfg):
    """Return world position of the target (fixed) object's root body."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w  # (num_envs, 3)
