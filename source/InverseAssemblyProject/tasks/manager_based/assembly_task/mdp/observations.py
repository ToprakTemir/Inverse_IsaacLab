# observation_helpers.py
from isaaclab.managers import SceneEntityCfg
import torch
from isaaclab.utils.math import quat_apply_inverse


def object_pos(env, asset_cfg: SceneEntityCfg):
    """Return world position of the object's root body."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w  # (num_envs, 3)

def object_quat(env, asset_cfg: SceneEntityCfg):
    """Return world orientation of the object's root body."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_quat_w  # (num_envs, 4)

def target_pos(env, asset_cfg: SceneEntityCfg):
    """Return world position of the target (fixed) object's root body."""
    base_pos = env.scene[asset_cfg.name].data.root_pos_w
    offset = torch.tensor([0.0594, -0.09007075, -0.0214], device=base_pos.device)
    return base_pos + offset

def disk_target_distance(env, disk_asset_cfg: SceneEntityCfg, base_asset_cfg: SceneEntityCfg):
    disk_pos = env.scene[disk_asset_cfg.name].data.root_pos_w
    goal_pos = target_pos(env, base_asset_cfg)
    return torch.norm(disk_pos - goal_pos, dim=-1, keepdim=True)  # (num_envs, 1)

def ee_ft_sensor(env):

    robot = env.scene["robot"]

    # grab wrench on the EE link (world frame)
    ee_idx = robot.data.body_names.index("ee_base_link")
    wrench_w = robot.root_physx_view.get_link_incoming_joint_force()[:, ee_idx]  # (N_env, 6)
    force_w, torque_w = wrench_w[:, :3], wrench_w[:, 3:]

    # rotate into the EE link frame
    quat_w = robot.data.body_link_quat_w[:, ee_idx]  # (N_env, 4, wxyz)
    force = quat_apply_inverse(quat_w, force_w)
    torque = quat_apply_inverse(quat_w, torque_w)

    return torch.cat([force, torque], dim=-1)  # (N_env, 6)


def ee_tip_pose(env, ee_tip_config: SceneEntityCfg = SceneEntityCfg(name="ee_center")):
    data = env.scene[ee_tip_config.name].data

    # target_pos_source has shape (num_envs, num_targets, n), so we select all environments, the first (and only) target, and all coordinates
    pos_relative_to_source = data.target_pos_source[..., 0, :]  # (num_envs, 3)
    quat_relative_to_source = data.target_quat_source[..., 0, :]  # (num_envs, 4)
    return torch.cat([pos_relative_to_source, quat_relative_to_source], dim=-1)  # (num_envs, 7)

