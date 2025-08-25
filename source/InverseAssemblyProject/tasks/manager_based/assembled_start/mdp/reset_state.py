import random

import torch
from isaaclab.managers import SceneEntityCfg

def reset_robot_joints(env, env_ids, asset_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    noise = 0.05 * (2 * torch.rand_like(robot.data.joint_pos[env_ids]) - 1.0)
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    default_joint_pos = robot.data.default_joint_pos.clone()
    joint_pos[env_ids] = default_joint_pos[env_ids] + noise
    joint_vel[env_ids] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

def _write_cords_to_asset(env, env_ids, asset_cfg: SceneEntityCfg, local_pos, quat=None):
    asset = env.scene[asset_cfg.name]

    if quat is None:
        quat = torch.tensor([1, 0, 0, 0], device=asset.data.root_quat_w.device).repeat(len(env_ids), 1)

    # clone full tensors, update subset, then write back
    pos = asset.data.root_pos_w.clone()
    new_quat = asset.data.root_quat_w.clone()
    lin = asset.data.root_lin_vel_w.clone()
    ang = asset.data.root_ang_vel_w.clone()
    pos[env_ids] = local_pos + env.scene.env_origins[env_ids]
    new_quat[env_ids] = quat
    lin[env_ids] = 0.0
    ang[env_ids] = 0.0

    root_state = torch.cat([pos, new_quat, lin, ang], dim=-1)  # (num_envs, 14)
    asset.write_root_state_to_sim(root_state[env_ids], env_ids)

# def reset_disk_pose(env, env_ids, asset_cfg: SceneEntityCfg):
#     disk = env.scene[asset_cfg.name]
#     # low, high = 0.3, 0.6
#     # xy = low + (high - low) * torch.rand(len(env_ids), 2, device=disk.data.root_pos_w.device)
#     xy = torch.tensor([0.1, 0.3], device=disk.data.root_pos_w.device).repeat(len(env_ids), 1)
#
#     # create a normal around xy
#     offset = torch.distributions.Normal(xy, 0.1).sample()
#
#     xy = xy + offset
#
#     z = torch.full((len(env_ids), 1), 0.01, device=disk.data.root_pos_w.device)
#     local_pos = torch.cat([xy, z], dim=-1)
#     _write_cords_to_asset(env, env_ids, asset_cfg, local_pos)
#
# def reset_base_pose(env, env_ids, asset_cfg: SceneEntityCfg):
#     # put the disk in a set position in front of the robot, a bit in the air and rotated so that it is turned towards the robot
#
#     base = env.scene[asset_cfg.name]
#
#     xy = (0.25, 0.55)  # fixed position in front of the robot
#     z = 0.15  # a bit in the air
#
#     # rotate 90 degrees around x-axis
#     new_quat = torch.tensor([0.7071, 0.7071, 0, 0], device=base.data.root_quat_w.device).repeat(len(env_ids), 1)
#     local_pos = torch.tensor([xy[0], xy[1], z], device=base.data.root_pos_w.device).repeat(len(env_ids), 1)
#     _write_cords_to_asset(env, env_ids, asset_cfg, local_pos)

def _get_random_base_xyz(num_envs, device):
    x = torch.distributions.Uniform(-0.4, 0.4).sample((num_envs, 1)).to(device)
    y = torch.distributions.Uniform(0.6, 0.72).sample((num_envs, 1)).to(device)
    z = torch.distributions.Uniform(0.1, 0.45).sample((num_envs, 1)).to(device)
    return torch.cat([x, y, z], dim=-1)

def _get_random_disk_xyz(num_envs, device):
    x = torch.distributions.Uniform(-0.6, 0.6).sample((num_envs, 1)).to(device)
    y = torch.distributions.Uniform(0.3, 0.6).sample((num_envs, 1)).to(device)
    z = torch.zeros((num_envs, 1), device=device) + 0.001
    return torch.cat([x, y, z], dim=-1)

def reset_to_assembled_pose(env, env_ids, base_asset_cfg: SceneEntityCfg, disk_asset_cfg: SceneEntityCfg):
    """Reset such that the disk is already inserted to the base."""

    base = env.scene[base_asset_cfg.name]
    device = base.data.root_pos_w.device

    # put the disk in a set position in front of the robot
    base_pos = _get_random_base_xyz(len(env_ids), device)
    base_quat = torch.tensor([0.7071, 0.7071, 0, 0], device=device).repeat(len(env_ids), 1) # 90 degrees around x-axis
    _write_cords_to_asset(env, env_ids, base_asset_cfg, base_pos, base_quat)

    env.sim.step(render=False)  # step sim to avoid collisions

    disk_offset = torch.tensor([0.0594, -0.09007075, -0.0214], device=device).repeat(len(env_ids), 1)
    disk_pos = base_pos + disk_offset
    disk_quat = torch.tensor([0.7071, 0, 0, 0.7071], device=device).repeat(len(env_ids), 1)
    _write_cords_to_asset(env, env_ids, disk_asset_cfg, disk_pos, disk_quat)


def reset_to_disassembled_pose(env, env_ids, base_asset_cfg: SceneEntityCfg, disk_asset_cfg: SceneEntityCfg):
    """Reset such that the disk is on the ground."""

    device = env.scene[base_asset_cfg.name].data.root_pos_w.device

    base_pos = _get_random_base_xyz(len(env_ids), device)
    base_quat = torch.tensor([0.7071, 0.7071, 0, 0], device=device).repeat(len(env_ids), 1) # 90 degrees around x-axis
    _write_cords_to_asset(env, env_ids, base_asset_cfg, base_pos, base_quat)

    disk_quat = torch.tensor([0.5, 0.5, 0.5, 0.5], device=device).repeat(len(env_ids), 1) # +90 degrees in y and z coordinates
    disk_pos = _get_random_disk_xyz(len(env_ids), device)
    _write_cords_to_asset(env, env_ids, disk_asset_cfg, disk_pos, disk_quat)


