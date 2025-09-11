import random

import torch
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils.math import (
    quat_mul,
    quat_conjugate,
    quat_box_minus,
)


from . import observations

DISK_OFFSET_FROM_BASE = torch.tensor([0.0594, -0.09007075, -0.0214])
DISK_SPAWN_QUAT = torch.tensor([0.7071, 0, 0, 0.7071]) # 90 degrees around x-axis

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

def reset_to_assembled_pose(env, env_ids, base_asset_cfg: SceneEntityCfg, disk_asset_cfg: SceneEntityCfg, set_robot_to_reach_pose: bool = False):
    """Reset such that the disk is already inserted to the base."""

    base = env.scene[base_asset_cfg.name]
    device = base.data.root_pos_w.device

    # put the disk in a set position in front of the robot
    base_pos = _get_random_base_xyz(len(env_ids), device)
    base_quat = torch.tensor([0.7071, 0.7071, 0, 0], device=device).repeat(len(env_ids), 1) # 90 degrees around x-axis
    _write_cords_to_asset(env, env_ids, base_asset_cfg, base_pos, base_quat)

    env.sim.step(render=False)  # step sim to avoid collisions

    disk_pos = base_pos + DISK_OFFSET_FROM_BASE.to(device)
    disk_quat = DISK_SPAWN_QUAT.to(device)
    _write_cords_to_asset(env, env_ids, disk_asset_cfg, disk_pos, disk_quat)

    if set_robot_to_reach_pose:
        # set the robot joints to reach for the disk

        epsilon = 0.01
        gripper_tip_offset_from_disk = torch.tensor([0, -0.05, 0], device=device).repeat(len(env_ids), 1) # TODO: tune
        gripper_target_pos = disk_pos + gripper_tip_offset_from_disk
        gripper_target_quat = torch.tensor([0, 0, 0, 0], device=device).repeat(len(env_ids), 1) # same as the spawn orientation

        normal_dist = torch.distributions.Normal(loc=gripper_target_pos, scale=epsilon) # add some noise
        gripper_target_pos = normal_dist.sample()

        set_ee_pose_ik(env, env_ids, gripper_target_pos, gripper_target_quat)


def reset_to_disassembled_pose(env, env_ids, base_asset_cfg: SceneEntityCfg, disk_asset_cfg: SceneEntityCfg):
    """Reset such that the disk is on the ground."""

    device = env.scene[base_asset_cfg.name].data.root_pos_w.device

    base_pos = _get_random_base_xyz(len(env_ids), device)
    base_quat = torch.tensor([0.7071, 0.7071, 0, 0], device=device).repeat(len(env_ids), 1) # 90 degrees around x-axis
    _write_cords_to_asset(env, env_ids, base_asset_cfg, base_pos, base_quat)

    disk_quat = torch.tensor([0.5, 0.5, 0.5, -0.5], device=device).repeat(len(env_ids), 1) # +90 degrees in x and z coordinates
    disk_pos = _get_random_disk_xyz(len(env_ids), device)
    _write_cords_to_asset(env, env_ids, disk_asset_cfg, disk_pos, disk_quat)

def set_ee_pose_ik(env, env_ids, fingertip_target_pos, fingertip_target_quat):
    """Set robot joint position using DLS IK."""
    terms_backup = env.recorder_manager._terms
    names_backup = env.recorder_manager._term_names
    env.recorder_manager._terms = {}
    env.recorder_manager._term_names = []

    ik_time = 0.0
    err = torch.ones((len(env_ids), 6), device=fingertip_target_pos.device) * 10.0  # initialize with large error
    while torch.norm(err) > 1e-3:
        # get current ee pose
        ee_pose = observations.ee_tip_pose(env)
        ee_pos = ee_pose[env_ids, :3]  # (num_envs, 3)
        ee_quat = ee_pose[env_ids, 3:]  # (num_envs, 4)

        # compute error
        pos_err = fingertip_target_pos - ee_pos
        quat_err = quat_box_minus(fingertip_target_quat, ee_quat) # quat difference operator function
        err = torch.cat([pos_err, quat_err], dim=-1)  # (num_envs, 6)

        # set the differential IK controller to move towards the target
        action = torch.zeros_like(env.action_manager.action)
        action[env_ids, :6] = err[env_ids]
        env.step(action)
        ik_time += env.sim.cfg.dt
        if ik_time > 5.0:  # timeout after 10 seconds
            print("IK timeout")
            break

    env.recorder_manager._terms = terms_backup
    env.recorder_manager._term_names = names_backup