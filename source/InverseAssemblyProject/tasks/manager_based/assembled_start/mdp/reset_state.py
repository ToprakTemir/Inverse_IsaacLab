import torch
from isaaclab.managers import SceneEntityCfg


def reset_robot_joints(env, env_ids, asset_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    # modify only the selected envs
    noise = 0.05 * (2 * torch.rand_like(robot.data.joint_pos[env_ids]) - 1.0)
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    joint_pos[env_ids] = noise
    joint_vel[env_ids] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


def reset_disk_pose(env, env_ids, asset_cfg: SceneEntityCfg):
    disk = env.scene[asset_cfg.name]
    # low, high = 0.3, 0.6
    # xy = low + (high - low) * torch.rand(len(env_ids), 2, device=disk.data.root_pos_w.device)
    xy = torch.tensor([0.1, 0.3], device=disk.data.root_pos_w.device).repeat(len(env_ids), 1)

    # create a normal around xy
    offset = torch.distributions.Normal(xy, 0.1).sample()

    xy = xy + offset

    z = torch.full((len(env_ids), 1), 0.01, device=disk.data.root_pos_w.device)
    local_pos = torch.cat([xy, z], dim=-1)
    new_quat = torch.zeros(len(env_ids), 4, device=disk.data.root_quat_w.device)
    new_quat[:, 0] = 1.0

    # clone full tensors, update subset, then write back
    pos = disk.data.root_pos_w.clone()
    quat = disk.data.root_quat_w.clone()
    lin = disk.data.root_lin_vel_w.clone()
    ang = disk.data.root_ang_vel_w.clone()
    pos[env_ids] = local_pos + env.scene.env_origins[env_ids]
    quat[env_ids] = new_quat
    lin[env_ids] = 0.0
    ang[env_ids] = 0.0

    root_state = torch.cat([pos, quat, lin, ang], dim=-1)  # (num_envs, 14)
    disk.write_root_state_to_sim(root_state[env_ids], env_ids)

def reset_base_pose(env, env_ids, asset_cfg: SceneEntityCfg):
    # put the disk in a set position in front of the robot, a bit in the air and rotated so that it is turned towards the robot

    base = env.scene[asset_cfg.name]

    xy = (0.25, 0.55)  # fixed position in front of the robot
    z = 0.1  # a bit in the air

    # rotate 90 degrees around x-axis
    new_quat = torch.tensor([0.7071, 0.7071, 0, 0], device=base.data.root_quat_w.device).repeat(len(env_ids), 1)
    local_pos = torch.tensor([xy[0], xy[1], z], device=base.data.root_pos_w.device).repeat(len(env_ids), 1)
    # clone full tensors, update subset, then write back
    pos = base.data.root_pos_w.clone()
    quat = base.data.root_quat_w.clone()
    lin = base.data.root_lin_vel_w.clone()
    ang = base.data.root_ang_vel_w.clone()
    pos[env_ids] = local_pos + env.scene.env_origins[env_ids]
    quat[env_ids] = new_quat
    lin[env_ids] = 0.0
    ang[env_ids] = 0.0
    root_state = torch.cat([pos, quat, lin, ang], dim=-1)  # (num_envs, 14)
    base.write_root_state_to_sim(root_state[env_ids], env_ids)

def reset_to_assembled_pose(env, env_ids, base_asset_cfg: SceneEntityCfg, disk_asset_cfg: SceneEntityCfg):
    """Reset such that the disk is already inserted to the base."""

    reset_base_pose(env, env_ids, base_asset_cfg)

    disk = env.scene[disk_asset_cfg.name]

    # base's transform
    local_pos = torch.tensor([0.25, 0.55, 0.1], device=disk.data.root_pos_w.device).repeat(len(env_ids), 1)
    # offset of the tip of the base
    local_pos += torch.tensor([0.0594, -0.09807075, -0.0214], device=disk.data.root_pos_w.device).repeat(len(env_ids), 1)
    # local_pos = torch.tensor([0, 0, 0], device=disk .data.root_pos_w.device).repeat(len(env_ids), 1)

    new_quat = torch.tensor([0.7071, 0, 0, 0.7071], device=disk.data.root_quat_w.device).repeat(len(env_ids), 1)
    # clone full tensors, update subset, then write back
    pos = disk.data.root_pos_w.clone()
    quat = disk.data.root_quat_w.clone()
    lin = disk.data.root_lin_vel_w.clone()
    ang = disk.data.root_ang_vel_w.clone()
    pos[env_ids] = local_pos + env.scene.env_origins[env_ids]
    quat[env_ids] = new_quat
    lin[env_ids] = 0.0
    ang[env_ids] = 0.0
    root_state = torch.cat([pos, quat, lin, ang], dim=-1)  # (num_envs, 14)

    disk.write_root_state_to_sim(root_state[env_ids], env_ids)



