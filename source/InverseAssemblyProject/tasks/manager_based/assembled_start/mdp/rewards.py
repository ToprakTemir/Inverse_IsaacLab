
# disassembly reward
from isaaclab.app import AppLauncher
from isaaclab.managers import SceneEntityCfg
import torch

def disassembly_dist_reward(env):
    """Reward based on distance between disk and target base."""

    disk_pos = env.scene["moved_obj"].data.root_pos_w

    target_base = env.scene["fixed_obj"].data.root_pos_w
    target_offset = torch.tensor([0.0594, -0.09807075, -0.0214], device=disk_pos.device).repeat(num_environments, 1)
    target_pos = target_base + target_offset

    diff = disk_pos - target_pos
    dist = torch.linalg.norm(diff, dim=-1)
    return dist

def disassembly_success_reward(env, pos_tolerance: float = 0.06):
    """Success reward when distance between disk and target base is >= tolerance."""
    disk_pos = env.scene["moved_obj"].data.root_pos_w

    target_base = env.scene["fixed_obj"].data.root_pos_w
    target_offset = torch.tensor([0.0594, -0.09807075, -0.0214], device=disk_pos.device).repeat(num_environments, 1)
    target_pos = target_base + target_offset

    diff = disk_pos - target_pos
    dist = torch.linalg.norm(diff, dim=-1)

    success = dist >= pos_tolerance

    # Return a reward of 1.0 for success, 0.0 otherwise
    return success.float()

def object_ee_proximity_reward(env, asset_cfg: SceneEntityCfg):
    """Reward based on proximity of the robot's end effector to the object."""
    robot = env.scene["robot"]
    object_pos = env.scene[asset_cfg.name].data.root_pos_w

    finger_1 = robot.data.body_pos_w[:, -1, :]
    finger_2 = robot.data.body_pos_w[:, -2, :]
    # let the "end effector position" be the average of the two fingers
    ee_pos = (finger_1 + finger_2) / 2.0

    diff = ee_pos - object_pos
    dist = torch.linalg.norm(diff, dim=-1)

    # Reward is higher when closer than the tolerance
    return 1 - dist  # Normalize distance to [0, 1] range

def control_penalty(env):
    """Penalty for control actions to encourage smooth movements."""
    # Assuming env.action_manager.action is the action tensor
    action = env.action_manager.action
    return -torch.linalg.norm(action, dim=-1)

