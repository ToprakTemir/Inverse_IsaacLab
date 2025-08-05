
# disassembly reward
from isaaclab.app import AppLauncher
from isaaclab.managers import SceneEntityCfg
import torch

def disassembly_dist_reward(env):
    """Reward based on distance between disk and target base."""

    disk = env.scene["moved_obj"]
    target = env.scene["fixed_obj"]

    diff = disk.data.root_pos_w - target.data.root_pos_w
    dist = torch.linalg.norm(diff, dim=-1)
    return dist

def disassembly_success_reward(env, pos_tolerance: float = 0.05):
    """Success reward when distance between disk and target base is >= tolerance."""
    disk = env.scene["moved_obj"]
    target = env.scene["fixed_obj"]

    diff = disk.data.root_pos_w - target.data.root_pos_w
    dist = torch.linalg.norm(diff, dim=-1)
    success = dist >= pos_tolerance

    # Return a reward of 1.0 for success, 0.0 otherwise
    return success.float()

def control_penalty(env):
    """Penalty for control actions to encourage smooth movements."""
    # Assuming env.action_manager.action is the action tensor
    action = env.action_manager.action
    return -torch.linalg.norm(action, dim=-1)

