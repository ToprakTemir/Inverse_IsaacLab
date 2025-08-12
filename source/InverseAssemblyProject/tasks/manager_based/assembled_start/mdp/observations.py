# observation_helpers.py
from isaaclab.managers import SceneEntityCfg
import torch
from isaaclab.utils.math import quat_apply_inverse


def object_pos(env, asset_cfg: SceneEntityCfg):
    """Return world position of the object's root body."""
    asset = env.scene[asset_cfg.name]
    # root_state_w shape: (num_envs, 13) -> first 3 entries are position
    return asset.data.root_pos_w  # (num_envs, 3)


def target_pos(env, asset_cfg: SceneEntityCfg):
    """Return world position of the target (fixed) object's root body."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w  # (num_envs, 3)

def ee_ft_sensor(env, asset_cfg: SceneEntityCfg):

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

    # sensor = env.scene[asset_cfg.name]          # grab the ContactSensor instance
    # force_w = sensor.data.net_forces_w  # (N, B, 3)
    #
    # # Converting to local orientation from world orientation
    # robot = env.scene["robot"].data
    # ee_link_idx = robot.body_names.index("ee_base_link")  # find the index of the end effector link
    # pos_w, quat_w = robot.body_link_pose_w[:, ee_link_idx]  # get the quaternion of the end effector link
    # force = quat_rotate_inverse(quat_w, force_w)
    # force = force.sum(dim=1)  # sum forces to get net force on the end effector link | (N, 3)
    #
    # # compute torque as cross product of position vector and force
    # contacts = sensor.contact_physx_view.get_contact_data()
    # r = contacts.pos_w - sensor.data.pos_w.unsqueeze(1)  # (N, B, 3) - (N, 1, 3) -> (N, B, 3)
    # torque = torch.cross(r, force.unsqueeze(1), dim=-1)  # (N, B, 3) x (N, 1, 3) -> (N, B, 3)
    #
    # return torch.cat([force, torque], dim=-1) # (N, 6)

