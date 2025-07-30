import torch

def time_out(env):
    """Placeholder for time-out termination (handled by DoneTerm with time_out=True)."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def check_success(env, pos_tolerance: float):
    """Success when distance between disk and target base is <= tolerance."""
    disk = env.scene["moved_obj"]
    target = env.scene["fixed_obj"]

    diff = disk.data.root_pos_w - target.data.root_pos_w
    dist = torch.linalg.norm(diff, dim=-1)
    return dist <= pos_tolerance


def out_of_bounds(env, limit: float = 1.5, min_z: float = -0.2):
    """Terminate if robot base or disk leaves a bounding box.

    limit: |x| or |y| beyond this -> done
    min_z: z coordinate below this -> done
    """
    disk = env.scene["moved_obj"]
    robot = env.scene["robot"]

    disk_pos = disk.data.root_pos_w
    robot_pos = robot.data.root_pos_w

    disk_oob = (
        (disk_pos[:, 0].abs() > limit)
        | (disk_pos[:, 1].abs() > limit)
        | (disk_pos[:, 2] < min_z)
    )
    robot_oob = (
        (robot_pos[:, 0].abs() > limit)
        | (robot_pos[:, 1].abs() > limit)
        | (robot_pos[:, 2] < min_z)
    )
    return disk_oob | robot_oob
