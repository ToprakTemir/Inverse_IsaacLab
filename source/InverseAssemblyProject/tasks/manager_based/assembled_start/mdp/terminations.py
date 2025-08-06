import torch

def disassembly_success(env, pos_tolerance: float = 0.06):
    """Success when distance between disk and target base is <= tolerance."""
    disk_pos = env.scene["moved_obj"].data.root_pos_w
    # target = env.scene["fixed_obj"]
    num_environments = disk_pos.shape[0]

    target_base = env.scene["fixed_obj"].data.root_pos_w
    target_offset = torch.tensor([0.0594, -0.09807075, -0.0214], device=disk_pos.device).repeat(num_environments, 1)
    target_pos = target_base + target_offset

    diff = disk_pos - target_pos
    dist = torch.linalg.norm(diff, dim=-1)

    return dist >= pos_tolerance

def assembly_success(env, pos_tolerance: float = 0.05):
    return not disassembly_success(env, pos_tolerance) # dist < pos_tolerance

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
