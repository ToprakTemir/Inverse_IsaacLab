import torch

def disassembly_success(env):
    """Count it success when the disk hits the ground"""

    moved_obj = env.scene["moved_obj"]
    moved_obj_pos = moved_obj.data.root_pos_w
    z = moved_obj_pos[:, 2]
    return z < 0.02


# def assembly_success(env, pos_tolerance: float = 0.05):
#     return not disassembly_success(env, pos_tolerance) # dist < pos_tolerance

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
