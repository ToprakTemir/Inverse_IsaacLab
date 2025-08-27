# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

import gymnasium as gym

gym.register(
    id="Disassembly-UR3e-IK-v0",      # anything unique
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            "InverseAssemblyProject.tasks.manager_based.assembly_task.assembled_start_cfg:AssembledStartEnvCfg",
    },
)

