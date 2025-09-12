import math

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    EventTermCfg as EventTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg

from . import mdp
from .assembled_start_cfg import AssemblySceneCfg, ActionsCfg, ObservationsCfg


# --------------------------------------------------------------------------------------
# Events (resets / randomization)
# --------------------------------------------------------------------------------------
@configclass
class EventCfg:
    """Reset events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_robot_joints,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    reset_disk_to_disassembled_pose = EventTerm(
        func=mdp.reset_to_disassembled_pose,
        mode="reset",
        params={
            "base_asset_cfg": SceneEntityCfg("fixed_obj"),
            "disk_asset_cfg": SceneEntityCfg("moved_obj"),
        },
    )

    set_robot_to_reach_pose = EventTerm(
        func=mdp.set_robot_to_insertion_pose,
        mode="reset",
        params={
            "base_asset_cfg": SceneEntityCfg("fixed_obj"),
            "disk_asset_cfg": SceneEntityCfg("moved_obj"),
        },
    )

    set_robot_holding_disk = EventTerm(
        func=mdp.set_robot_holding_disk,
        mode="reset",
        params={
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "disk_asset_cfg": SceneEntityCfg("moved_obj"),
            },
    )


# --------------------------------------------------------------------------------------
# Rewards
# --------------------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """Reward shaping terms."""

    # disassembly_progress_reward = RewTerm(func=mdp.disassembly_dist_reward, weight=2.0)
    # success_reward = RewTerm(func=mdp.disassembly_success_reward, weight=1.0)
    # proximity_reward = RewTerm(func=mdp.object_ee_proximity_reward, weight=1.0, params={"asset_cfg": SceneEntityCfg("moved_obj")})
    # control_penalty = RewTerm(func=mdp.control_penalty, weight=0.01)


# --------------------------------------------------------------------------------------
# Terminations
# --------------------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.assembly_success, time_out=False)


# --------------------------------------------------------------------------------------
# Top‑level env config
# --------------------------------------------------------------------------------------
@configclass
class DisassembledStartEnvCfg(ManagerBasedRLEnvCfg):

    num_envs: int = 1024
    env_spacing: float = 2.5

    rw_progress: float = 2.0
    rw_success: float = 1.0
    rw_proximity: float = 1.0
    rw_control_penalty: float = 0.01

    scene: AssemblySceneCfg = AssemblySceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # rewards.disassembly_progress_reward.weight = float(rw_progress)
    # rewards.success_reward.weight = float(rw_success)
    # rewards.proximity_reward.weight = float(rw_proximity)
    # rewards.control_penalty.weight = float(rw_control_penalty)

    sim = sim_utils.SimulationCfg()
    sim.dt = 1.0 / 720
    sim.physx.max_position_iteration_count = 200
    sim.physx.max_velocity_iteration_count = 20
    episode_length_s = 10.0
    sim.physx.gpu_max_rigid_patch_count = 2621440
    sim.physx.gpu_collision_stack_size = 2 ** 29
    sim.device = "cuda:0"
    sim.use_gpu_pipeline = True
    sim.physx.use_gpu = True
    sim.enable_scene_query_support = False


    def __post_init__(self):
        # propagate top-level overrides down into nested configs
        self.scene.num_envs = int(self.num_envs)
        self.scene.env_spacing = float(self.env_spacing)

        self.decimation = 6 # the period of the policy being queried, in simulation steps
        self.sim.render_interval = self.decimation

        self.viewer.eye = (0.5, 0.25, 0.2)
        self.viewer.lookat = (0.25, 0.5, 0.25)
