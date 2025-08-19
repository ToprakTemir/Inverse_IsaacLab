import os
import math

from pygame.gfxdraw import aatrigon
from sympy.physics.vector import kinematic_equations

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import activate_contact_sensors
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
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg

from . import mdp


# --------------------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------------------

ARM_KP = 15000.0 # Proportional gain for the arm joints
ARM_KD = 1200.0 # Derivative gain for the arm joints

@configclass
class AssembledStartSceneCfg(InteractiveSceneCfg):
    """Scene for Task 1: separate assets."""

    ground = AssetBaseCfg(
        prim_path="/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(8192.0, 8192.0)),
    )

    # UR5e robot with gripper
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="../assets/robot.usd",
            activate_contact_sensors=True,  # Enable contact sensors for the robot
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -math.pi / 2,
                "elbow_joint": math.pi / 2,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
                "Slider_1": 0.0,
                "Slider_2": 0.0,
            },
        ),
        actuators={
            "shoulder_pan_actuator": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint"],
                effort_limit_sim=10000.0,
                velocity_limit_sim=100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "shoulder_lift_actuator": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_lift_joint"],
                effort_limit_sim
                =10000.0,
                velocity_limit_sim
                =100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "elbow_actuator": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                effort_limit_sim
                =10000.0,
                velocity_limit_sim
                =100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "wrist_1_actuator": ImplicitActuatorCfg(
                joint_names_expr=["wrist_1_joint"],
                effort_limit_sim
                =1000.0,
                velocity_limit_sim
                =100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "wrist_2_actuator": ImplicitActuatorCfg(
                joint_names_expr=["wrist_2_joint"],
                effort_limit_sim
                =1000.0,
                velocity_limit_sim
                =100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "wrist_3_actuator": ImplicitActuatorCfg(
                joint_names_expr=["wrist_3_joint"],
                effort_limit_sim
                =1000.0,
                velocity_limit_sim
                =100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "Slider_1_actuator": ImplicitActuatorCfg(
                joint_names_expr=["Slider_1"],
                effort_limit_sim
                =100000000.0,
                velocity_limit_sim
                =100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
        },
    )

    ee_ft_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/robot/ur5e/RobotIq_Hand_E_base/ee_base_link",
        update_period=0.0,  # 0.0 means update every step
        history_length=0,  # No history, only current contact forces
        debug_vis=False, # Set to True to visualize 3d force arrows
    )


    # Fixed base / rod
    fixed_obj: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/base",
        spawn=sim_utils.UsdFileCfg(usd_path="../assets/task1_fixed.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.25, 0.55, 0.15),  # Initial position in front of the robot
            rot=(0.7071, 0.7071, 0, 0),  # Rotate 90 degrees around x-axis
            lin_vel=(0.0, 0.0, 0.0),  # Initial linear velocity
            ang_vel=(0.0, 0.0, 0.0),  # Initial angular velocity
        ),
    )

    moved_obj: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/disk",
        spawn=sim_utils.UsdFileCfg(
            usd_path="../assets/task1_moved.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=2.0, max_contact_impulse=1.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3094, 0.4599, 0.1286),  # Initial position inserted into the base
            rot=(0.7071, 0, 0, 0.7071),  # Rotate 90 degrees around x-axis
            lin_vel=(0.0, 0.0, 0.0),  # Initial linear velocity
            ang_vel=(0.0, 0.0, 0.0),  # Initial angular velocity
        ),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0)),
    )


# --------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """Action space: joint angle targets for the robot."""

    # joint_angles = mdp.JointPositionToLimitsActionCfg(
    #     asset_name="robot",
    #     joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "Slider_1"],
    #     rescale_to_limits=True
    # )


    ee_IK_delta = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        body_name = "ee_base_link",
        scale=0.15,
        controller=mdp.DifferentialIKControllerCfg(
            command_type = "pose",  # position: 3d, pose: 6d (position + orientation)
            use_relative_mode=True,  # Relative changes in position/pose
            ik_method="dls",
        )
    )
    # ee_joint_angles = mdp.JointPositionToLimitsActionCfg(
    #     asset_name="robot",
    #     joint_names=["Slider_1"],
    #     rescale_to_limits=True,
    #     scale=1.0,  # Scale the action to the joint limits
    # )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["Slider_1"],
        open_command_expr={"Slider_1": 0.0},  # Open gripper
        close_command_expr={"Slider_1": 0.025},  # Close gripper
    )

    # gripper_action = mdp.JointPositionToLimitsActionCfg(
    #     asset_name="robot",
    #     joint_names=["Slider_1"],
    #     rescale_to_limits=True,
    #     scale=1.0,  # Scale the action to the joint limits
    # )


# --------------------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    """Observation groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation vector (concatenated)."""

        joint_positions = ObsTerm(func=mdp.joint_pos_rel)
        object_pos = ObsTerm(func=mdp.object_pos, params={"asset_cfg": SceneEntityCfg("moved_obj")})
        target_pos = ObsTerm(func=mdp.target_pos, params={"asset_cfg": SceneEntityCfg("fixed_obj")})
        ft_sensor = ObsTerm(func=mdp.ee_ft_sensor, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


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

    # reset_disk_pose = EventTerm(
    #     func=mdp.reset_disk_pose,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("moved_obj"),
    #     },
    # )
    # reset_base_pose = EventTerm(
    #     func=mdp.reset_base_pose,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("fixed_obj"),
    #     },
    # )

    reset_disk_to_assembled_pose = EventTerm(
        func=mdp.reset_disk_to_assembled_pose,
        mode="reset",
        params={
            "base_asset_cfg": SceneEntityCfg("fixed_obj"),
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
    success = DoneTerm(func=mdp.disassembly_success, time_out=False)


# --------------------------------------------------------------------------------------
# Top‑level env config
# --------------------------------------------------------------------------------------
@configclass
class AssembledStartEnvCfg(ManagerBasedRLEnvCfg):

    # externally overridable knobs (accepted by constructor)I'm trying to collect demonstrations of robot holding something and putting it somewhere else in Isaaclab using record_demos.py script preinstalled with isaaclab
    num_envs: int = 1            # will be pushed to scene.num_envs
    env_spacing: float = 2.5     # will be pushed to scene.env_spacing

    rw_progress: float = 2.0
    rw_success: float = 1.0
    rw_proximity: float = 1.0
    rw_control_penalty: float = 0.01

    scene: AssembledStartSceneCfg = AssembledStartSceneCfg(num_envs=1, env_spacing=2.5)
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
    sim.dt = 1.0 / 360
    sim.physx.max_position_iteration_count = 200
    sim.physx.max_velocity_iteration_count = 20
    episode_length_s = 24.0
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

        self.decimation = 3 # the period of the policy being queried, in simulation steps
        self.sim.render_interval = self.decimation

        self.viewer.eye = (0.5, 0.25, 0.2)
        self.viewer.lookat = (0.25, 0.5, 0.25)
