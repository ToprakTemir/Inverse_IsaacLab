import math

from sympy.physics.vector import kinematic_equations

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


# --------------------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------------------

ARM_KP = 80.0 # Proportional gain for the arm joints
ARM_KD = 10.0 # Derivative gain for the arm joints

@configclass
class Task1SceneCfg(InteractiveSceneCfg):
    """Scene for Task 1: separate assets."""

    ground = AssetBaseCfg(
        prim_path="/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(16384.0, 16384.0)),

    )

    # UR3e robot with gripper
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="assets/robot.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
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
                # "finger_joint": 0.0,  # Gripper joint
            },
        ),
        actuators={
            "shoulder_pan_actuator": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint"],
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "shoulder_lift_actuator": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_lift_joint"],
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "elbow_actuator": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "wrist_1_actuator": ImplicitActuatorCfg(
                joint_names_expr=["wrist_1_joint"],
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "wrist_2_actuator": ImplicitActuatorCfg(
                joint_names_expr=["wrist_2_joint"],
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "wrist_3_actuator": ImplicitActuatorCfg(
                joint_names_expr=["wrist_3_joint"],
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "Slider_1": ImplicitActuatorCfg(
                joint_names_expr=["Slider_1"],
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
            "Slider_2": ImplicitActuatorCfg(
                joint_names_expr=["Slider_2"],
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=ARM_KP,
                damping=ARM_KD,
            ),
        },
    )

    # Movable disk object (dynamic rigid body)
    moved_obj: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/disk",
        spawn=sim_utils.UsdFileCfg(usd_path="assets/task1_moved_obj.usd"),
    )

    # Fixed base / rod
    fixed_obj: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/base",
        spawn=sim_utils.UsdFileCfg(usd_path="assets/task1_fixed_obj.usd", rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),

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

    joint_angles = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "Slider_1", "Slider_2"],
        rescale_to_limits=True
    )

    # ee_pose_delta = mdp.DifferentialInverseKinematicsActionCfg(
    #     asset_name="robot",
    #     joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
    #     body_name = "ee_base_link",
    #     controller=mdp.DifferentialIKControllerCfg(
    #         command_type = "position",  # position: 3d, pose: 6d (position + orientation)
    #         use_relative_mode=True,  # Relative changes in position/pose
    #         ik_method="pinv",  # Pseudo-inverse method
    #     )
    # )
    # ee_joint_angles = mdp.JointPositionToLimitsActionCfg(
    #     asset_name="robot",
    #     joint_names=["Slider_1", "Slider_2"],
    #     rescale_to_limits=True,
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

    reset_disk_pose = EventTerm(
        func=mdp.reset_disk_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("moved_obj"),
        },
    )

    reset_base_pose = EventTerm(
        func=mdp.reset_base_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("fixed_obj"),
        },
    )


# --------------------------------------------------------------------------------------
# Rewards
# --------------------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """Reward shaping terms."""

    dummy_zero_reward = RewTerm(
        func=lambda: 0.0,
        weight=0.0,
    )


# --------------------------------------------------------------------------------------
# Terminations
# --------------------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # success = DoneTerm(func=mdp.check_success, params={"pos_tolerance": 0.005})
    # failure_out_of_bounds = DoneTerm(func=mdp.out_of_bounds, params={"limit": 1.5, "min_z": -0.2})


# --------------------------------------------------------------------------------------
# Top‑level env config
# --------------------------------------------------------------------------------------
@configclass
class Task1EnvCfg(ManagerBasedRLEnvCfg):

    scene: Task1SceneCfg = Task1SceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1.0 / 120.0
        self.episode_length_s = 8.0
        self.sim.render_interval = self.decimation

        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.5)
