
import argparse
import time

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
app = app_launcher.app

from InverseAssemblyProject.tasks.manager_based.assembled_start.assembled_start_cfg import AssembledStartEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

import torch
from disassembly.NN import MLPPolicy

# Create env

env = ManagerBasedRLEnv(AssembledStartEnvCfg(num_envs=3))

# policy_path = "checkpoints/run-ppo-20250807-112615-best.pth"
policy_path = None

obs_dim = env.observation_space["policy"].shape[1]
act_dim = env.action_space.shape[1]
print(f"obs_dim: {obs_dim}, act_dim: {act_dim}")

# Load policy
if policy_path:
    policy = MLPPolicy(obs_dim, act_dim, hidden=256)
    policy_weights = torch.load(policy_path, map_location=env.device)
    policy.load_state_dict(policy_weights)
    policy.eval()
    policy.to(env.device)
    print(f"Loaded policy from {policy_path}")
    def act(obs):
        return policy.act(obs, deterministic=True)[0]  # deterministic action
else:
    print("No policy provided — running with random actions")
    def act(obs):
        # return torch.rand_like(env.action_manager.action) * 2 - 1 # random actions in [-1, 1]
        joints = torch.zeros_like(env.action_manager.action[:, :-1])  # zero actions
        # gripper = torch.randint_like(env.action_manager.action[:, -1], low=0, high=2).unsqueeze(-1)
        gripper = torch.rand_like(env.action_manager.action[:, -1:]) * 2 - 1  # random gripper action in [-1, 1]
        random_gripper_action = torch.cat([joints, gripper], dim=-1)  # random gripper action
        return random_gripper_action
        # return torch.zeros_like(env.action_manager.action)



# Run demo
obs, info = env.reset()

for _ in range(10):
    env.sim.step(render=True)  # step the simulation to initialize the scene
    time.sleep(0.1)  # wait a bit to see the initial state


# while True:
    # sleep
    # time.sleep(0.01)
    # env.render()

    # robot = env.scene["robot"]
    #
    # # find indices for your sliders
    # jidx = [i for i, n in enumerate(robot.data.joint_names) if n in ("Slider_1")]
    # print("[DEBUG] indices:", jidx)
    #
    # # (a) check limits
    # print("[DEBUG] limits low/high:", robot.data.joint_pos_limits[0, jidx])
    #
    # # (b) try opening once
    # q_target = robot.data.default_joint_pos.clone()
    # print("[DEBUG] default joint positions:", q_target)
    # for i in jidx:
    #     q_target[:, i] = 0.0245  # inside limits
    # print("[DEBUG] target joint positions:", q_target)
    # robot.set_joint_position_target(q_target)  # API is available on Articulation
    # env.sim.step(render=True)

while True:
    obs = obs["policy"]  # Get the policy observation
    action = act(obs)
    obs, rew, terminated, time_out, info = env.step(action)

app.close()
