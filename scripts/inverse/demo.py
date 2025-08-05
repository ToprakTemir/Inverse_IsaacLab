
import argparse
from isaaclab.app import AppLauncher

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model (.pt)")
args = parser.parse_args()

app_launcher = AppLauncher(headless=False)
app = app_launcher.app

from InverseAssemblyProject.tasks.manager_based.task_1.task1_env_cfg import Task1EnvCfg
from isaaclab.envs import ManagerBasedRLEnv

import torch
import time

# Create env
env = ManagerBasedRLEnv(Task1EnvCfg())

obs_dim = env.observation_space
act_dim = env.action_space
print(f"Observation space: {obs_dim}, Action space: {act_dim}")

# Load policy
if args.checkpoint:
    policy = torch.load(args.checkpoint)
    policy.eval()
    def act(obs): return policy(obs)
else:
    print("No policy provided — running with random actions")
    def act(obs):
        return torch.rand_like(env.action_manager.action) * 2 - 1

    # def act(obs):
    #     obs = obs["policy"]
    #     joint_angles = obs[:, :8]  # get joint angles
    #     action = joint_angles
    #     return action

# Run demo
env.reset()
env.step(torch.rand_like(env.action_manager.action) * 2 - 1)
env.step(torch.rand_like(env.action_manager.action) * 2 - 1)
env.step(torch.rand_like(env.action_manager.action) * 2 - 1)
obs, info = env.reset()

# while True:
#     time.sleep(0.001)

while True:
    action = act(obs)
    obs, rew, terminated, time_out, info = env.step(action)

app.close()
