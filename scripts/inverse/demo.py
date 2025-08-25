
import argparse
import time

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
app = app_launcher.app

from InverseAssemblyProject.tasks.manager_based.assembled_start.assembled_start_cfg import AssembledStartEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

import torch
from disassembly.NN import MLPPolicy

# Create environment
env = ManagerBasedRLEnv(AssembledStartEnvCfg(num_envs=50))

obs_dim = env.observation_space["policy"].shape[1]
act_dim = env.action_space.shape[1]
print(f"obs_dim: {obs_dim}, act_dim: {act_dim}")

# policy_path = "checkpoints/run-ppo-20250807-112615-best.pth"
policy_path = None

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
        return torch.rand_like(env.action_manager.action) * 2 - 1 # random actions in [-1, 1]
        # joints = torch.zeros_like(env.action_manager.action[:, :-1])  # zero actions
        # gripper = torch.rand_like(env.action_manager.action[:, -1:]) * 2 - 1  # random gripper action in [-1, 1]
        # return torch.cat([joints, gripper], dim=-1)  # random gripper action


# Run demo
obs, info = env.reset()
while True:
    obs = obs["policy"]  # Get the policy observation
    action = act(obs)
    obs, rew, terminated, time_out, info = env.step(action)
    env.render()

app.close()
