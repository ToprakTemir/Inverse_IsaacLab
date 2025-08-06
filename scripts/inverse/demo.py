
import argparse
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
app = app_launcher.app

from InverseAssemblyProject.tasks.manager_based.assembled_start.assembled_start_cfg import AssembledStartEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

import torch
from disassembly.NN import MLPPolicy

# Create env
env = ManagerBasedRLEnv(AssembledStartEnvCfg())

policy_path = "checkpoints/ppo_disassembly_best.pth"
# policy_path = None

obs_dim = env.observation_space["policy"].shape[1]
act_dim = env.action_space.shape[1]
print(f"obs_dim: {obs_dim}, act_dim: {act_dim}")

# Load policy
if policy_path:
    policy = MLPPolicy(obs_dim, act_dim)
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



# Run demo
env.reset()
env.step(torch.rand_like(env.action_manager.action) * 2 - 1)
env.step(torch.rand_like(env.action_manager.action) * 2 - 1)
env.step(torch.rand_like(env.action_manager.action) * 2 - 1)
obs, info = env.reset()

while True:
    obs = obs["policy"]  # Get the policy observation
    action = act(obs)
    obs, rew, terminated, time_out, info = env.step(action)
    env.render()

app.close()
