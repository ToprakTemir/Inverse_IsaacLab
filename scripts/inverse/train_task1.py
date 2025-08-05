# scripts/train_task1.py
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from InverseAssemblyProject.tasks.manager_based.assembled_start.assembled_start_cfg import AssembledStartEnvCfg
import torch
import time

env = ManagerBasedRLEnv(AssembledStartEnvCfg())

obs_dim = env.observation_space
act_dim = env.action_space
print(f"Observation space: {obs_dim}, Action space: {act_dim}")

# dummy random policy
def random_policy(obs: torch.Tensor) -> torch.Tensor:
    return torch.rand_like(env.action_manager.action_tensor) * 2 - 1  # random in [-1, 1]

# Training loop
for epoch in range(1000):
    obs = env.reset()
    total_reward = torch.zeros(env.num_envs, device=obs.device)

    for step in range(env.max_episode_length):
        action = random_policy(obs)
        obs, reward, terminated, time_out, info = env.step(action)
        total_reward += reward

        # Reset done environments (Isaac Lab handles this automatically inside .step)
        # Optional: do something custom here if you want manual resets

    print(f"Epoch {epoch}, mean reward: {total_reward.mean().item():.2f}")

app.close()
