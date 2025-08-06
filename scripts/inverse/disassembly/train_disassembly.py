#!/usr/bin/env python3
# train_disassembly.py
"""
Train a PPO policy to learn the disk-removal task described in AssembledStartEnvCfg.
Designed for Isaac-Lab 2025.05+  (manager-based RL API).

Run:
    python train_disassembly.py                               # default hyper-params
    wandb sweep sweep.yaml && wandb agent <SWEEP_ID>          # hyper-param tuning
"""

# ------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------

import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model (.pt)")
args = parser.parse_args()
app_launcher = AppLauncher(headless=True)
app = app_launcher.app


from types import SimpleNamespace
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
import wandb
from tqdm import tqdm

from isaaclab.envs import ManagerBasedRLEnv
from InverseAssemblyProject.tasks.manager_based.assembled_start.assembled_start_cfg import AssembledStartEnvCfg


from NN import MLPPolicy, RolloutBuffer


def make_env():
    cfg = AssembledStartEnvCfg()
    env = ManagerBasedRLEnv(cfg)
    # env = gym_wrapper.GymEnvWrapper(env, flatten_obs=True)   # obs -> 1-D float32
    return env


# ------------------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------------------
def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Environment
    env = make_env()
    obs_dim = env.observation_space['policy'].shape[1]  # policy observation space
    act_dim = env.action_space.shape[1] # action space
    print(f"obs_dim: {obs_dim}, act_dim: {act_dim}")

    n_envs  = env.num_envs

    # 2. Agent
    policy = MLPPolicy(obs_dim, act_dim, hidden=config.hidden_size).to(device)
    optim_ = optim.Adam(policy.parameters(), lr=config.lr, eps=1e-5)

    # 3. Rollout buffer
    buf = RolloutBuffer(
        n_envs=n_envs,
        steps=config.rollout_steps,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
    )

    # 4. W&B
    wandb.init(project="disassembly-policy",
               name=config.run_name,
               config=config.__dict__,
               save_code=True)
    wandb.watch(policy, log="gradients", log_freq=1000)

    # 5. Training
    obs, _ = env.reset()
    obs = obs["policy"]  # extract policy observation
    obs = torch.as_tensor(obs, device=device)
    global_step = 0
    best_episode_reward = -np.inf

    for update in tqdm(range(config.updates), desc="PPO updates"):

        # Collect experience
        buf.ptr = 0 # reset buffer pointer to overwrite old data
        mean_episode_reward = 0
        for _ in range(config.rollout_steps):
            with torch.no_grad():
                mu, std, val = policy(obs)
                dist = Normal(mu, std)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
            next_obs, rew, terminated, time_out, info = env.step(act)
            done = terminated | time_out
            next_obs = next_obs["policy"]  # extract policy observation
            buf.add(
                obs, act, torch.as_tensor(rew, device=device),
                torch.as_tensor(done, device=device),
                logp, val.squeeze(-1),
            )
            obs = torch.as_tensor(next_obs, device=device)
            global_step += n_envs

            # ----------------------- Logging -------------------------------
            mean_reward = rew.mean().item()
            mean_episode_reward += mean_reward
            wandb.log({
                "mean_reward": mean_reward,
                "global_step": global_step,
            })

        with torch.no_grad():
            _, _, last_val = policy(obs)
            last_val = last_val.squeeze(-1)
        buf.vals[-1] = last_val            # bootstrap final value
        buf.compute_returns_and_adv(config.gamma, config.gae_lambda)

        # Update policy
        for _ in range(config.backprops_per_rollout):
            for b_obs, b_act, b_logp, b_adv, b_ret in buf.get(config.batch_size):
                mu, std, val = policy(b_obs)
                dist = Normal(mu, std)
                new_logp = dist.log_prob(b_act).sum(-1)
                entropy  = dist.entropy().sum(-1).mean()

                ratio = (new_logp - b_logp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - config.clip_eps,
                                              1.0 + config.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = (b_ret - val.squeeze(-1)).pow(2).mean()
                loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

                optim_.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optim_.step()


        # Check-pointing ------------------------------------------------------------
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(policy.state_dict(), ckpt_dir / f"run-{cfg.run_name}-latest.pth")
        if mean_episode_reward > best_episode_reward:
            best_return = best_episode_reward
            torch.save(policy.state_dict(), ckpt_dir / f"run-{cfg.run_name}-best.pth")


# ------------------------------------------------------------------------------------
# Entry point & sweep support
# ------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Default hyper-params (will be overridden by W&B sweep if present)
    default_cfg = dict(
        run_name=f"ppo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        lr=3e-4,
        hidden_size=256,
        rollout_steps=256,
        backprops_per_rollout=4,
        batch_size=16384,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        updates=1000,
    )

    cfg = SimpleNamespace(**default_cfg)
    # If launched by wandb agent, parameters are injected automatically
    if wandb.run is not None and wandb.config is not None:
        cfg.__dict__.update(wandb.config)

    train(cfg)
