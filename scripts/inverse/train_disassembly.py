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

from isaaclab.envs import ManagerBasedRLEnv
from InverseAssemblyProject.tasks.manager_based.assembled_start.assembled_start_cfg import AssembledStartEnvCfg


# ------------------------------------------------------------------------------------
# PPO utilities
# ------------------------------------------------------------------------------------
class MLPPolicy(nn.Module):
    """Shared-backbone actor–critic with separate heads."""
    def __init__(self, obs_shape: int, act_shape: int, hidden=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_shape, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden, act_shape)
        self.log_std     = nn.Parameter(torch.zeros(act_shape))
        self.critic      = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.backbone(x)
        return self.actor_mean(x), self.log_std.exp(), self.critic(x)


class RolloutBuffer:
    """Stores experience for one PPO iteration."""
    def __init__(self, n_envs, steps, obs_dim, act_dim, device):
        self.n_envs, self.steps = n_envs, steps
        self.device = device
        self.obs   = torch.zeros(steps, n_envs, obs_dim, device=device)
        self.acts  = torch.zeros(steps, n_envs, act_dim, device=device)
        self.rews  = torch.zeros(steps, n_envs, device=device)
        self.dones = torch.zeros(steps, n_envs, device=device)
        self.logps = torch.zeros(steps, n_envs, device=device)
        self.vals  = torch.zeros(steps, n_envs, device=device)
        self.ptr = 0

    def add(self, obs, act, rew, done, logp, val):
        self.obs[self.ptr]  = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.dones[self.ptr] = done
        self.logps[self.ptr] = logp
        self.vals[self.ptr]  = val
        self.ptr += 1

    def compute_returns_and_adv(self, gamma, lam):
        adv = torch.zeros_like(self.rews, device=self.device)
        last_gae = torch.zeros(self.n_envs, device=self.device)
        for t in reversed(range(self.steps)):
            next_vals = self.vals[t+1] if t < self.steps-1 else 0.0
            mask = 1.0 - self.dones[t]
            delta = self.rews[t] + gamma * next_vals * mask - self.vals[t]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        returns = adv + self.vals
        self.adv, self.returns = adv.flatten(), returns.flatten()

    def get(self, batch_size):
        idx = torch.randperm(self.steps * self.n_envs, device=self.device)
        for start in range(0, len(idx), batch_size):
            end = start + batch_size
            b = idx[start:end]
            yield (
                self.obs.flatten(0,1)[b],
                self.acts.flatten(0,1)[b],
                self.logps.flatten()[b],
                self.adv[b],
                self.returns[b],
            )


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

    for update in range(config.updates):

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
            mean_episode_reward += rew.mean().item()
            done = terminated | time_out
            next_obs = next_obs["policy"]  # extract policy observation
            buf.add(
                obs, act, torch.as_tensor(rew, device=device),
                torch.as_tensor(done, device=device),
                logp, val.squeeze(-1),
            )
            obs = torch.as_tensor(next_obs, device=device)
            global_step += n_envs

        with torch.no_grad():
            _, _, last_val = policy(obs)
            last_val = last_val.squeeze(-1)
        buf.vals[-1] = last_val            # bootstrap final value
        buf.compute_returns_and_adv(config.gamma, config.gae_lambda)

        # Update policy
        for epoch in range(config.epochs):
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

        # Logging ------------------------------------------------------------------
        wandb.log({
            "mean_reward": mean_episode_reward,
            "learning_rate": optim_.param_groups[0]["lr"],
            "global_step": global_step,
        })

        # Check-pointing ------------------------------------------------------------
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(policy.state_dict(), ckpt_dir / "ppo_disassembly_latest.pth")
        if mean_episode_reward > best_episode_reward:
            best_return = best_episode_reward
            torch.save(policy.state_dict(), ckpt_dir / "ppo_disassembly_best.pth")


# ------------------------------------------------------------------------------------
# Entry point & sweep support
# ------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Default hyper-params (will be overridden by W&B sweep if present)
    default_cfg = dict(
        run_name        = f"ppo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        lr              = 3e-4,
        hidden_size     = 256,
        rollout_steps   = 1024,        # 1024 * 512 envs ≈ 0.5 M samples / update
        epochs          = 4,
        batch_size      = 16384,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_eps        = 0.2,
        ent_coef        = 0.0,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        updates         = 10_000,
    )

    cfg = SimpleNamespace(**default_cfg)
    # If launched by wandb agent, parameters are injected automatically
    if wandb.run is not None and wandb.config is not None:
        cfg.__dict__.update(wandb.config)

    train(cfg)
