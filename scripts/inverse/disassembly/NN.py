
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

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

    def act(self, obs, deterministic=False):
        """Returns action and log probability."""
        mean, std, _ = self(obs)
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)
        return action, logp, mean, std


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