from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader


class GaussianPolicy(nn.Module):
    """
    Simple Gaussian policy for continuous actions:
      inputs: observation
      outputs: action distribution (mean, std)
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (256, 256), log_std_init: float = -0.5):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(last, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(obs)
        mu = self.mu(h)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std

    def dist(self, obs: torch.Tensor) -> Normal:
        mu, std = self(obs)
        return Normal(mu, std)


class BCDemoDataset(Dataset):
    """
    Creates (state, target_action) pairs from demonstrations.
    You can choose reversed-time pairing to imitate the inverse behavior:
      target_action(t) := action(t - offset), offset >= 1
    By default offset = 1; you used (t - 10) previously; that’s configurable.
    """
    def __init__(
        self,
        demos: List[dict],
        non_robot_indices: Optional[Sequence[int]],
        action_offset: int = 1,
        reverse_time: bool = True,
        device: Optional[torch.device] = None,
        sample_per_traj: Optional[int] = None,
    ):
        self.samples = []
        self.device = device
        self.non_robot_indices = non_robot_indices

        for demo in demos:
            obs = demo["observations"]  # (T, obs_dim)
            act = demo["actions"]       # (T, act_dim)
            T = min(len(obs), len(act))
            if T < (action_offset + 1):
                continue

            if sample_per_traj is None or sample_per_traj >= T:
                idxs = np.arange(T)
            else:
                idxs = np.random.choice(T, size=sample_per_traj, replace=False)

            for t in idxs:
                # clamp valid target index depending on direction
                if reverse_time:
                    target_t = max(0, t - action_offset)
                else:
                    target_t = min(T - 1, t + action_offset)

                s = obs[t]
                if non_robot_indices is not None and len(non_robot_indices) > 0:
                    s = s[non_robot_indices]
                a = act[target_t]
                self.samples.append((s.astype(np.float32), a.astype(np.float32)))

        if len(self.samples) == 0:
            raise ValueError("BCDemoDataset constructed with 0 samples. Check demos/action_offset.")

        self.obs_dim = self.samples[0][0].shape[0]
        self.act_dim = self.samples[0][1].shape[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s, a = self.samples[idx]
        st = torch.from_numpy(s)
        at = torch.from_numpy(a)
        if self.device is not None:
            st = st.to(self.device)
            at = at.to(self.device)
        return st, at


@dataclass
class BCConfig:
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 50_000
    logprob_loss: bool = True   # if False, MSE on actions
    val_period: int = 500
    patience: int = 10_000
    num_workers: int = 0
    shuffle: bool = True


def train_bc_policy(
        policy: GaussianPolicy,
        train_ds: BCDemoDataset,
        val_ds: Optional[BCDemoDataset] = None,
        cfg: BCConfig = BCConfig(),
        device: Optional[torch.device] = None,
        save_best_path: Optional[str] = None,
        save_latest_path: Optional[str] = None,
) -> dict:
    """
    Behavior Cloning trainer. Returns history dict.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=cfg.shuffle,
        num_workers=cfg.num_workers, drop_last=True
    )
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)

    opt = optim.Adam(policy.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()
    hist = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    bad = 0

    for epoch in range(cfg.epochs):
        policy.train()
        losses = []
        for s, a in dl:
            s = s.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            if cfg.logprob_loss:
                dist = policy.dist(s)
                # For multidim actions, Normal.log_prob returns (B, act_dim); sum over dims
                lp = dist.log_prob(a).sum(dim=-1)
                loss = -lp.mean()
            else:
                mu, _ = policy(s)
                loss = mse(mu, a)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        hist["train_loss"].append(float(np.mean(losses)))
        print(f"Epoch {epoch + 1}/{cfg.epochs}, Train Loss: {hist['train_loss'][-1]:.4f}")

        if val_dl is not None and (epoch % cfg.val_period == 0):
            policy.eval()
            vlosses = []
            with torch.no_grad():
                for s, a in val_dl:
                    s = s.to(device, non_blocking=True)
                    a = a.to(device, non_blocking=True)
                    if cfg.logprob_loss:
                        dist = policy.dist(s)
                        lp = dist.log_prob(a).sum(dim=-1)
                        vlosses.append(float((-lp.mean()).item()))
                    else:
                        mu, _ = policy(s)
                        vlosses.append(float(mse(mu, a).item()))
            v = float(np.mean(vlosses))
            hist["val_loss"].append(v)

            print(f"Epoch {epoch + 1}/{cfg.epochs}, Val Loss: {v:.4f}")
            torch.save(policy.state_dict(), save_latest_path)

            # best model check
            if v < best_val:
                best_val = v
                bad = 0
                if save_best_path is not None:
                    torch.save(policy.state_dict(), save_best_path)
                    print("Saving best model to", save_best_path)
            else:
                bad += cfg.val_period
                if bad >= cfg.patience:
                    break

    return hist