"""
What this script does:
1) Loads HDF5 demonstrations (your format) and flattens obs.
2) Loads a GaussianPolicy (from pretrain_policy.py) with your obs/action dims.
3) Over a single (or several) episode(s), computes policy predictions and compares
   them to the "reverse-time" target actions with an offset (as in your BC setup).
4) Plots predicted vs. actual joint angles (first 6 dims), optionally gripper.

Notes:
- We compare to actions from the demo (ground-truth). Since the action space is
  7D = 6 joints + 1 binary gripper flip, we plot 6 joints and optionally the gripper.
- If you used observation slicing (non_robot_indices) during training, set it here.
"""

import os
import random
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch

from pretrain_policy import GaussianPolicy


# ---------------------------- CONFIG ---------------------------- #

HDF5_PATH = "path/to/your_demos.hdf5"           # << set me
BC_POLICY_WEIGHTS = "path/to/bc_policy.pth"      # << set me

# Slice obs fed to policy (must match training). If None/empty, use full obs.
NON_ROBOT_INDICES: Optional[Sequence[int]] = None  # e.g., [0,1,2,...]

ACTION_DIM = 7   # 6 joints + 1 gripper
PLOT_GRIPPER = True
JOINT_DIMS_TO_PLOT = 6  # first 6 dims are joints

REVERSE_TIME = True     # imitate inverse behavior as in training
ACTION_OFFSET = 10      # you used 10 earlier; tune here
NUM_EPISODES_TO_PLOT = 1
DET = True              # deterministic: use mean (mu) instead of sampling

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG = np.random.default_rng(42)


# ----------------------- Demo Loader (HDF5) ---------------------- #

def load_demos_from_hdf5(hdf5_path: str) -> List[Dict[str, np.ndarray]]:
    demos: List[Dict[str, np.ndarray]] = []
    with h5py.File(hdf5_path, "r") as f:
        for group_name in f.keys():
            group = f[group_name]
            for demo_name in group.keys():
                demo_group = group[demo_name]

                actions = np.array(demo_group["actions"])
                obs_group = demo_group["obs"]

                obs_arrays = []
                for k in obs_group.keys():
                    arr = np.array(obs_group[k])
                    obs_arrays.append(arr.reshape(arr.shape[0], -1))

                observations = np.concatenate(obs_arrays, axis=-1)

                T = actions.shape[0]
                if observations.shape[0] > T:
                    observations = observations[:T]
                elif observations.shape[0] < T:
                    actions = actions[:observations.shape[0]]

                demos.append({"observations": observations, "actions": actions})
    return demos


# ---------------------------- Policy ----------------------------- #

def build_policy_for_weights(weights_path: str, obs_dim: int, act_dim: int) -> GaussianPolicy:
    policy = GaussianPolicy(obs_dim, act_dim, hidden=(256, 256))
    sd = torch.load(weights_path, map_location="cpu")
    policy.load_state_dict(sd, strict=True)
    return policy.to(DEVICE).eval()


def select_obs_slice(obs: np.ndarray, indices: Optional[Sequence[int]]) -> np.ndarray:
    if indices is None or len(indices) == 0:
        return obs
    return obs[..., indices]


# ---------------------------- Plotting --------------------------- #

def plot_episode_predictions(policy: GaussianPolicy,
                             episode: Dict[str, np.ndarray],
                             slice_indices: Optional[Sequence[int]],
                             reverse_time: bool,
                             action_offset: int,
                             det: bool,
                             plot_gripper: bool):
    obs = episode["observations"]
    acts = episode["actions"]
    T = min(len(obs), len(acts))
    if T <= action_offset:
        raise ValueError(f"Episode too short for action_offset={action_offset}")

    # Build aligned targets: a_tgt = a[t - offset] if reverse_time else a[t + offset]
    s_list, a_tgt_list = [], []
    for t in range(T):
        if reverse_time:
            tgt_t = max(0, t - action_offset)
        else:
            tgt_t = min(T - 1, t + action_offset)
        s = obs[t]
        a = acts[tgt_t]
        s_list.append(s)
        a_tgt_list.append(a)
    S = np.stack(s_list, axis=0)
    A = np.stack(a_tgt_list, axis=0)

    # Predict
    S_in = select_obs_slice(S, slice_indices)
    with torch.no_grad():
        St = torch.as_tensor(S_in, dtype=torch.float32, device=DEVICE)
        mu, std = policy(St)
        pred = mu if det else (mu + std * torch.randn_like(std))
        Pred = pred.detach().cpu().numpy()

    # Plot joints (first 6 dims)
    steps = np.arange(T)
    colors = plt.cm.get_cmap('tab10', max(JOINT_DIMS_TO_PLOT, 1))
    plt.figure(figsize=(12, 8))
    for j in range(JOINT_DIMS_TO_PLOT):
        plt.plot(steps, Pred[:, j], label=f"Pred Joint {j}", linewidth=2, alpha=0.9, color=colors(j))
    for j in range(JOINT_DIMS_TO_PLOT):
        plt.plot(steps, A[:, j], label=f"Actual Joint {j}", linewidth=1.8, linestyle="--", color="black", alpha=0.6)

    plt.xlabel("Steps")
    plt.ylabel("Joint Angle")
    plt.title("Initial Policy: Predicted vs Actual Joint Angles")
    # plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, ncol=1)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Optional: gripper flip (7th dim)
    if plot_gripper and A.shape[1] >= 7:
        plt.figure(figsize=(12, 4))
        plt.plot(steps, Pred[:, 6], label="Pred Gripper", linewidth=2)
        plt.plot(steps, A[:, 6], label="Actual Gripper", linewidth=1.8, linestyle="--")
        plt.xlabel("Steps")
        plt.ylabel("Gripper (logits/angle)")
        plt.title("Initial Policy: Predicted vs Actual Gripper Channel")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()


# ------------------------------ Main ------------------------------ #

def main():
    demos = load_demos_from_hdf5(HDF5_PATH)
    if len(demos) == 0:
        raise RuntimeError(f"No demos loaded from {HDF5_PATH}")

    obs_dim_full = demos[0]["observations"].shape[-1]
    obs_dim_in = len(NON_ROBOT_INDICES) if (NON_ROBOT_INDICES and len(NON_ROBOT_INDICES) > 0) else obs_dim_full

    policy = build_policy_for_weights(BC_POLICY_WEIGHTS, obs_dim=obs_dim_in, act_dim=ACTION_DIM)

    chosen = RNG.choice(len(demos), size=min(NUM_EPISODES_TO_PLOT, len(demos)), replace=False)
    for idx in chosen:
        episode = demos[idx]
        plot_episode_predictions(policy, episode, NON_ROBOT_INDICES, REVERSE_TIME, ACTION_OFFSET, DET, PLOT_GRIPPER)


if __name__ == "__main__":
    main()
