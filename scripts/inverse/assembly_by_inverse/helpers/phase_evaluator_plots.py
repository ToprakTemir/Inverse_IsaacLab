
"""
What this script does:
1) Loads HDF5 demonstrations and flattens observations.
2) Loads a trained PhaseEvaluator and plots:
   a) Predicted vs. actual normalized timestamps (scatter over random episodes).
   b) Three 2D heatmaps of predicted phase on XY / YZ / XZ planes with the third
      coordinate fixed to the dataset mean (configurable).

— Tune the CONFIG section for:
    - HDF5 demo path(s)
    - PhaseEvaluator weights path
    - Which three indices correspond to the object (x, y, z) for heatmaps
    - Plane bounds and resolution
"""

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from scripts.inverse.assembly_by_inverse import PhaseEvaluator
from scripts.inverse.assembly_by_inverse.helpers.load_demos import load_demos_from_hdf5

# ---------------------------- CONFIG ---------------------------- #

HDF5_PATH = "../../datasets/disassembly_validation_5.hdf5"
# HDF5_PATH = "../../datasets/disassembly_15.hdf5"

directories = sorted(os.listdir("../../models"))
latest_time = directories[-1]

time = "2025-08-27-20:16"
# time = latest_time

PHASE_EVAL_WEIGHTS = f"../../models/{time}/phase_evaluator_best.pth"
NON_ROBOT_INDICES = slice(8, None) # or list of indices, e.g. [8, 9, 10, ...]

# Heatmap plane bounds
X_MIN, X_MAX = -0.4, 0.4
Y_MIN, Y_MAX = -0.4, 0.4
Z_MIN, Z_MAX =  0.0, 0.3

GRID_RES = 100  # grid resolution per axis
NUM_EPISODES_TO_PLOT = 15  # for the timestamp scatter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG = np.random.default_rng(42)

# --------------------------- Helpers ----------------------------- #

def select_obs_slice(obs: np.ndarray, indices: Optional[Sequence[int]]) -> np.ndarray:
    if indices is None or len(indices) == 0:
        return obs
    return obs[..., indices]


def build_phase_evaluator_for_weights(weights_path: str, input_dim: int):
    # The default architecture in PhaseEvaluator.py is (256, 256) with Sigmoid head.
    model = PhaseEvaluator.PhaseEvaluator(input_dim=input_dim, hidden_dims=(256, 256))
    sd = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()


def get_dataset_means_for_xyz(demos: List[Dict[str, np.ndarray]], xyz_idx: Tuple[int, int, int]) -> Tuple[float, float, float]:
    xs, ys, zs = [], [], []
    ix, iy, iz = xyz_idx
    for d in demos:
        o = d["observations"]
        xs.append(o[:, ix])
        ys.append(o[:, iy])
        zs.append(o[:, iz])
    x_mean = float(np.mean(np.concatenate(xs)))
    y_mean = float(np.mean(np.concatenate(ys)))
    z_mean = float(np.mean(np.concatenate(zs)))
    return x_mean, y_mean, z_mean


# ---------------------------- Plots ------------------------------ #

def plot_pred_vs_actual_timestamps(demos: List[Dict[str, np.ndarray]],
                                   phase_eval: PhaseEvaluator,
                                   slice_indices: Optional[Sequence[int]],
                                   num_episodes: int = 5):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab10', num_episodes)

    chosen = RNG.choice(len(demos), size=min(num_episodes, len(demos)), replace=False)
    for i, idx in enumerate(chosen, start=1):
        demo = demos[idx]
        obs = demo["observations"]
        T = obs.shape[0]
        actual = np.linspace(0.0, 1.0, T, dtype=np.float32)

        with torch.no_grad():
            xs = select_obs_slice(obs, slice_indices)
            xt = torch.as_tensor(xs, dtype=torch.float32, device=DEVICE)
            preds = phase_eval(xt).squeeze(-1).detach().cpu().numpy()

        plt.scatter(actual, preds, s=5, label=f"Episode {i}", color=colors(i - 1))

    # Ideal diagonal
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")

    plt.xlabel("Actual Normalized Time")
    plt.ylabel("Predicted Phase")
    plt.title("PhaseEvaluator: Predicted vs Actual Normalized Timestamps")
    # plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def _grid_eval_on_plane(phase_eval: PhaseEvaluator,
                        fixed_val: float,
                        plane: str,
                        xyz_idx: Tuple[int, int, int],
                        bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                        res: int,
                        in_dim: int):
    """
    Evaluate phase on a 2D plane:
      plane ∈ {"xy", "yz", "xz"}
      fixed_val: the fixed coordinate for the third axis
    """
    ix, iy, iz = xyz_idx
    (u_min, u_max), (v_min, v_max) = bounds
    u = np.linspace(u_min, u_max, res)
    v = np.linspace(v_min, v_max, res)
    U, V = np.meshgrid(u, v)

    # We'll synthesize an obs vector where only xyz indices are filled; others 0.
    # If your evaluator expects more features than xyz, zeros should be acceptable for "just a visualization".
    # For more accurate heatmaps, you can pass typical/mean values for other dims.
    # Build a template of zeros for full obs, then slice if needed.
    # We infer full obs dim from the model's first layer.
    first_linear = None
    for m in phase_eval.modules():
        if isinstance(m, torch.nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        raise RuntimeError("Could not infer input size from PhaseEvaluator.")
    full_obs_dim = first_linear.in_features


    points = []
    if plane == "xy":
        for uu, vv in zip(U.flatten(), V.flatten()):
            full = np.zeros((full_obs_dim,), dtype=np.float32)
            full[ix], full[iy], full[iz] = uu, vv, fixed_val
            points.append(full)
    elif plane == "yz":
        for uu, vv in zip(U.flatten(), V.flatten()):
            full = np.zeros((full_obs_dim,), dtype=np.float32)
            full[ix], full[iy], full[iz] = fixed_val, uu, vv
            points.append(full)
    elif plane == "xz":
        for uu, vv in zip(U.flatten(), V.flatten()):
            full = np.zeros((full_obs_dim,), dtype=np.float32)
            full[ix], full[iy], full[iz] = uu, fixed_val, vv
            points.append(full)
    else:
        raise ValueError("plane must be one of {'xy','yz','xz'}")

    pts = np.stack(points, axis=0)

    with torch.no_grad():
        xt = torch.as_tensor(pts, dtype=torch.float32, device=DEVICE)
        preds = phase_eval(xt).squeeze(-1).detach().cpu().numpy()

    return U, V, preds.reshape(U.shape)


def plot_plane_heatmaps(phase_eval: PhaseEvaluator,
                        demos: List[Dict[str, np.ndarray]],
                        xyz_idx: Tuple[int, int, int],
                        x_bounds: Tuple[float, float],
                        y_bounds: Tuple[float, float],
                        z_bounds: Tuple[float, float],
                        in_dim: int,
                        res: int):
    ix, iy, iz = xyz_idx
    x_mean, y_mean, z_mean = get_dataset_means_for_xyz(demos, xyz_idx)

    # XY at z=z_mean
    U, V, P = _grid_eval_on_plane(
        phase_eval, fixed_val=z_mean, plane="xy", xyz_idx=xyz_idx, bounds=(x_bounds, y_bounds), res=res, in_dim=in_dim)
    plt.figure(figsize=(8, 6))
    plt.imshow(P, extent=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]), origin="lower", interpolation="bilinear", aspect="auto")
    plt.colorbar(label="Predicted Phase")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Phase Heatmap on XY Plane (Z fixed at {z_mean:.3f})")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    # YZ at x=x_mean
    U, V, P = _grid_eval_on_plane(phase_eval, fixed_val=x_mean, plane="yz", xyz_idx=xyz_idx, bounds=(y_bounds, z_bounds), res=res, in_dim=in_dim)
    plt.figure(figsize=(8, 6))
    plt.imshow(P, extent=(y_bounds[0], y_bounds[1], z_bounds[0], z_bounds[1]),
               origin="lower", interpolation="bilinear", aspect="auto")
    plt.colorbar(label="Predicted Phase")
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.title(f"Phase Heatmap on YZ Plane (X fixed at {x_mean:.3f})")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    # XZ at y=y_mean
    U, V, P = _grid_eval_on_plane(
        phase_eval, fixed_val=y_mean, plane="xz", xyz_idx=xyz_idx,bounds=(x_bounds, z_bounds), res=res, in_dim=in_dim)
    plt.figure(figsize=(8, 6))
    plt.imshow(P, extent=(x_bounds[0], x_bounds[1], z_bounds[0], z_bounds[1]),
               origin="lower", interpolation="bilinear", aspect="auto")
    plt.colorbar(label="Predicted Phase")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(f"Phase Heatmap on XZ Plane (Y fixed at {y_mean:.3f})")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# ------------------------------ Main ------------------------------ #

def main():
    demos = load_demos_from_hdf5(HDF5_PATH)
    print(f"{len(demos)} demos loaded from {HDF5_PATH}")

    obs_dim = demos[0]["observations"][0].shape[-1]

    non_robot_indices = list(range(*NON_ROBOT_INDICES.indices(obs_dim)))
    input_dim = len(non_robot_indices)

    model = build_phase_evaluator_for_weights(PHASE_EVAL_WEIGHTS, input_dim=input_dim)

    plot_pred_vs_actual_timestamps(
        demos=demos,
        phase_eval=model,
        slice_indices=non_robot_indices,
        num_episodes=NUM_EPISODES_TO_PLOT,
    )

    # plot_plane_heatmaps(
    #     phase_eval=model,
    #     demos=demos,
    #     xyz_idx=(0, 1, 2),
    #     x_bounds=(X_MIN, X_MAX),
    #     y_bounds=(Y_MIN, Y_MAX),
    #     z_bounds=(Z_MIN, Z_MAX),
    #     res=GRID_RES,
    #     in_dim=input_dim
    # )


if __name__ == "__main__":
    main()
