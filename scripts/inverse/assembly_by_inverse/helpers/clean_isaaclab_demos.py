#!/usr/bin/env python3
"""
Isaac Lab HDF5 demo cleaner (short, hardcoded).
Removes:
  1) All leading frames before the first non-zero action.
  2) Any mid-episode frames where action≈0 AND observation unchanged.

Edit the CONFIG below, then run this file.
"""
from __future__ import annotations
import h5py
import numpy as np

# ========= CONFIG (EDIT THESE) =========
INPUT_PATH  = "../../datasets/dataset.hdf5"
OUTPUT_PATH = "../../datasets/only_pullout_validation_5.hdf5"

# Prefer processed_actions if present; fall back to actions
ACTIONS_PRIMARY   = "processed_actions"
ACTIONS_FALLBACK  = "actions"

# We'll build observations by concatenating all direct children under this group
OBS_GROUP   = "obs"   # e.g., obs/ft_sensor, obs/joint_positions, ...

# Episode groups are named like demo_0, demo_1, ...
GROUP_PREFIXES = ("demo_",)

EPS_ACTION  = 1e-8    # near-zero threshold for action vectors
EPS_OBS     = 1e-8    # near-equal threshold for observations
DROP_PREFIX_ONLY = False
DRY_RUN = False
# ======================================


def _copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _is_all_zero_actions(actions: np.ndarray, eps: float) -> np.ndarray:
    """Boolean mask: action is (near-)zero at each t. Works for (T,) or (T,D)."""
    a = np.abs(actions)
    if a.ndim == 1:
        return a <= eps
    return np.max(a, axis=-1) <= eps


def _obs_equal(o1: np.ndarray, o2: np.ndarray, eps: float) -> bool:
    """True if two obs vectors are effectively identical."""
    if o1.shape != o2.shape:  # conservative
        return False
    if not np.issubdtype(o1.dtype, np.floating):
        o1 = o1.astype(np.float32, copy=False)
    if not np.issubdtype(o2.dtype, np.floating):
        o2 = o2.astype(np.float32, copy=False)
    return np.max(np.abs(o1 - o2)) <= eps


def _keep_mask(actions: np.ndarray, obs: np.ndarray|None,
               eps_action: float, eps_obs: float, drop_prefix_only: bool) -> np.ndarray:
    """Return boolean mask of timesteps to keep."""
    T = actions.shape[0]
    zero_mask = _is_all_zero_actions(actions, eps_action)

    # 1) trim leading all-zero prefix
    first_nz = np.argmax(~zero_mask) if np.any(~zero_mask) else None
    if first_nz is None:
        return np.zeros(T, dtype=bool)  # entire episode is idle

    keep = np.ones(T, dtype=bool)
    keep[:first_nz] = False
    if drop_prefix_only or obs is None:
        return keep

    # 2) drop mid-episode frames where (action≈0 AND obs unchanged from last kept frame)
    last_kept_obs = None
    for t in range(T):
        if not keep[t]:
            continue
        if last_kept_obs is None:
            last_kept_obs = obs[t]
            continue
        if zero_mask[t] and _obs_equal(obs[t], last_kept_obs, eps_obs):
            keep[t] = False
        else:
            last_kept_obs = obs[t]
    return keep


def _episode_groups(f: h5py.File) -> list[str]:
    names = [n for n, it in f.items() if isinstance(it, h5py.Group) and n.startswith(GROUP_PREFIXES)]
    return names or ["/"]


def _pick_actions_group(g) -> tuple[str, np.ndarray]:
    """Choose actions array: prefer ACTIONS_PRIMARY, else ACTIONS_FALLBACK."""
    if ACTIONS_PRIMARY in g:
        return ACTIONS_PRIMARY, np.array(g[ACTIONS_PRIMARY])
    if ACTIONS_FALLBACK in g:
        return ACTIONS_FALLBACK, np.array(g[ACTIONS_FALLBACK])
    raise KeyError(f"No actions found in group '{g.name}' (looked for '{ACTIONS_PRIMARY}' and '{ACTIONS_FALLBACK}').")


def _concat_obs_group(g, T: int) -> np.ndarray | None:
    """
    Concatenate all datasets directly under g[OBS_GROUP] along the last axis,
    keeping only those with leading dimension == T. Returns (T, Dsum) or None.
    """
    if OBS_GROUP not in g:
        return None
    og = g[OBS_GROUP]
    parts = []
    for name, obj in og.items():
        if not isinstance(obj, h5py.Dataset):
            continue
        arr = np.array(obj)
        if arr.ndim >= 1 and arr.shape[0] == T:
            parts.append(arr.reshape(T, -1))
    if not parts:
        return None
    return np.concatenate(parts, axis=-1)


def clean_file(inp: str, outp: str) -> dict[str, int]:
    summary = {"episodes": 0, "frames_total": 0, "frames_kept": 0, "frames_dropped": 0}
    with h5py.File(inp, "r") as fin:
        ep_names = _episode_groups(fin)
        if DRY_RUN:
            for ep in ep_names:
                g = fin if ep == "/" else fin[ep]
                try:
                    act_key, actions = _pick_actions_group(g)
                except KeyError:
                    continue
                T = len(actions)
                obs = _concat_obs_group(g, T)
                keep = _keep_mask(actions, obs, EPS_ACTION, EPS_OBS, DROP_PREFIX_ONLY)

                summary["episodes"] += 1
                summary["frames_total"] += T
                summary["frames_kept"] += int(np.count_nonzero(keep))
                summary["frames_dropped"] += int(np.count_nonzero(~keep))
            return summary

        with h5py.File(outp, "w") as fout:
            _copy_attrs(fin, fout)
            for ep in ep_names:
                src = fin if ep == "/" else fin[ep]
                dst = fout if ep == "/" else fout.create_group(ep)
                _copy_attrs(src, dst)

                # --- build keep mask from actions + concatenated obs ---
                try:
                    act_key, actions = _pick_actions_group(src)
                except KeyError:
                    # No actions: copy group verbatim
                    for name, obj in src.items():
                        if isinstance(obj, h5py.Dataset):
                            dst.create_dataset(name, data=obj[...], compression="gzip", compression_opts=4)
                            _copy_attrs(obj, dst[name])
                        elif isinstance(obj, h5py.Group):
                            fin.copy(obj, dst, name=name)
                    continue

                T = len(actions)
                obs = _concat_obs_group(src, T)
                keep = _keep_mask(actions, obs, EPS_ACTION, EPS_OBS, DROP_PREFIX_ONLY)

                # --- recursive copy with filtering on leading dimension ---
                def copy_filter(node_src, node_dst):
                    for name, obj in node_src.items():
                        if isinstance(obj, h5py.Dataset):
                            data = obj[...]
                            # Filter any dataset whose first dimension equals T (and T>1)
                            if data.ndim >= 1 and data.shape[0] == T and T > 1:
                                data = data[keep]
                            d = node_dst.create_dataset(name, data=data, compression="gzip", compression_opts=4)
                            _copy_attrs(obj, d)
                        elif isinstance(obj, h5py.Group):
                            sub = node_dst.create_group(name)
                            _copy_attrs(obj, sub)
                            copy_filter(obj, sub)

                copy_filter(src, dst)

                # Update summary
                summary["episodes"] += 1
                summary["frames_total"] += T
                summary["frames_kept"] += int(np.count_nonzero(keep))
                summary["frames_dropped"] += int(np.count_nonzero(~keep))

    return summary


if __name__ == "__main__":
    s = clean_file(INPUT_PATH, OUTPUT_PATH)
    print(f"[SUMMARY] Episodes: {s['episodes']} | Frames: {s['frames_total']} | "
          f"Kept: {s['frames_kept']} | Dropped: {s['frames_dropped']}")