from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
app = app_launcher.app

from InverseAssemblyProject.tasks.manager_based.assembled_start.assembled_start_cfg import AssembledStartEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from PhaseEvaluator import (
    PhaseEvaluator,
    PhaseDemoDataset,
    PhaseTrainConfig,
    train_phase_evaluator,
)
from pretrain_policy import (
    GaussianPolicy,
    BCDemoDataset,
    BCConfig,
    train_bc_policy,
)


# ---------------------------- Demo Loading ---------------------------- #

def load_demos_from_npz(npz_path: str) -> List[dict]:
    """
    Minimal example loader. Expected file format:
      observations: list of arrays or a single (N, T_i, obs_dim) concatenated
      actions:      list of arrays or (N, T_i, act_dim)
    Replace with your actual demo reader (Minari-like, your own logger, etc).
    """
    data = np.load(npz_path, allow_pickle=True)
    demos = []

    obs = data["observations"]  # could be an array of objects or a 3D array
    acts = data["actions"]

    if obs.dtype == object:
        # list of trajectories
        assert len(obs) == len(acts)
        for o, a in zip(obs, acts):
            demos.append({"observations": o, "actions": a})
    else:
        # stacked; assume first dim = episodes
        assert obs.shape[0] == acts.shape[0]
        for i in range(obs.shape[0]):
            demos.append({"observations": obs[i], "actions": acts[i]})
    return demos


# -------------------- Reward Wrapper using Phase --------------------- #

class PhaseRewardWrapper(gym.Wrapper):
    """
    Adds a shaping reward based on predicted phase from PhaseEvaluator:
      r_new = r_base + reward_weight * (1 - phase_t)
    where phase_t ∈ [0,1], ideally 0 near the *initial* state.
    You can flip sign or change shaping as needed.
    """
    def __init__(
        self,
        env: gym.Env,
        phase_eval: PhaseEvaluator,
        non_robot_indices: Optional[Sequence[int]],
        reward_weight: float = 1.0,
        device: Optional[torch.device] = None,
        obs_is_dict_key: Optional[str] = None,
    ):
        super().__init__(env)
        self.phase_eval = phase_eval.eval()
        self.non_robot_indices = list(non_robot_indices) if non_robot_indices else []
        self.reward_weight = reward_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_is_dict_key = obs_is_dict_key  # if env.obs is a dict, pick a key to run evaluator on
        self.phase_eval.to(self.device)

    def _extract_obs_for_phase(self, obs) -> np.ndarray:
        """
        Supports raw ndarray or dict observations. Extend if your env returns tuples, etc.
        """
        if isinstance(obs, dict):
            key = self.obs_is_dict_key or next(iter(obs.keys()))
            x = obs[key]
        else:
            x = obs

        if self.non_robot_indices:
            x = x[..., self.non_robot_indices]
        return x

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x = self._extract_obs_for_phase(obs)
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            phase = self.phase_eval(xt).squeeze().item()  # [0,1]

        shaped = reward + self.reward_weight * (1.0 - phase)
        info["phase"] = phase
        info["reward_phase"] = self.reward_weight * (1.0 - phase)
        return obs, shaped, terminated, truncated, info


# ------------------------------ Agent -------------------------------- #

class InverseAgent(nn.Module):
    def __init__(
        self,
        env: ManagerBasedRLEnv,
        demos: List[dict],
        validation_demos: Optional[List[dict]] = None,
        non_robot_indices_in_obs: Optional[Sequence[int]] = None,
        hyperparams: Optional[dict] = None,
    ):
        super().__init__()
        self.env = env
        self.demos = demos
        self.validation_demos = validation_demos
        self.non_robot_indices_in_obs = list(non_robot_indices_in_obs) if non_robot_indices_in_obs else []
        self.hyperparams = hyperparams or {}

        self.phase_evaluator: Optional[PhaseEvaluator] = None
        self.phase_evaluator_trained: bool = False

        self.bc_policy: Optional[GaussianPolicy] = None
        self.bc_trained: bool = False

        self.inverse_model: Optional[PPO] = None

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------ Phase Evaluator ------------------------ #

    def build_phase_evaluator(self, obs_dim_full: int) -> None:
        in_dim = obs_dim_full if not self.non_robot_indices_in_obs else len(self.non_robot_indices_in_obs)
        hidden = self.hyperparams.get("phase_hidden", (256, 256))
        self.phase_evaluator = PhaseEvaluator(in_dim, hidden_dims=hidden).to(self.device)

    def train_phase_evaluator(self, save_best_path: Optional[str] = None) -> None:
        # infer obs_dim from demos
        sample_obs = self.demos[0]["observations"][0]
        obs_dim_full = sample_obs.shape[-1]
        self.build_phase_evaluator(obs_dim_full)

        train_ds = PhaseDemoDataset(
            self.demos,
            non_robot_indices=self.non_robot_indices_in_obs,
            sample_per_traj=self.hyperparams.get("phase_samples_per_traj", None),
            device=self.device,
        )
        val_ds = None
        if self.validation_demos is not None:
            val_ds = PhaseDemoDataset(
                self.validation_demos,
                non_robot_indices=self.non_robot_indices_in_obs,
                sample_per_traj=self.hyperparams.get("phase_val_samples_per_traj", None),
                device=self.device,
            )

        cfg = PhaseTrainConfig(
            lr=self.hyperparams.get("phase_lr", 1e-3),
            batch_size=self.hyperparams.get("phase_batch", 256),
            epochs=self.hyperparams.get("phase_epochs", 10_000),
            tv_weight=self.hyperparams.get("phase_tv_weight", 0.15),
            val_period=self.hyperparams.get("phase_val_period", 100),
            patience=self.hyperparams.get("phase_patience", 2_000),
            num_workers=self.hyperparams.get("num_workers", 0),
            shuffle=True,
        )

        train_phase_evaluator(
            self.phase_evaluator, train_ds, val_ds, cfg, device=self.device, save_best_path=save_best_path
        )
        self.phase_evaluator_trained = True

    # --------------------------- BC Policy -------------------------- #

    def build_bc_policy(self, obs_dim_full: int, act_dim: int) -> None:
        in_dim = obs_dim_full if not self.non_robot_indices_in_obs else len(self.non_robot_indices_in_obs)
        hidden = self.hyperparams.get("bc_hidden", (256, 256))
        self.bc_policy = GaussianPolicy(in_dim, act_dim, hidden=hidden).to(self.device)

    def pretrain_bc_policy(self, save_best_path: Optional[str] = None) -> None:
        sample_obs = self.demos[0]["observations"][0]
        sample_act = self.demos[0]["actions"][0]
        obs_dim_full = sample_obs.shape[-1]
        act_dim = sample_act.shape[-1]

        self.build_bc_policy(obs_dim_full, act_dim)

        train_ds = BCDemoDataset(
            self.demos,
            non_robot_indices=self.non_robot_indices_in_obs,
            action_offset=self.hyperparams.get("bc_action_offset", 1),   # you used 10 previously
            reverse_time=self.hyperparams.get("bc_reverse_time", True),
            device=self.device,
            sample_per_traj=self.hyperparams.get("bc_samples_per_traj", None),
        )
        val_ds = None
        if self.validation_demos is not None:
            val_ds = BCDemoDataset(
                self.validation_demos,
                non_robot_indices=self.non_robot_indices_in_obs,
                action_offset=self.hyperparams.get("bc_action_offset", 1),
                reverse_time=self.hyperparams.get("bc_reverse_time", True),
                device=self.device,
                sample_per_traj=self.hyperparams.get("bc_val_samples_per_traj", None),
            )

        cfg = BCConfig(
            lr=self.hyperparams.get("bc_lr", 1e-3),
            batch_size=self.hyperparams.get("bc_batch", 256),
            epochs=self.hyperparams.get("bc_epochs", 50_000),
            logprob_loss=self.hyperparams.get("bc_logprob_loss", True),
            val_period=self.hyperparams.get("bc_val_period", 500),
            patience=self.hyperparams.get("bc_patience", 10_000),
            num_workers=self.hyperparams.get("num_workers", 0),
            shuffle=True,
        )

        train_bc_policy(
            self.bc_policy, train_ds, val_ds, cfg, device=self.device, save_best_path=save_best_path
        )
        self.bc_trained = True

    # ------------------------ PPO Finetuning ------------------------ #

    def _make_wrapped_env(self, reward_weight: float = 1.0) -> gym.Env:
        assert self.phase_evaluator is not None and self.phase_evaluator_trained
        # Isaac Lab envs are gymnasium-compatible
        env: gym.Env = self.env

        wrapped = PhaseRewardWrapper(
            env,
            self.phase_evaluator,
            non_robot_indices=self.non_robot_indices_in_obs,
            reward_weight=reward_weight,
            device=self.device,
            obs_is_dict_key=self.hyperparams.get("obs_is_dict_key", None),
        )
        return wrapped

    def _init_ppo(self, env: gym.Env) -> PPO:
        # Use SB3 defaults; feel free to tune
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device=self.device,
            n_steps=self.hyperparams.get("ppo_n_steps", 2048),
            batch_size=self.hyperparams.get("ppo_batch_size", 256),
            learning_rate=self.hyperparams.get("ppo_lr", 3e-4),
            ent_coef=self.hyperparams.get("ppo_ent_coef", 0.0),
            vf_coef=self.hyperparams.get("ppo_vf_coef", 1.0),
            gae_lambda=self.hyperparams.get("ppo_gae_lambda", 0.95),
            gamma=self.hyperparams.get("ppo_gamma", 0.99),
            clip_range=self.hyperparams.get("ppo_clip_range", 0.2),
        )

        # Optional: load BC weights into PPO policy's actor (best-effort name match)
        if self.bc_policy is not None and self.bc_trained:
            with torch.no_grad():
                ppo_sd = model.policy.state_dict()
                bc_sd = self.bc_policy.state_dict()

                # Map common layers by heuristic: mlp extractor / shared net vs bc.mlp
                # Simple strategy: copy layers with same names or same shapes
                mapped = 0
                for k_ppo, v_ppo in ppo_sd.items():
                    if not isinstance(v_ppo, torch.Tensor):
                        continue
                    # try same name
                    if k_ppo in bc_sd and bc_sd[k_ppo].shape == v_ppo.shape:
                        ppo_sd[k_ppo].copy_(bc_sd[k_ppo])
                        mapped += 1
                    # try bc.mlp.* to policy.mlp_extractor.policy_net.*
                    elif k_ppo.startswith("mlp_extractor.policy_net") and "mlp" in bc_sd:
                        # Best-effort: find next unused LIN in order
                        pass
                if mapped > 0:
                    model.policy.load_state_dict(ppo_sd, strict=False)
                    print(f"[PPO Init] Loaded {mapped} parameter tensors from BC policy into PPO actor.")
        return model

    def finetune_with_ppo(self, total_timesteps: int, reward_weight: float = 1.0, log_dir: Optional[str] = None):
        env_wrapped = self._make_wrapped_env(reward_weight=reward_weight)

        # vectorize as DummyVec for SB3
        venv = DummyVecEnv([lambda: env_wrapped])

        model = self._init_ppo(venv)

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        self.inverse_model = model

    # ----------------------------- IO ------------------------------- #

    def save_phase_evaluator(self, path: str):
        assert self.phase_evaluator is not None
        torch.save(self.phase_evaluator.state_dict(), path)

    def load_phase_evaluator(self, path: str, obs_dim_full: int):
        self.build_phase_evaluator(obs_dim_full)
        self.phase_evaluator.load_state_dict(torch.load(path, map_location=self.device))
        self.phase_evaluator.to(self.device).eval()
        self.phase_evaluator_trained = True

    def save_bc_policy(self, path: str):
        assert self.bc_policy is not None
        torch.save(self.bc_policy.state_dict(), path)

    def load_bc_policy(self, path: str, obs_dim_full: int, act_dim: int):
        self.build_bc_policy(obs_dim_full, act_dim)
        self.bc_policy.load_state_dict(torch.load(path, map_location=self.device))
        self.bc_policy.to(self.device).eval()
        self.bc_trained = True

    def save_inverse_model(self, path: str):
        assert self.inverse_model is not None
        self.inverse_model.save(path)


# ----------------------------- Script ------------------------------- #

if __name__ == "__main__":
    # 1) Create Isaac Lab env (Manager-Based RL)
    cfg = AssembledStartEnvCfg()
    env = ManagerBasedRLEnv(cfg)

    # 2) Load demonstrations (replace with your own loader!)
    # Example expected .npz with arrays/lists of trajectories
    # npz_path = "path/to/your_demos.npz"
    # demos = load_demos_from_npz(npz_path)
    # val_demos = load_demos_from_npz("path/to/val_demos.npz")
    # For now, raise if you haven't set this up:
    raise_if_missing = False  # flip to True to enforce
    demos: List[dict] = []    # fill me with your (obs, actions)
    val_demos: Optional[List[dict]] = None

    if raise_if_missing and len(demos) == 0:
        raise RuntimeError("Please load your demonstrations into `demos` before running.")

    # 3) Configure indices (non-robot obs) and hyperparams
    non_robot_indices = []  # e.g., [0,1,2] as you used for object position
    hparams = dict(
        # phase/evaluator
        phase_hidden=(256, 256),
        phase_lr=1e-3,
        phase_batch=256,
        phase_epochs=10_000,
        phase_tv_weight=0.15,
        phase_val_period=100,
        phase_patience=2_000,

        # bc
        bc_hidden=(256, 256),
        bc_lr=1e-3,
        bc_batch=256,
        bc_epochs=50_000,
        bc_logprob_loss=True,
        bc_action_offset=10,     # you used 10 before
        bc_reverse_time=True,

        # ppo
        ppo_n_steps=2048,
        ppo_batch_size=256,
        ppo_lr=3e-4,
        ppo_ent_coef=0.0,
        ppo_vf_coef=1.0,
        ppo_gamma=0.99,
        ppo_gae_lambda=0.95,
        ppo_clip_range=0.2,

        # obs dict key (if your env returns dict obs)
        obs_is_dict_key=None,
    )

    agent = InverseAgent(
        env=env,
        demos=demos,
        validation_demos=val_demos,
        non_robot_indices_in_obs=non_robot_indices,
        hyperparams=hparams,
    )

    # 4) Train the Phase Evaluator (Evaluator)
    # agent.train_phase_evaluator(save_best_path="checkpoints/phase_evaluator_best.pth")

    # 5) Pretrain BC policy
    # agent.pretrain_bc_policy(save_best_path="checkpoints/bc_policy_best.pth")

    # 6) PPO Finetune with phase-shaped rewards
    # agent.finetune_with_ppo(total_timesteps=10_000_000, reward_weight=1.0, log_dir="ppo_logs")

    # 7) Save PPO model
    # agent.save_inverse_model("checkpoints/inverse_skill_ppo.zip")