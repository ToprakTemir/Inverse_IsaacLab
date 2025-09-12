from __future__ import annotations
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnNoModelImprovement, EvalCallback

import argparse
import os
from typing import List, Optional, Sequence, Dict

import numpy as np
import gymnasium as gym
import datetime

import torch
import torch.nn as nn

import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3 import PPO

from helpers.load_demos import load_demos_from_hdf5

# -------------------- Reward Wrapper using Phase --------------------- #


class PhaseRewardVecWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that adds shaping: r_new = r_base + w * (1 - phase_t)
    where phase_t in [0,1] predicted by a PhaseEvaluator network.

    Handles dict observations via `obs_is_dict_key`.
    `non_robot_indices` selects a subvector from the observation before feeding the phase net.
    """
    def __init__(
        self,
        venv,
        phase_eval: PhaseEvaluator,
        non_robot_indices: Optional[Sequence[int]] = None,
        reward_weight: float = 1.0,
        device: Optional[torch.device] = None,
        obs_is_dict_key: Optional[str] = None,
    ):
        super().__init__(venv)
        self.phase_eval = phase_eval.eval()
        self.non_robot_indices = list(non_robot_indices) if non_robot_indices else []
        self.reward_weight = reward_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_is_dict_key = obs_is_dict_key
        self.phase_eval.to(self.device)

    # --- utils ---
    def _extract_batch(self, obs) -> np.ndarray:
        # obs can be np.ndarray, or dict[str, np.ndarray] with shape (n_envs, *dims)
        if isinstance(obs, dict):
            key = self.obs_is_dict_key or next(iter(obs.keys()))
            x = obs[key]
        else:
            x = obs
        if self.non_robot_indices:
            x = x[..., self.non_robot_indices]
        return x

    def _predict_phase(self, x_np: np.ndarray) -> np.ndarray:
        # x_np shape: (n_envs, obs_dim_selected)
        xt = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            phase = self.phase_eval(xt).squeeze(-1)  # shape (n_envs,)
        return phase.clamp(0.0, 1.0).detach().cpu().numpy()

    # --- VecEnv interface overrides ---
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        x = self._extract_batch(obs)
        phases = self._predict_phase(x)  # (n_envs,)
        shaped_bonus = self.reward_weight * (1.0 - phases)
        rewards = rewards + shaped_bonus

        # annotate infos per env
        for i in range(len(infos)):
            if infos[i] is None:
                infos[i] = {}
            infos[i]["phase"] = float(phases[i])
            infos[i]["reward_phase"] = float(shaped_bonus[i])
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()


class PhaseMetricsCallback(BaseCallback):
    """
    Aggregates 'phase' and 'reward_phase' from infos (added by PhaseRewardVecWrapper)
    and logs their running means to W&B every rollout.
    """
    def __init__(self, log_every_n_rollouts: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.n = 0
        self.buffer_phase = []
        self.buffer_rp = []
        self.log_every = log_every_n_rollouts

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict):
                if "phase" in info:
                    self.buffer_phase.append(info["phase"])
                if "reward_phase" in info:
                    self.buffer_rp.append(info["reward_phase"])
        return True

    def _on_rollout_end(self) -> None:
        self.n += 1
        if self.n % self.log_every == 0 and (self.buffer_phase or self.buffer_rp):
            mean_phase = float(np.mean(self.buffer_phase)) if self.buffer_phase else None
            mean_rp = float(np.mean(self.buffer_rp)) if self.buffer_rp else None
            data: Dict[str] = {}
            if mean_phase is not None:
                data["phase/mean"] = mean_phase
            if mean_rp is not None:
                data["phase/reward_phase_mean"] = mean_rp
            if data:
                wandb.log(data)
            self.buffer_phase.clear()
            self.buffer_rp.clear()

class EpisodeRewardCallback(BaseCallback):
    """Logs each finished episode's reward/length to W&B."""
    def _on_step(self) -> bool:
        if wandb.run is None:
            return True
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep:  # provided by VecMonitor
                wandb.log(
                    {
                        "rollout/ep_rew_mean": ep["r"],
                        "rollout/ep_len_mean": ep["l"],
                    },
                    step=self.num_timesteps,
                )
        return True



# ------------------------------ Agent -------------------------------- #

class InverseAgent(nn.Module):
    def __init__(
        self,
        env: ManagerBasedRLEnv,
        demos: List[dict],
        validation_demos: Optional[List[dict]] = None,
        non_robot_indices_in_obs: Optional[Sequence[int] | slice] = None,
        hyperparams: Optional[dict] = None,
    ):
        super().__init__()
        self.env = env
        self.demos = demos
        self.validation_demos = validation_demos
        self.hyperparams = hyperparams or {}

        if non_robot_indices_in_obs is None:
            self.non_robot_indices_in_obs = []
        elif isinstance(non_robot_indices_in_obs, slice):
            self.non_robot_indices_in_obs = list(range(*non_robot_indices_in_obs.indices(env.observation_space["policy"].shape[-1])))
        elif isinstance(non_robot_indices_in_obs, (list, tuple)):
            self.non_robot_indices_in_obs = list(non_robot_indices_in_obs)
        else:
            raise ValueError(f"Unsupported type for non_robot_indices_in_obs: {type(non_robot_indices_in_obs)}")


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

    def train_phase_evaluator(self, save_best_path: Optional[str] = None, save_latest_path: Optional[str] = None) -> None:
        # infer obs_dim from demos

        sample_obs = self.demos[0]["observations"][0]
        obs_dim_full = sample_obs.shape[-1]
        self.build_phase_evaluator(obs_dim_full)

        training_dataset = PhaseDemoDataset(
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
            patience=self.hyperparams.get("phase_patience", 800),
            num_workers=self.hyperparams.get("num_workers", 0),
            shuffle=True,
        )

        train_phase_evaluator(
            self.phase_evaluator, training_dataset, val_ds, cfg, device=self.device, save_best_path=save_best_path, save_latest_path=save_latest_path
        )
        self.phase_evaluator_trained = True

    # --------------------------- BC Policy -------------------------- #

    def build_bc_policy(self, obs_dim: int, act_dim: int) -> None:
        hidden = self.hyperparams.get("bc_hidden", (256, 256))
        self.bc_policy = GaussianPolicy(obs_dim, act_dim, hidden=hidden).to(self.device)

    def pretrain_bc_policy(self, save_best_path: Optional[str] = None, save_latest_path: Optional[str] = None) -> None:
        sample_obs = self.demos[0]["observations"][0]
        sample_act = self.demos[0]["actions"][0]
        obs_dim_full = sample_obs.shape[-1]
        act_dim = sample_act.shape[-1]

        self.build_bc_policy(obs_dim_full, act_dim)

        train_ds = BCDemoDataset(
            self.demos,
            action_offset=self.hyperparams.get("bc_action_offset", 1),
            reverse_time=self.hyperparams.get("bc_reverse_time", True),
            device=self.device,
            sample_per_traj=self.hyperparams.get("bc_samples_per_traj", None),
        )
        val_ds = None
        if self.validation_demos is not None:
            val_ds = BCDemoDataset(
                self.validation_demos,
                action_offset=self.hyperparams.get("bc_action_offset", 1),
                reverse_time=self.hyperparams.get("bc_reverse_time", True),
                device=self.device,
                sample_per_traj=self.hyperparams.get("bc_val_samples_per_traj", None),
            )

        cfg = BCConfig(
            lr=self.hyperparams.get("bc_lr", 1e-3),
            batch_size=self.hyperparams.get("bc_batch", 256),
            epochs=self.hyperparams.get("bc_epochs", 5000),
            logprob_loss=self.hyperparams.get("bc_logprob_loss", True),
            val_period=self.hyperparams.get("bc_val_period", 100),
            patience=self.hyperparams.get("bc_patience", 500),
            num_workers=self.hyperparams.get("num_workers", 0),
            shuffle=True,
        )

        train_bc_policy(
            self.bc_policy, train_ds, val_ds, cfg, device=self.device, save_best_path=save_best_path, save_latest_path=save_latest_path,
        )
        self.bc_trained = True

    # ------------------------ PPO Finetuning ------------------------ #

    def _make_wrapped_env(self, reward_weight: float = 1.0):
        assert self.phase_evaluator is not None and self.phase_evaluator_trained

        env = self.env

        wrapped = Sb3VecEnvWrapper(env)

        wrapped = PhaseRewardVecWrapper(
            wrapped,
            self.phase_evaluator,
            non_robot_indices=self.non_robot_indices_in_obs,
            reward_weight=reward_weight,
            device=self.device,
            obs_is_dict_key=self.hyperparams.get("obs_is_dict_key", None),
        )

        wrapped = VecMonitor(wrapped)  # record episode stats

        return wrapped

    def _init_ppo(self, env: gym.Env) -> PPO:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device=self.device,
            n_steps=self.hyperparams.get("ppo_n_steps", 256),
            batch_size=self.hyperparams.get("ppo_batch_size", 256),
            learning_rate=self.hyperparams.get("ppo_lr", 3e-4),
            ent_coef=self.hyperparams.get("ppo_ent_coef", 0.0),
            vf_coef=self.hyperparams.get("ppo_vf_coef", 1.0),
            gae_lambda=self.hyperparams.get("ppo_gae_lambda", 0.95),
            gamma=self.hyperparams.get("ppo_gamma", 0.99),
            clip_range=self.hyperparams.get("ppo_clip_range", 0.2),
            tensorboard_log=self.hyperparams.get("ppo_tensorboard_log", None)
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

    def finetune_with_ppo(
            self,
            total_timesteps: int,
            reward_weight: float = 1.0,
            model_dir: str = None,
            log_dir: Optional[str] = None,
            extra_callbacks: Optional[List[BaseCallback]] = None,
            run: Optional[wandb.sdk.wandb_run.Run] = None,
    ):
        venv = self._make_wrapped_env(reward_weight=reward_weight)
        model = self._init_ppo(venv)
        # wandb.watch(model.policy, log="all", log_freq=10)

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)

        # --- SB3 callbacks you already had ---
        save_freq = 10
        report_freq = 10
        patience = 20000
        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_dir)
        stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=patience, verbose=1)
        eval_callback = EvalCallback(
            venv,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=report_freq,
            callback_after_eval=stop_callback
        )
        callback = [checkpoint_callback, eval_callback] + (extra_callbacks if extra_callbacks else [])

        print(f"Starting PPO finetuning for {total_timesteps} timesteps")
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
        if run is not None:
            run.finish()

        self.inverse_model = model

    # ----------------------------- IO ------------------------------- #

    def save_phase_evaluator(self, path: str):
        assert self.phase_evaluator is not None
        torch.save(self.phase_evaluator.state_dict(), path)

    def load_phase_evaluator(self, path: str, obs_dim: int):
        self.build_phase_evaluator(obs_dim)
        self.phase_evaluator.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.phase_evaluator.to(self.device).eval()
        self.phase_evaluator_trained = True

    def save_bc_policy(self, path: str):
        assert self.bc_policy is not None
        torch.save(self.bc_policy.state_dict(), path)

    def load_bc_policy(self, path: str, obs_dim: int, act_dim: int):
        self.build_bc_policy(obs_dim, act_dim)
        self.bc_policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.bc_policy.to(self.device).eval()
        self.bc_trained = True

    def save_inverse_model(self, path: str):
        assert self.inverse_model is not None
        self.inverse_model.save(path)


# ----------------------------- Script ------------------------------- #
import ast
def py_literal(v):
    # parses "[512, 256]" or "(1024, 512)" etc.
    return ast.literal_eval(v)

if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    app = app_launcher.app
    from InverseAssemblyProject.tasks.manager_based.assembly_task.disassembled_start_cfg import DisassembledStartEnvCfg
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
    from isaaclab.envs import ManagerBasedRLEnv
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--reward_weight", type=float, default=1.0)

    # phase evaluator
    parser.add_argument("--phase_hidden", type=py_literal, default=[256, 256])
    parser.add_argument("--phase_lr", type=float, default=1e-3)
    parser.add_argument("--phase_batch", type=int, default=256)
    parser.add_argument("--phase_epochs", type=int, default=10_000)
    parser.add_argument("--phase_tv_weight", type=float, default=0.10)
    parser.add_argument("--phase_val_period", type=int, default=50)
    parser.add_argument("--phase_patience", type=int, default=800)
    parser.add_argument("--phase_load_path", type=str, default=None)
    parser.add_argument("--phase_obs_dim", type=int, default=None)  # optional; prefer inferring

    # BC policy
    parser.add_argument("--bc_hidden", type=py_literal, default=[256, 256])
    parser.add_argument("--bc_lr", type=float, default=1e-3)
    parser.add_argument("--bc_batch", type=int, default=256)
    parser.add_argument("--bc_epochs", type=int, default=50_000)
    parser.add_argument("--bc_logprob_loss", type=str, default="True")  # parse later to bool if you use it
    parser.add_argument("--bc_action_offset", type=int, default=5)
    parser.add_argument("--bc_reverse_time", type=str, default="True")  # parse later to bool
    parser.add_argument("--bc_val_period", type=int, default=50)
    parser.add_argument("--bc_patience", type=int, default=500)
    parser.add_argument("--bc_load_path", type=str, default=None)
    parser.add_argument("--bc_obs_dim", type=int, default=None)  # optional; prefer inferring
    parser.add_argument("--bc_act_dim", type=int, default=None)  # optional; prefer inferring

    # PPO
    parser.add_argument("--ppo_n_steps", type=int, default=256)
    parser.add_argument("--ppo_batch_size", type=int, default=256)
    parser.add_argument("--ppo_lr", type=float, default=3e-4)
    parser.add_argument("--ppo_ent_coef", type=float, default=0.0)
    parser.add_argument("--ppo_vf_coef", type=float, default=1.0)
    parser.add_argument("--ppo_gamma", type=float, default=0.99)
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_clip_range", type=float, default=0.2)

    # misc training controls
    parser.add_argument("--non_robot_start", type=int, default=8)  # to build slice(8, None)
    parser.add_argument("--ckpt_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=20)

    # toggles (W&B sends "True"/"False"); we'll convert after parsing
    parser.add_argument("--train_phase", type=str, default="True")
    parser.add_argument("--train_bc", type=str, default="True")
    parser.add_argument("--train_ppo", type=str, default="True")
    args = parser.parse_args()

    # 1) Create env (Manager-Based RL)
    cfg = DisassembledStartEnvCfg()
    env = ManagerBasedRLEnv(cfg)

    # 2) Load demos
    demo_path = "./datasets/only_pullout_15.hdf5"
    demos = load_demos_from_hdf5(demo_path)
    print(f"Loaded {len(demos)} demos from {os.path.abspath(demo_path)}")

    val_demos_path = "./datasets/only_pullout_validation_5.hdf5"
    val_demos = load_demos_from_hdf5(val_demos_path)
    print(f"Loaded {len(val_demos)} validation demos from {os.path.abspath(val_demos_path)}")


    # 3) Hyperparameters

    # first 8 dimensions are robot joints (including gripper), rest are non-robot
    non_robot_indices = slice(8, None) # or list of indices, e.g. [8, 9, 10, ...]
    default_hparams = dict(
        # phase evaluator
        phase_hidden=(256, 256),
        phase_lr=1e-3,
        phase_batch=256,
        phase_epochs=10_000,
        phase_tv_weight=0.10,
        phase_val_period=50,
        phase_patience=800,

        # bc
        bc_hidden=(256, 256),
        bc_lr=1e-3,
        bc_batch=256,
        bc_epochs=50_000,
        bc_logprob_loss=True,
        bc_action_offset=5,
        bc_reverse_time=True,
        bc_val_period = 50,
        bc_patience = 500,

        # ppo
        ppo_n_steps=256,
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

    model_dir = "./models"
    log_dir = "./models/logs"
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name

    agent = InverseAgent(
        env=env,
        demos=demos,
        validation_demos=val_demos,
        non_robot_indices_in_obs=non_robot_indices,
        hyperparams=default_hparams,
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    os.makedirs(f"models/{timestamp}", exist_ok=False)
    tb_dir = os.path.join("models", timestamp, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    # make PPO write TensorBoard scalars here
    default_hparams["tensorboard_log"] = tb_dir

    # 4) Train the Phase Evaluator (Evaluator)

    agent.train_phase_evaluator(save_best_path=f"models/{timestamp}/phase_evaluator_best.pth", save_latest_path=f"models/{timestamp}/phase_evaluator_latest.pth")
    # agent.load_phase_evaluator(f"models/2025-08-28-17:45/phase_evaluator_best.pth", obs_dim=9999999)

    # 5) Pretrain BC policy
    agent.pretrain_bc_policy(save_best_path=f"models/{timestamp}/bc_policy_best.pth", save_latest_path=f"models/{timestamp}/bc_policy_latest.pth")
    # agent.load_bc_policy(f"models/2025-08-27-20:07/bc_policy_best.pth", obs_dim=25, act_dim=7)

    # 6) PPO Finetune with phase-shaped rewards
    agent.finetune_with_ppo(
        total_timesteps=args.total_timesteps,
        reward_weight=args.reward_weight,
        model_dir=f"models/{timestamp}",
        log_dir=f"models/{timestamp}/logs",
    )
    agent.save_inverse_model(f"models/{timestamp}/final_inverse_model.zip")