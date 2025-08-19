from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
app = app_launcher.app

from InverseAssemblyProject.tasks.manager_based.assembled_start.assembled_start_cfg import AssembledStartEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO

class InverseAgent(nn.Module):
    def __init__(
            self,
            env: ManagerBasedRLEnv,
            dataset: torch.utils.data.Dataset,
            validation_dataset: torch.utils.data.Dataset = None,
            non_robot_indices_in_obs: list = [],
            hyperparams: dict = {},
    ):
        super(InverseAgent, self).__init__()
        self.env = env
        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.non_robot_indices_in_obs = non_robot_indices_in_obs
        self.hyperparams = hyperparams

        self.phase_evaluator = None
        self.phase_evaluator_trained = False
        self.pretrained_policy = None

        self.inverse_model: PPO | None = None

        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams.get("learning_rate", 0.001))