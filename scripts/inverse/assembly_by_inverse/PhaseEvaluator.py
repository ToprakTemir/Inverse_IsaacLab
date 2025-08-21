import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class PhaseEvaluator(nn.Module):
    """
    Predicts a normalized phase in [0,1] given an observation (generally with
    robot dimensions removed).
    """
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 256), dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1), nn.Sigmoid()]  # phase in [0,1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns shape (B, 1)
        return self.net(x)


# ------------------------------- Data --------------------------------- #

class PhaseDemoDataset(Dataset):
    """
    Expects a list of demonstrations, each a dict with keys:
      - 'observations': np.ndarray of shape (T, obs_dim)
    Optionally, you can pass non_robot_indices to slice observations.
    Targets are t / (T-1).
    """
    def __init__(
        self,
        demos: List[dict],
        non_robot_indices: Optional[Sequence[int]] = None,
        sample_per_traj: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.samples: List[Tuple[np.ndarray, float]] = []
        self.non_robot_indices = non_robot_indices
        self.device = device

        for demo in demos:
            obs = demo["observations"]  # shape (T, obs_dim)
            T = obs.shape[0]
            if T < 2:
                continue
            if sample_per_traj is None or sample_per_traj >= T:
                idxs = np.arange(T)
            else:
                idxs = np.random.choice(T, size=sample_per_traj, replace=False)

            denom = max(T - 1, 1)
            for t in idxs:
                x = obs[t]
                if non_robot_indices is not None and len(non_robot_indices) > 0:
                    x = x[non_robot_indices]
                y = float(t / denom)
                self.samples.append((x.astype(np.float32), y))

        if len(self.samples) == 0:
            raise ValueError("PhaseDemoDataset constructed with 0 samples. Check your demos input.")

        self.input_dim = self.samples[0][0].shape[0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        xt = torch.from_numpy(x)
        yt = torch.tensor([y], dtype=torch.float32)
        return xt, yt


@dataclass
class PhaseTrainConfig:
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 10_000  # you can also think in steps; this is per-epoch over DataLoader
    tv_weight: float = 0.15  # optional smoothness across batch ordering (weak regularizer)
    val_period: int = 100
    patience: int = 1000  # early stopping patience in validation checks
    num_workers: int = 0
    shuffle: bool = True


# ---------------------------- Training -------------------------------- #

def train_phase_evaluator(
    model: PhaseEvaluator,
    train_dataset: PhaseDemoDataset,
    val_dataset: Optional[PhaseDemoDataset] = None,
    cfg: PhaseTrainConfig = PhaseTrainConfig(),
    device: Optional[torch.device] = None,
    save_best_path: Optional[str] = None,
) -> dict:
    """
    Trains the PhaseEvaluator to predict normalized phase.
    Returns a training history dict.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
        )
    print(f"Training PhaseEvaluator with {train_dataset} dataset")

    optim_ = optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}

    best_val = math.inf
    bad = 0

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb).squeeze(-1)

            # main loss
            loss_mse = mse(pred, yb.squeeze(-1))

            # optional "temporal smoothness" across the CURRENT minibatch ordering
            # (not real temporal order, but works as a small Lipschitz regularizer)
            if cfg.tv_weight > 0 and xb.shape[0] > 1:
                tv = torch.mean(torch.abs(pred[1:] - pred[:-1]))
                loss = loss_mse + cfg.tv_weight * tv
            else:
                loss = loss_mse

            optim_.zero_grad(set_to_none=True)
            loss.backward()
            optim_.step()

            epoch_loss += loss.item()
            step += 1

        epoch_loss /= max(1, len(train_loader))
        history["train_loss"].append(epoch_loss)
        print(f"Epoch {epoch + 1}/{cfg.epochs}, Train Loss: {epoch_loss:.4f}")

        # validation
        if val_loader is None or (epoch % cfg.val_period != 0):
            continue

        if save_best_path is not None:
            print(f"Saving best model to {save_best_path}")


        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb).squeeze(-1)
                vals.append(mse(pred, yb.squeeze(-1)).item())
        v = float(np.mean(vals)) if len(vals) else epoch_loss
        history["val_loss"].append(v)

        improved = v < best_val
        if improved:
            best_val = v
            bad = 0
            if save_best_path is not None:
                torch.save(model.state_dict(), save_best_path)
        else:
            bad += 1
            if bad * cfg.val_period >= cfg.patience:
                # early stop
                break

    return history