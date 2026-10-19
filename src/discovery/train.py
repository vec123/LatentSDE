"""
Training loop of SVISE's equation discovery: Adam with its own learning rate for the SDE parameters, beta from 0 to 1
over transition_iters steps, learning-rate decay every transition_iters / 2 steps, and random subsets of the data.

Adapted from SVISE, reference/svise/experiments/4_symbolic_goveq_discovery/experiment_util.py (GPL-3.0). The loop
keeps SVISE's DataLoader, so a run uses the random numbers in the same order as SVISE. Changes: callbacks
(step, sde, loss) after each step, e.g. Checkpoint, History and Log from src/core/callbacks.py, and the time
warpings skip their bound checks during the loop (src/kernels/warping.py, fixed_bounds).
"""
from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from ..core.callbacks import Checkpoint, History, Log
from ..kernels.warping import fixed_bounds
from .learner import SDELearner

__all__ = ["fit", "Checkpoint", "History", "Log"]


def fit(sde: SDELearner, t: Tensor, y: Tensor, n_iters: int = 20000, transition_iters: int = 5000, lr: float = 1e-3,
        lr_sde: float = 1e-2, lr_decay: float = 0.9, batch_size: int = 128,
        callbacks: Iterable[Callable[[int, SDELearner, float], None]] = ()) -> None:
    """Trains sde on the observations y (N, num_meas) at the times t (N,); steps are counted from 1."""
    num_data = len(t)
    num_mc_samples = int(min(batch_size, num_data))
    scheduler_freq = transition_iters // 2
    optimizer = torch.optim.Adam([{"params": sde.state_params()}, {"params": sde.sde_params(), "lr": lr_sde}], lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)
    train_loader = DataLoader(TensorDataset(t, y), batch_size=num_mc_samples, shuffle=True)
    sde.train()
    # every time in the loop is an observation time or lies in the quadrature interval
    span = torch.tensor([float(sde.quad_scheme.a), float(sde.quad_scheme.b)], dtype=t.dtype, device=t.device)
    j = 0
    with fixed_bounds(sde, torch.cat([t, span])):
        for _ in range(n_iters // len(train_loader)):
            for _ in train_loader:
                j += 1
                optimizer.zero_grad()
                idx = np.random.choice(np.arange(num_data), num_mc_samples, replace=False)
                beta = min(1.0, j / transition_iters)
                if j % scheduler_freq == 0:
                    scheduler.step()
                loss = -sde.elbo(t[idx], y[idx], beta, num_data)
                loss.backward()
                optimizer.step()
                for callback in callbacks:
                    callback(j, sde, loss.item())
