"""
Loading the reaction-diffusion trajectories (test_data/scripts/reaction_diffusion) into LatentSDE batches.

A field u (n_traj, n_steps, n, n) is cut into non-overlapping windows of a fixed length, pooling the windows of
every trajectory into one axis (window_trajectories); the pool is then split into batch dicts of equal size
(make_batches), e.g. one dict of all the windows, or several smaller ones to cycle through with latent.train.fit.

A dataset may also have a per-trajectory parameter (e.g. generate_reaction_diffusion_batch.py's --beta-min/
--beta-max, stored as a "beta" dataset): load_reaction_diffusion's param reads it as mu, and window_trajectories
and make_batches carry it through, constant over every window cut from its trajectory, to make_batch's mu.
"""
from typing import List, Optional, Tuple

import h5py
import torch
from torch import Tensor

from ..latent.batch import make_batch


def load_reaction_diffusion(path: str, field: str = "u", param: Optional[str] = None
                            ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    t (n_steps,), field (n_traj, n_steps, n, n) or (n_steps, n, n) for the single-trajectory dataset, and mu
    (n_traj, 1) from the per-trajectory dataset named param, or None if param is not given.
    """
    dtype = torch.get_default_dtype()
    with h5py.File(path, "r") as f:
        t = torch.as_tensor(f["t"][:], dtype=dtype)
        x = torch.as_tensor(f[field][:], dtype=dtype)
        mu = torch.as_tensor(f[param][:], dtype=dtype).unsqueeze(-1) if param else None
    return t, x, mu


def window_trajectories(t: Tensor, x: Tensor, length: int, stride: Optional[int] = None, mu: Optional[Tensor] = None
                        ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Cuts every trajectory of x (n_traj, n_steps, *shape) into windows of `length` steps (stride `length` by
    default, i.e. non-overlapping), flattens *shape, and pools the windows of all trajectories into one axis.

    Args:
        t (Tensor): (n_steps,), shared by every trajectory
        x (Tensor): (n_traj, n_steps, *shape)
        length (int): window length, in steps
        stride (int, optional): step between window starts; `length` (non-overlapping) by default
        mu (Tensor, optional): (n_traj, dim_mu), one parameter vector per trajectory

    Returns:
        Tensor: t_windows (K, length)
        Tensor: x_windows (K, length, prod(shape)), K = n_traj * number of windows per trajectory
        Tensor or None: mu_windows (K, dim_mu), each trajectory's mu repeated over its windows; None if mu is None
    """
    stride = length if stride is None else stride
    n_traj, n_steps = x.shape[:2]
    n_windows = (n_steps - length) // stride + 1
    if n_windows < 1:
        raise ValueError(f"length {length} is longer than n_steps {n_steps}")
    starts = torch.arange(n_windows) * stride
    t_windows = torch.stack([t[s:s + length] for s in starts]).repeat(n_traj, 1)
    x = x.flatten(2)
    x_windows = torch.stack([x[:, s:s + length] for s in starts], dim=1).reshape(n_traj * n_windows, length, -1)
    mu_windows = None
    if mu is not None:
        mu_windows = mu.unsqueeze(1).expand(-1, n_windows, -1).reshape(n_traj * n_windows, -1)
    return t_windows, x_windows, mu_windows


def make_batches(t_windows: Tensor, x_windows: Tensor, n_groups: int, mu_windows: Optional[Tensor] = None
                 ) -> List[dict]:
    """
    Splits the K pooled windows into n_groups batch dicts of K // n_groups windows each (make_batch, no forcing).
    K must be a multiple of n_groups; n_groups=1 gives a single batch of every window, n_groups=K one window per
    batch. mu_windows (K, dim_mu), from window_trajectories, becomes each batch's mu; omitted, mu stays empty.
    """
    K = t_windows.shape[0]
    if K % n_groups:
        raise ValueError(f"{K} windows do not split evenly into {n_groups} groups")
    size = K // n_groups
    return [make_batch(t_windows[i * size:(i + 1) * size], x_windows[i * size:(i + 1) * size],
                       mu=mu_windows[i * size:(i + 1) * size] if mu_windows is not None else None)
            for i in range(n_groups)]


def load_batches(path: str, field: str, length: int, n_groups: int = 1, stride: Optional[int] = None,
                 param: Optional[str] = None) -> List[dict]:
    """load_reaction_diffusion, window_trajectories and make_batches in one call."""
    t, x, mu = load_reaction_diffusion(path, field, param)
    t_windows, x_windows, mu_windows = window_trajectories(t, x, length, stride, mu)
    return make_batches(t_windows, x_windows, n_groups, mu_windows)
