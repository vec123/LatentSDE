"""
Euler-Maruyama integration, for predictions only (training is simulation-free).

EulerMaruyama: adapted from SVISE, reference/svise/svise/utils.py (GPL-3.0).
sdeint_euler replaces the torchsde.sdeint(method="euler") calls of SVISE and ROM-VISDE.
"""
import math
from typing import Callable, Tuple, Union

import torch
from torch import Tensor

from .linalg import sample_covariance


class EulerMaruyama:
    """
    Euler-Maruyama paths of dx = f(t, x) dt + L dbeta with E[dbeta dbeta^T] = Q dt, computed on construction.

    Args:
        f (Callable): (t, (bs, d)) -> (bs, d) drift
        L (Union[Callable, Tensor]): (t, (bs, d)) -> (bs, d, S), or a constant (d, S) matrix
        Q (Tensor): (S, S) diffusion matrix of the Brownian motion
        t_span (Tuple): integration window
        x0 (Tensor): (bs, d) initial states
        dt (float): time step

    Attributes:
        x (Tensor): (bs, n_steps, d) paths; t (Tensor): time grid; mean, var: statistics over the batch if bs > 1
    """

    def __init__(self, f: Callable, L: Union[Callable, Tensor], Q: Tensor, t_span: Tuple, x0: Tensor, dt: float):
        bs = x0.shape[0]
        Q_chol = torch.linalg.cholesky(Q) * math.sqrt(dt)
        t = torch.arange(t_span[0], t_span[1] + dt, dt)
        xk = x0.clone()
        x_list = [xk]
        for tk in t[:-1]:
            dbeta = torch.randn(bs, Q_chol.shape[0]) @ Q_chol.t()
            xk = xk + f(tk, xk) * dt
            if callable(L):
                noise_samples = torch.einsum("ijk,ik->ij", L(tk, xk), dbeta)
            else:
                noise_samples = dbeta @ L.t()
            xk += noise_samples
            x_list.append(xk.clone())
        self.x = torch.stack(x_list, dim=1)
        if x0.shape[0] > 1:
            self.mean = self.x.mean(0)
            self.var = sample_covariance(self.x.transpose(1, 0))
        self.t = t


def sdeint_euler(f: Callable, g: Callable, x0: Tensor, ts: Tensor, noise_type: str = "diagonal") -> Tensor:
    """
    Euler-Maruyama path of dx = f(t, x) dt + g(t, x) dW on the time grid ts, as torchsde.sdeint(method="euler").

    Args:
        f: (t, x (bs, d)) -> (bs, d) drift, t a scalar tensor
        g: (t, x) -> (bs, d) diagonal diffusion ("diagonal") or (bs, d, m) matrix ("general")
        x0 (Tensor): (bs, d) initial states
        ts (Tensor): (n,) increasing times

    Returns:
        Tensor: (n, bs, d) states, as torchsde returns them
    """
    xs = [x0]
    x = x0
    for t0, t1 in zip(ts[:-1], ts[1:]):
        dt = t1 - t0
        gx = g(t0, x)
        if noise_type == "diagonal":
            noise = gx * torch.randn_like(x) * dt.sqrt()
        elif noise_type == "general":
            dW = torch.randn(x.shape[0], gx.shape[-1], dtype=x.dtype, device=x.device) * dt.sqrt()
            noise = (gx @ dW.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError(f"Unknown noise_type '{noise_type}', choose 'diagonal' or 'general'")
        x = x + f(t0, x) * dt + noise
        xs.append(x)
    return torch.stack(xs, dim=0)
