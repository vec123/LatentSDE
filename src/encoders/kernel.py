"""
The svise type posterior: a Gaussian process, given by kernel interpolation in time. An encoder maps each window
of num_window observations to a Gaussian node (m_k, v_k) at its first time; a learned kernel interpolates m and
log v in time. Needs evenly spaced, fully observed trajectories: kernels.DeepGaussianKernel fixes one dt and one
batch_size at construction.

Adapted from ROM-VISDE, reference/rom-visde/visde/sdevar.py, AmortizedLatentVarGP (GPL-3.0).
"""
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn

from ..core.nn import mlp


class KernelPosterior(nn.Module):
    """
    kernel(t1 (..., n, 1), t2 (..., m, 1), return_grad) has an interpolation noise kernel.var, e.g.
    DeepGaussianKernel(batch_size=N - num_window + 1, dt=dt) for N observations.
    """

    def __init__(self, dim_x: int, dim_z: int, num_window: int, kernel: nn.Module, dim_mu: int = 0,
                 hidden: Sequence[int] = (128, 128)):
        super().__init__()
        self.num_window = num_window
        self.kernel = kernel
        self.encoder = mlp(num_window * dim_x + dim_mu, 2 * dim_z, hidden)

    def condition(self, batch: dict) -> Tensor:
        """Encodes the windows; returns (B, N), True at the window starts (the times the posterior covers)."""
        t, mu = batch["t"], batch["mu"]
        windows = batch["x"].unfold(1, self.num_window, 1).flatten(-2)  # (B, K, D * num_window)
        K = windows.shape[1]
        nodes = self.encoder(torch.cat([windows, mu.unsqueeze(1).expand(-1, K, -1)], -1))  # [m_k, log v_k]
        self.t0 = t[:, :1]
        self.t_nodes = (t[:, :K] - self.t0).unsqueeze(-1)
        gram = self.kernel(self.t_nodes, self.t_nodes)
        gram = gram + self.kernel.var * torch.eye(K, dtype=t.dtype, device=t.device)
        self.weights = torch.cholesky_solve(nodes, torch.linalg.cholesky(gram))
        return (torch.arange(t.shape[1], device=t.device) < K).expand_as(t)

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """m, log v, dm/dt, dlog v/dt at t (B, Q), each (B, Q, dim_z)."""
        k, dk = self.kernel((t - self.t0).unsqueeze(-1), self.t_nodes, return_grad=True)
        m, logv = (k @ self.weights).chunk(2, -1)
        dm, dlogv = (dk @ self.weights).chunk(2, -1)
        return m, logv, dm, dlogv
