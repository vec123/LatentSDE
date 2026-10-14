"""
Interpolation.

BarycentricInterpolate: adapted from SVISE, reference/svise/svise/quadrature/_barycentric_interp.py (GPL-3.0).
linterp: adapted from ROM-VISDE, reference/rom-visde/visde/utils.py (GPL-3.0); vectorised, same results.
"""
import torch
from torch import Tensor
import torch.nn as nn


class BarycentricInterpolate(nn.Module):
    """Polynomial interpolant through (x, f), based on the scipy implementation."""

    def __init__(self, x: Tensor, f: Tensor) -> None:
        super().__init__()
        self.n = len(x)
        self.dtype = x.dtype
        assert self.n == len(f), "Interpolation requires x and f to be the same length."
        self.register_buffer("xi", x)
        self.register_buffer("fi", f)
        self.register_buffer("wi", torch.zeros(self.n, dtype=self.dtype, device=x.device))
        # scipy implementation for computing weights
        self.wi[0] = 1
        for j in range(1, self.n):
            self.wi[:j] *= self.xi[j] - self.xi[:j]
            self.wi[j] = (self.xi[:j] - self.xi[j]).prod()
        self.wi = self.wi.pow(-1)

    def forward(self, x: Tensor) -> Tensor:
        c = x.unsqueeze(1) - self.xi.unsqueeze(0)
        z = c == 0
        c[z] = 1.0
        c = c.pow(-1)
        r = torch.nonzero(z, as_tuple=True)
        numer = c @ (self.fi * self.wi)
        denom = c @ self.wi
        out = numer / denom
        out[r[0]] = self.fi[r[1]]
        return out


def linterp(y_batch: Tensor, t_batch: Tensor, t_query: Tensor) -> Tensor:
    """
    Piecewise linear interpolation of y_batch (n, dim_y) at sorted times t_batch (n,), evaluated at t_query (q,).
    Outside [t_batch[0], t_batch[-1]] the end values are held.

    Returns:
        Tensor: (q, dim_y)
    """
    n = t_batch.shape[0]
    idx = torch.searchsorted(t_batch, t_query)
    hi = idx.clamp(1, n - 1)
    t0, t1 = t_batch[hi - 1], t_batch[hi]
    weight = ((t_query - t0) / (t1 - t0)).unsqueeze(-1)
    y = y_batch[hi - 1] + (y_batch[hi] - y_batch[hi - 1]) * weight
    y = torch.where((idx == 0).unsqueeze(-1), y_batch[0], y)
    return torch.where((idx == n).unsqueeze(-1), y_batch[-1], y)
