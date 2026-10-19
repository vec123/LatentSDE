"""
A posterior whose mean is the integral of an interpolated rate: for observations that are sparse and irregular
next to the speed of the latent state, where interpolating the state itself (SequencePosterior's context at t,
KernelPosterior's kernel interpolation) gives paths that are too smooth between observations.

A GRU (bidirectional by default) reads [x * mask, mask, t - t_0] at the observation times, as in
SequencePosterior. From its state h_k at every step, a head gives a rate a_k and a log variance b_k, and a
second head gives m_0 from the first step; mu goes into both. Then, with A(t) the linear interpolation of the
a_k (held constant outside the steps) and B(t) that of the b_k,

    m(t) = m_0 + int_{t_0}^t A(s) ds,   dm/dt = A(t),   log v(t) = B(t),   dlog v/dt = B'(t),

all in closed form (the integral of a piecewise linear function), so there are no steps between the
observations, no attention over them, and no forward-mode AD. The cost is O(N) per trajectory to condition and
a sorted search per query time. Padded rows (the last time repeated, mask 0) are handled as interp does
(src/latent/batch.py): the first copy of the last time is the end.

Over a long gap, the state moves by the integral of the rates the network gives inside it, so a quantity that
accumulates (a phase, src/latent/angles.py; a position) comes out of the rates, and the observations at the end
of the gap correct it through the gradient of that integral.
"""
from typing import Tuple

import torch
from torch import Tensor, nn

from ..core.nn import mlp


def integrate_linear(t: Tensor, a: Tensor, t_query: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    For the linear interpolation A of a (B, N, F) at sorted times t (B, N), held constant outside them, at t_query
    (B, Q) in any order: int_{t[:, 0]}^{t_query} A, A(t_query) and A'(t_query) (0 outside the steps), each
    (B, Q, F).
    """
    t = t.contiguous()
    i = torch.minimum(torch.searchsorted(t, t_query.contiguous()), torch.searchsorted(t, t[:, -1:])).clamp(min=1)
    t0, t1 = t.gather(1, i - 1), t.gather(1, i)
    seg = (t1 - t0).clamp_min(1e-12).unsqueeze(-1)
    cum = torch.cat([torch.zeros_like(a[:, :1]),
                     torch.cumsum(0.5 * (a[:, 1:] + a[:, :-1]) * (t[:, 1:] - t[:, :-1]).unsqueeze(-1), 1)], 1)
    idx = (i - 1).unsqueeze(-1).expand(-1, -1, a.shape[-1])
    a0, a1, c0 = a.gather(1, idx), a.gather(1, idx + 1), cum.gather(1, idx)
    delta = (t_query - t0).unsqueeze(-1)
    inside = torch.minimum(delta.clamp(min=0), seg)
    before, after = delta.clamp(max=0), (t_query - t1).clamp(min=0).unsqueeze(-1)
    slope = (a1 - a0) / seg
    integral = c0 + a0 * inside + 0.5 * slope * inside ** 2 + a0 * before + a1 * after
    within = (delta >= 0) & (after == 0)
    return integral, a0 + slope * inside, torch.where(within, slope, torch.zeros_like(slope))


class IntegratedPosterior(nn.Module):
    """q(z(t)) = N(m(t), diag v(t)) with m the integral of a rate interpolated between the observation times."""

    def __init__(self, dim_x: int, dim_z: int, dim_mu: int = 0, hidden: int = 64, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.GRU(2 * dim_x + 1, hidden, batch_first=True, bidirectional=bidirectional)
        width = (2 if bidirectional else 1) * hidden
        self.rate_head = mlp(width + dim_mu, 2 * dim_z, (hidden, hidden))
        self.start_head = mlp(width + dim_mu, dim_z, (hidden,))

    def condition(self, batch: dict) -> Tensor:
        """Runs the GRU and the heads; returns (B, N), True at the steps with an observed value."""
        t, mask, mu = batch["t"], batch["mask"], batch["mu"]
        h, _ = self.gru(torch.cat([batch["x"] * mask, mask, (t - t[:, :1]).unsqueeze(-1)], -1))
        self.t_obs = t
        self.rate, self.logvar = self.rate_head(torch.cat([h, mu.unsqueeze(1).expand(-1, t.shape[1], -1)], -1)
                                                ).chunk(2, -1)
        self.m0 = self.start_head(torch.cat([h[:, 0], mu], -1))
        return mask.any(-1)

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """m, log v, dm/dt, dlog v/dt at t (B, Q), each (B, Q, dim_z)."""
        integral, dm, _ = integrate_linear(self.t_obs, self.rate, t)
        _, logv, dlogv = integrate_linear(self.t_obs, self.logvar, t)
        return self.m0.unsqueeze(1) + integral, logv, dm, dlogv
