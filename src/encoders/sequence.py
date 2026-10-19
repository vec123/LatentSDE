"""
The SDE-matching type posterior: a network of state and time, reparameterized with Gaussian noise (Bartosh et al.
2025). A bidirectional GRU reads observations at any times, with a mask, building a state (its hidden sequence);
an MLP of the attention context at time t and t itself gives m(t) and log v(t). Written from the paper, because
the sde_matching repository has no license.

With gauge=True, a separate head on the last hidden layer of the MLP also gives the gauge generator Omega(t), a
skew-symmetric (dim_z, dim_z) matrix from its dim_z (dim_z - 1) / 2 entries above the diagonal (Gauge Matching,
arXiv 2610.04390, appendix: posterior architecture). The head starts at zero, the fixed gauge of SDE matching.

With full_cov=True, the MLP gives the Cholesky factor L(t) of a full covariance instead of log v(t): the marginals
are N(m, L L^T), so the posterior can correlate the latent dimensions. A state driven by another one with no noise
of its own (dx = y dt) needs this: with diagonal marginals, the posterior drift of x cannot follow the noise of y,
and the drift residual of x grows as its diffusion goes to zero (runs/simple_multimodal_width_300_20261017-131926:
the learned diffusion of x 0.5 for 0, the drive 0.4 for 1). Use it with gauge=True: with Omega = 0, the posterior
drift of the first latent dimension depends on that dimension only.
"""
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from ..core.nn import mlp
from ..core.transforms import inverse_softplus_scalar


class SequencePosterior(nn.Module):
    """
    A GRU (bidirectional by default, or forward-only) reads [x * mask, mask, t - t_0] at the observation times.
    At a time t, the context is the mean of the GRU outputs with weights softmax(-((t - t_k) / width)^2) over
    the observed steps; an MLP of [context, mu, t - t_0] gives m(t) and log v(t), and forward-mode AD their
    time derivatives. The width is learned from its start value, or fixed with learn_width=False: the drift
    residual prefers a smooth posterior, so a learned width can grow to several observation gaps
    (runs/simple_multimodal_20261016-154052: 3.0 for gaps of 1.5), and the posterior then smooths the fast motion.

    With gap_features=True, the MLP also reads log(1 + bridge time) of each input column (bridge_times). The
    context alone changes smoothly over one width, so the MLP cannot see where the observations are, and v(t) came
    out almost flat (runs/simple_multimodal_width_300_20261017-131926: the std of y2 0.083 to 0.125, where the exact
    smoother goes from 0.044 at a B observation to 0.179 between two). The bridge time is the shape of that
    variance: 0 at an observation of the column, the largest in the middle of a gap.

    set_input_scaling standardizes the inputs of the networks: each column of x to (x - x_shift) / x_scale and the
    time t - t_0 to (t - t_0) / t_scale (the attention and the bridge times keep the time units). Without it, the GRU
    reads raw values, e.g. A of simple_multimodal_test (std 3.5, up to about 20) and t - t_0 up to 100, where its
    gates saturate; the posterior mean of x was 0.5 to 1.0 from the exact posterior mean of its own model. It did
    not help (runs/simple_multimodal_standardized_20261017-165954).

    With forward_only=True the posterior is a forward posterior (a filter): m(t) and log v(t) depend only on the
    observations at or before t. The
    GRU is forward-only, and the attention context ignores the steps after t (a forward GRU alone would not do: its
    output at a later step contains the later observations, and the attention averages over all the steps, so the
    posterior at t would still see the future). Not combined with gap_features or neighbor_context, which look at
    the next observation. A filter is a restricted posterior, so the ELBO is looser than that of the smoother.

    With neighbor_context=True, the MLP also reads, for each input column, the GRU outputs at the column's previous
    and next observation (neighbor_context), however far they are: the attention context sees only the steps
    within about one width, so in the middle of a long gap of one modality (A of simple_multimodal_test, 10 time
    units) nothing of that modality's observations reaches the MLP except through the GRU.
    """

    def __init__(self, dim_x: int, dim_z: int, dim_mu: int = 0, hidden: int = 64, width: float = 1.0,
                 bidirectional: bool = True, gauge: bool = False, learn_width: bool = True, full_cov: bool = False,
                 gap_features: bool = False, neighbor_context: bool = False, forward_only: bool = False):
        super().__init__()
        if forward_only and (bidirectional or gap_features or neighbor_context):
            raise ValueError("a forward posterior needs bidirectional=False, gap_features=False, neighbor_context=False")
        self.forward_only = forward_only  # not named forward: that is the method
        self.dim_z, self.full_cov, self.gap_features = dim_z, full_cov, gap_features
        self.use_neighbors = neighbor_context
        self.gru = nn.GRU(2 * dim_x + 1, hidden, batch_first=True, bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        n_scale = dim_z * (dim_z + 1) // 2 if full_cov else dim_z
        dim_in = num_directions * hidden + dim_mu + 1 + (dim_x if gap_features else 0)
        dim_in += 2 * dim_x * num_directions * hidden if neighbor_context else 0
        self.head = mlp(dim_in, dim_z + n_scale, (hidden, hidden))
        if full_cov:
            # entry (i, j) of L from [0, exp(log-diagonal / 2), below the diagonal]: 0 above the diagonal
            tril_map = torch.zeros(dim_z, dim_z, dtype=torch.long)
            tril_map[range(dim_z), range(dim_z)] = torch.arange(1, dim_z + 1)
            below = torch.tril_indices(dim_z, dim_z, -1)
            tril_map[below[0], below[1]] = torch.arange(dim_z + 1, n_scale + 1)
            self.register_buffer("tril_map", tril_map, persistent=False)
        self.register_buffer("x_shift", torch.zeros(dim_x))
        self.register_buffer("x_scale", torch.ones(dim_x))
        self.register_buffer("t_scale", torch.ones(()))
        raw_width = torch.tensor(inverse_softplus_scalar(width))
        if learn_width:
            self.raw_width = nn.Parameter(raw_width)
        else:
            self.register_buffer("raw_width", raw_width)
        self.gauge_head = None
        if gauge:
            self.gauge_head = nn.Linear(hidden, dim_z * (dim_z - 1) // 2)
            nn.init.zeros_(self.gauge_head.weight)
            nn.init.zeros_(self.gauge_head.bias)
            self.register_buffer("upper", torch.triu_indices(dim_z, dim_z, 1), persistent=False)

    def set_input_scaling(self, x_shift: Tensor, x_scale: Tensor, t_scale: float) -> None:
        """The standardization of the network inputs: x_shift, x_scale (dim_x,), e.g. the mean and std of each
        column's observed values in the training data, and t_scale, e.g. the longest span of observation times."""
        self.x_shift.copy_(x_shift)
        self.x_scale.copy_(x_scale)
        self.t_scale.fill_(float(t_scale))

    def _scales(self):
        """x_shift, x_scale, t_scale; no scaling for runs saved before set_input_scaling."""
        if getattr(self, "x_shift", None) is None:
            return 0.0, 1.0, 1.0
        return self.x_shift, self.x_scale, self.t_scale

    def condition(self, batch: dict) -> Tensor:
        """Runs the GRU; returns (B, N), True at the steps with an observed value."""
        t, mask = batch["t"], batch["mask"]
        self.t_obs, self.mu, self.observed = t, batch["mu"], mask.any(-1)
        self.column_observed = mask > 0
        shift, scale, t_scale = self._scales()
        x = (batch["x"] - shift) / scale * mask
        self.h, _ = self.gru(torch.cat([x, mask, ((t - t[:, :1]) / t_scale).unsqueeze(-1)], -1))
        return self.observed

    def _neighbors(self, t: Tensor):
        """
        For each column, at t (B, Q): d- the time since the column's last observation at or before t and d+ the time
        to its first one after t, both capped at the span of the trajectory's observations, so they stay finite
        (B, Q, D); the step indices of those two observations (B, Q, D); and whether they exist (B, Q, D).
        """
        lag = (t.unsqueeze(-1) - self.t_obs.unsqueeze(1)).unsqueeze(-1)  # (B, Q, N, 1)
        seen = self.column_observed.unsqueeze(1)  # (B, 1, N, D)
        observed_t = torch.where(self.observed, self.t_obs, torch.nan)
        span = (observed_t.nan_to_num(-torch.inf).amax(1) - observed_t.nan_to_num(torch.inf).amin(1)).clamp_min(1e-6)
        cap = span.view(-1, 1, 1)
        inf = torch.full_like(lag, torch.inf)
        before, i_prev = torch.where(seen & (lag >= 0), lag, inf).min(2)
        after, i_next = torch.where(seen & (lag < 0), -lag, inf).min(2)
        has_prev, has_next = torch.isfinite(before), torch.isfinite(after)
        return torch.minimum(before, cap), torch.minimum(after, cap), i_prev, i_next, has_prev, has_next

    def bridge_times(self, t: Tensor) -> Tensor:
        """
        For each column, at t (B, Q): d- d+ / (d- + d+), with d- the time since the column's last observation at or
        before t, and d+ the time to its first one after t: (B, Q, D). Both are capped at the span of the
        trajectory's observations, so it is d- after the last observation, d+ before the first one, and span / 2
        for a column with no observation. It is continuous in t (a jump in m or v would cost nothing in the loss,
        which never evaluates it at the jump) and 0 at the column's observations, with slope -1 before them and +1
        after them; d+ counts only later times, so it is never 0 / 0.
        """
        before, after = self._neighbors(t)[:2]
        return before * after / (before + after)

    def neighbor_context(self, t: Tensor) -> Tensor:
        """
        For each column, at t (B, Q), from the GRU outputs h_p and h_n at the column's previous and next observation:
        the interpolation (1 - s) h_p + s h_n and the difference 4 s (1 - s) (h_n - h_p), s = d- / (d- + d+): together
        both ends in the middle of a gap, and continuous across each observation (h_p, h_n change there, where s is
        1 then 0). With one end only (before the first, after the last observation), h of that end and 0; with none,
        0 and 0. (B, Q, 2 D H), H the width of the GRU output.
        """
        before, after, i_prev, i_next, has_prev, has_next = self._neighbors(t)
        B, Q, D = before.shape
        H = self.h.shape[-1]
        pick = lambda i: self.h.gather(1, i.reshape(B, Q * D, 1).expand(-1, -1, H)).view(B, Q, D, H)
        h_prev, h_next = pick(i_prev), pick(i_next)
        s = (before / (before + after)).unsqueeze(-1)
        both = (has_prev & has_next).unsqueeze(-1)
        one = torch.where(has_prev.unsqueeze(-1), h_prev, torch.where(has_next.unsqueeze(-1), h_next, 0.0))
        level = torch.where(both, (1 - s) * h_prev + s * h_next, one)
        change = torch.where(both, 4 * s * (1 - s) * (h_next - h_prev), 0.0)
        return torch.cat([level, change], -1).flatten(2)

    def _head_input(self, t: Tensor) -> Tensor:
        logits = -((t.unsqueeze(-1) - self.t_obs.unsqueeze(1)) / softplus(self.raw_width)).pow(2)  # (B, Q, N)
        allowed = self.observed.unsqueeze(1).expand_as(logits)
        if getattr(self, "forward_only", getattr(self, "causal", False)):  # a forward posterior: no step after t
            allowed = allowed & (self.t_obs.unsqueeze(1) <= t.unsqueeze(-1))
        # a large negative number, not -inf: the gradient of a softmax over a fully masked row is NaN. A time with no allowed
        # step (e.g. before the first observed step the input mask keeps) has no information: its context is zero.
        weights = logits.masked_fill(~allowed, -1e30).softmax(-1) * allowed.any(-1, keepdim=True)
        context = weights @ self.h
        mu = self.mu.unsqueeze(1).expand(-1, t.shape[1], -1)
        parts = [context, mu, ((t - self.t_obs[:, :1]) / self._scales()[2]).unsqueeze(-1)]
        if getattr(self, "gap_features", False):
            parts.append(self.bridge_times(t).log1p())
        if getattr(self, "use_neighbors", False):
            parts.append(self.neighbor_context(t))
        return torch.cat(parts, -1)

    def _mean_logvar(self, t: Tensor):
        return self.head(self._head_input(t)).chunk(2, -1)

    def _mean_cholesky(self, t: Tensor):
        out = self.head(self._head_input(t))
        m, logd, below = out.split([self.dim_z, self.dim_z, out.shape[-1] - 2 * self.dim_z], -1)
        entries = torch.cat([torch.zeros_like(m[..., :1]), (0.5 * logd).exp(), below], -1)
        return m, entries[..., self.tril_map]

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """m, log v, dm/dt, dlog v/dt at t (B, Q), each (B, Q, dim_z); with full_cov, m, L, dm/dt, dL/dt, with L
        (B, Q, dim_z, dim_z) lower triangular with a positive diagonal."""
        f = self._mean_cholesky if getattr(self, "full_cov", False) else self._mean_logvar
        (m, scale), (dm, dscale) = torch.func.jvp(f, (t,), (torch.ones_like(t),))
        return m, scale, dm, dscale

    def gauge(self, t: Tensor) -> Optional[Tensor]:
        """Omega at t (B, Q): (B, Q, dim_z, dim_z), skew-symmetric; None without a gauge head (Omega = 0)."""
        gauge_head = getattr(self, "gauge_head", None)  # runs saved before the gauge head have no attribute
        if gauge_head is None:
            return None
        entries = gauge_head(self.head[:-1](self._head_input(t)))
        d = getattr(self, "dim_z", self.head[-1].out_features // 2)  # runs saved before dim_z: 2 dim_z outputs
        upper = entries.new_zeros(*entries.shape[:-1], d, d)
        upper[..., self.upper[0], self.upper[1]] = entries
        return upper - upper.transpose(-1, -2)
