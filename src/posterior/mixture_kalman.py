"""
A mixture of K structured posteriors (MarginalKalmanPosterior) with constant, equal weights: q(z(t)) =
sum_k pi_k N(m_k(t), S_k(t)). Each component is a Kalman smoother of the current prior, so q still follows the drift
at once, but q is not Gaussian: it can have K modes. Example: in simple_multimodal_test_nonlinear, a trajectory
with A only sees the increments of x, so y2^2 but not the sign of y2; the true posterior of y2 has two modes, and
one Gaussian (with E[d f_x / d y2] = 6 E[y2] = 0) learns nothing about y2 from A, a large variational gap that
biased the learned y2^2 term (simple_multimodal_nonlinear_drift_20261018-110755: 1.2 for 3).

The components differ only in the start of their linearization: the first pass of component k linearizes under
N(m0 + offsets[k], P0) at every step (offsets, (K, dim_z), e.g. +-0.3 in y2), and the next passes under the
component's own smoothed marginals. A component linearized at y2 > 0 has d f_x / d y2 > 0, so in it the A values
inform y2. The K smoothers run as one batch of K B rows.

The drift: each component k is a process with the prior's noise G(z), drift f_k (marginal_drift) and marginals N_k.
The Fokker-Planck equation is linear in the density, so with constant weights the drift that keeps the mixture
marginals is

    f_q(z, t) = sum_k r_k(z, t) f_k(z, t),   r_k = pi_k N_k(z) / sum_j pi_j N_j(z)   (mixture_drift),

exact for any G(z). LatentSDE.loss (_mixture_loss) takes every expectation as sum_k pi_k E_{N_k}[.], with
n_samples reparametrized samples of each component (no discrete sampling), and KL(q(z(t0)) || p(z0)) by Monte
Carlo. forward(t) gives the moment-matched Gaussian of the mixture, for the evaluation tools.
"""
import math
from typing import Sequence

import torch
from torch import Tensor

from .marginal_kalman import MarginalKalmanPosterior, _drift_and_log_density


def _log_normal(z: Tensor, m: Tensor, L: Tensor) -> Tensor:
    """log N(z; m, L L^T), z (..., B, d) against m (B, d), L (B, d, d)."""
    L = L.expand(torch.broadcast_shapes(z.shape[:-1], L.shape[:-2]) + L.shape[-2:])
    w = torch.linalg.solve_triangular(L, (z - m).unsqueeze(-1), upper=False).squeeze(-1)
    return (-0.5 * w.pow(2).sum(-1) - L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            - 0.5 * z.shape[-1] * math.log(2 * math.pi))


class MixtureKalmanPosterior(MarginalKalmanPosterior):
    """
    K MarginalKalmanPosterior smoothers with equal weights (see the module docstring). condition(batch) runs them;
    components(t) gives m (K, B, T, d), L (K, B, T, d, d); mixture_drift(z, steps, G, div) the drift of q at samples
    z (S, K, B, Q, d) of the components; mixture_log_density(z, column) log q at one query column of the last
    components call; forward(t) the moment-matched Gaussian m, L and None, None.
    """

    def __init__(self, dim_z: int, offsets: Sequence[Sequence[float]], potentials=None, n_passes: int = 3,
                 extended: bool = None, jitter: float = 1e-10, dtype: str = "float64"):
        super().__init__(dim_z, potentials, n_passes, extended, jitter, dtype)
        offsets = torch.as_tensor(offsets, dtype=torch.get_default_dtype())
        if offsets.dim() != 2 or offsets.shape[1] != dim_z:
            raise ValueError(f"offsets needs shape (K, {dim_z}), got {tuple(offsets.shape)}")
        self.register_buffer("offsets", offsets)
        self.n_components = len(offsets)

    def log_weights(self, like: Tensor) -> Tensor:
        """log pi_k (K,): equal weights."""
        return like.new_full((self.n_components,), -math.log(self.n_components))

    def _first_linearization(self, m0: Tensor, P0: Tensor, B: int, N: int):
        """Component k (rows k B..(k+1) B) linearizes first under N(m0 + offsets[k], P0) at every step."""
        shift = self.offsets.to(m0.dtype).repeat_interleave(self.n_batch, 0)  # (K B, d)
        return (m0 + shift)[:, None].expand(B, N, -1), P0.expand(B, N, -1, -1)

    def condition(self, batch: dict) -> Tensor:
        """Runs the K smoothers as one batch of K B rows; returns (B, N), True where observed."""
        self.n_batch = batch["t"].shape[0]
        stacked = {k: torch.cat([v] * self.n_components, 0) for k, v in batch.items()}
        return super().condition(stacked)[:self.n_batch]

    def components(self, t: Tensor):
        """m (K, B, T, d) and L (K, B, T, d, d) of every component at the times t (B, T)."""
        K, B = self.n_components, self.n_batch
        m, L, _, _ = super().forward(t.repeat(K, 1))
        self.mix_m, self.mix_L = m.view(K, B, *m.shape[1:]), L.view(K, B, *L.shape[1:])
        return self.mix_m, self.mix_L

    def forward(self, t: Tensor):
        """The moment-matched Gaussian of the mixture at t (B, T): m, L and None, None (the loss uses components)."""
        m, L = self.components(t)
        pi = self.log_weights(m).exp().view(-1, 1, 1, 1)
        mean = (pi * m).sum(0)
        dm = m - mean
        cov = (pi.unsqueeze(-1) * (L @ L.transpose(-1, -2) + dm.unsqueeze(-1) * dm.unsqueeze(-2))).sum(0)
        I = torch.eye(m.shape[-1], dtype=m.dtype, device=m.device)
        return mean, torch.linalg.cholesky(cov + self.jitter * I), None, None

    def _component_query(self, k: int, steps):
        rows = slice(k * self.n_batch, (k + 1) * self.n_batch)
        return [v[rows, steps] for v in (self.q_F, self.q_b, self.q_D, self.Lam_b, self.eta_b, self.q_m, self.q_L)]

    def mixture_drift(self, z: Tensor, steps: slice, G: Tensor, div: Tensor) -> Tensor:
        """sum_k r_k(z) f_k(z) at the query times `steps` of the last components call; z (..., B, len(steps), d), G
        and div the prior's diffusion at z."""
        drifts, logs = zip(*(_drift_and_log_density(z, *self._component_query(k, steps), G, div)
                             for k in range(self.n_components)))
        log_pi = self.log_weights(z).view(-1, *[1] * logs[0].dim())
        r = torch.softmax(torch.stack(logs) + log_pi, 0)  # responsibilities (K, ..., B, Q)
        return (r.unsqueeze(-1) * torch.stack(drifts)).sum(0)

    def mixture_log_density(self, z: Tensor, column: int) -> Tensor:
        """log q(z) at one query column of the last components call; z (..., B, d)."""
        logs = torch.stack([_log_normal(z, self.mix_m[k, :, column], self.mix_L[k, :, column])
                            for k in range(self.n_components)])
        return torch.logsumexp(logs + self.log_weights(z).view(-1, *[1] * (logs.dim() - 1)), 0)
