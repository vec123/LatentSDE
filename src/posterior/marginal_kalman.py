"""
Kalman marginals for a prior with any diffusion G(z, mu, t). KalmanPosterior (kalman.py) needs a constant diagonal
G: its q is the smoother's own linear-Gaussian process, and the KL of the paths is finite only when q has the
prior's noise. Here q has the prior's noise G(z) and the smoother's marginals N(m(t), S(t)). The smoother still reads
the current drift and diffusion, so q follows them at once (no EM-like loop of a network posterior).

Filter: statistical linearization of the prior over each gap. Under a Gaussian N(m, S) of z, the noise becomes
D = E[G G^T] and a nonlinear drift F z + b, F = Cov(f, z) S^-1, b = E[f] - F m (a linear drift stays exact, with b
from u at the start of the gap, as in KalmanPosterior). The expectations use the 2 d cubature points
m +- sqrt(d) (columns of chol S) with equal weights: exact up to degree 3 (a quadratic drift, an affine G). The
Gaussians: in the first pass the prior N(z0_mean, exp(z0_logvar)) at every step, then the smoothed marginals of the
previous pass at each step, detached (n_passes passes in all). A gap takes the mean of the statistics at its two
ends. With the linearization fixed, each pass is the parallel scan of KalmanPosterior, with transitions per gap.

Drift: in a gap, the linearized smoother has the drift f_h = F z + b + D (eta_b - Lambda_b z) and the probability
flow v = f_h + D S^-1 (z - m) / 2. The drift with the noise G(z) and the same flow, so the same marginals
(Fokker-Planck), is

    f_q = f_h + (D - Sigma(z)) S^-1 (z - m) / 2 + div Sigma(z) / 2,   Sigma(z) = G(z) G(z)^T   (marginal_drift).

With a constant G, Sigma = D and f_q is KalmanPosterior.exact_drift. Any linearization keeps the ELBO a bound; the
residual then measures the linearization error and Sigma(z) - D over the width of q. No closed form: LatentSDE.loss
uses Monte Carlo, with marginal_drift in the residual.
"""
import math

import torch
from torch import Tensor

from ..latent.diffusion import cov_mv
from .kalman import KalmanPosterior, _mv, _pick, _scan_filters, _sym, transition


def _drift_and_log_density(z: Tensor, F: Tensor, b: Tensor, D: Tensor, Lam_b: Tensor, eta_b: Tensor, m: Tensor,
                           L: Tensor, G: Tensor, div: Tensor):
    """
    At z (..., B, Q, d): f_q = F z + b + D (eta_b - Lambda_b z) + (D - Sigma(z)) S^-1 (z - m) / 2 + div Sigma(z) / 2
    (see the module docstring) and log N(z; m, L L^T), from one query's quantities (B, Q, ...).
    """
    f_h = _mv(F, z) + b + _mv(D, eta_b - _mv(Lam_b, z))
    L = L.expand(torch.broadcast_shapes(z.shape[:-1], L.shape[:-2]) + L.shape[-2:])
    w = torch.linalg.solve_triangular(L, (z - m).unsqueeze(-1), upper=False)
    score = torch.linalg.solve_triangular(L.transpose(-1, -2), w, upper=True).squeeze(-1)  # S^-1 (z - m)
    log_density = (-0.5 * w.squeeze(-1).pow(2).sum(-1) - L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
                   - 0.5 * z.shape[-1] * math.log(2 * math.pi))
    return f_h + 0.5 * (_mv(D, score) - cov_mv(G, score) + div), log_density


def _cubature(m: Tensor, S: Tensor, jitter: float) -> Tensor:
    """The 2 d cubature points m +- sqrt(d) (columns of chol S) of N(m, S): (..., 2 d, d), equal weights."""
    d = m.shape[-1]
    I = torch.eye(d, dtype=m.dtype, device=m.device)
    cols = math.sqrt(d) * torch.linalg.cholesky(S + jitter * I).transpose(-1, -2)  # row i: sqrt(d) L[:, i]
    return m.unsqueeze(-2) + torch.cat([cols, -cols], -2)


class MarginalKalmanPosterior(KalmanPosterior):
    """
    KalmanPosterior for any diffusion and drift (see the module docstring). condition(batch) runs n_passes
    linearized smoothers; forward(t) gives m, L and None, None; marginal_drift(z, steps, G, div) gives the drift of q
    with the prior's noise. extended: True linearizes a linear drift too (for tests); None keeps it exact.
    """

    exact_drift = None  # LatentSDE.loss uses marginal_drift, with the prior's G(z)

    def __init__(self, dim_z: int, potentials=None, n_passes: int = 2, extended: bool = None, jitter: float = 1e-10,
                 dtype: str = "float64"):
        super().__init__(dim_z, potentials, extended, jitter, dtype)
        self.n_passes = n_passes

    def closed_form_terms(self):
        return None

    def _first_linearization(self, m0: Tensor, P0: Tensor, B: int, N: int):
        """The Gaussians of the first pass at each step (B, N, ...): the prior at the first step."""
        return m0.expand(B, N, -1), P0.expand(B, N, -1, -1)

    def _statistics(self, m: Tensor, S: Tensor, mu: Tensor, t: Tensor, u: Tensor, linear):
        """F, b of the drift and D = E[G G^T] under N(m, S) at each step (B, N, ...), in the compute dtype."""
        model, d, dtype = self._model(), self.dim_z, mu.dtype
        B, N = m.shape[:2]
        points = _cubature(m, S, self.jitter)  # (B, N, 2 d, d)
        n = points.shape[-2]
        p, mu_p = points.to(dtype), mu[:, None, None].expand(B, N, n, -1)
        G = model.diffusion(p, mu_p, t.unsqueeze(-1)).to(m.dtype)
        D = (torch.diag_embed(G.pow(2)) if G.dim() == p.dim() else G @ G.transpose(-1, -2)).mean(-3)
        if linear is not None:
            Wz, Wmu_u, bias = linear
            P = mu.shape[-1]
            b = bias + mu.to(m.dtype).unsqueeze(1) @ Wmu_u[:, :P].T + u.to(m.dtype) @ Wmu_u[:, P:].T
            return Wz.expand(B, N, d, d), b, D
        f = model.drift(p, mu_p, u[:, :, None].expand(-1, -1, n, -1)).to(m.dtype)
        Ef = f.mean(-2)
        C = torch.einsum("...pi,...pj->...ij", f - Ef.unsqueeze(-2), points - m.unsqueeze(-2)) / n  # Cov(f, z)
        I = torch.eye(d, dtype=m.dtype, device=m.device)
        F = torch.linalg.solve(S + self.jitter * I, C.transpose(-1, -2)).transpose(-1, -2)  # C S^-1, S symmetric
        return F, Ef - _mv(F, m), D

    def condition(self, batch: dict) -> Tensor:
        """Runs the n_passes linearized smoothers over the batch's steps; returns (B, N), True where observed."""
        model, d = self._model(), self.dim_z
        observed = batch["mask"].any(-1) > 0
        if not observed[:, 0].all():
            raise ValueError("MarginalKalmanPosterior needs an observation at the first step of each trajectory")
        cd = lambda v: v.to(self._dtype())
        t, x, mask, mu, u = batch["t"], batch["x"], batch["mask"], batch["mu"], batch["u"]
        if self.potentials is None:
            lam, eta = self._decoder_potentials(cd(x), cd(mask))
        else:
            lam, eta = (cd(v) for v in self.potentials(x, mask, mu))
        steps = cd(t)
        B, N = steps.shape
        I = torch.eye(d, dtype=steps.dtype, device=steps.device)
        m0, P0 = cd(model.z0_mean), torch.diag_embed(cd(model.z0_logvar).exp())
        gaps = steps[:, 1:] - steps[:, :-1]
        linear = self._linear_drift()
        mean = lambda v: 0.5 * (v[:, :-1] + v[:, 1:])  # a gap: the mean of its two ends
        m_lin, S_lin = self._first_linearization(m0.detach(), P0.detach(), B, N)
        n_passes = max(1, self.n_passes)
        for p in range(n_passes):
            F, b, D = self._statistics(m_lin, S_lin, mu, t, u, linear)
            F_gap, D_gap = mean(F), mean(D)
            b_gap = b[:, :-1] if linear is not None else mean(b)
            Phi, c, Q = transition(F_gap, b_gap, D_gap, gaps)
            mf, Pf, Lplus, eplus = _scan_filters(lam, eta, m0, P0, Phi, c, Q)
            if p == n_passes - 1:
                break
            # smoothed marginals at the steps: the prediction times the backward message of the steps k..N-1
            pred_m = torch.cat([m0.expand(B, 1, d), _mv(Phi, mf[:, :-1]) + c], 1)
            pred_P = torch.cat([P0.expand(B, 1, d, d), _sym(Phi @ Pf[:, :-1] @ Phi.transpose(-1, -2)) + Q], 1)
            sol = torch.linalg.solve(I + pred_P @ Lplus, torch.cat(
                [pred_P, (pred_m + _mv(pred_P, eplus)).unsqueeze(-1)], -1))
            m_lin, S_lin = sol[..., d].detach(), _sym(sol[..., :d]).detach()

        # index k in 0..N: the interval (t_{k-1}, t_k]; the first and the last step's statistics at the two ends
        ends = lambda v, gap: torch.cat([v[:, :1], gap, v[:, -1:]], 1)
        self.lin_F, self.lin_b, self.lin_D = ends(F, F_gap), ends(b, b_gap), ends(D, D_gap)
        self.lam, self.eta, self.pred_m, self.pred_P = lam, eta, None, None
        self.t, self.shared_F = steps, None
        self.fwd_m = torch.cat([m0.expand(B, 1, d), mf], 1)
        self.fwd_P = torch.cat([P0.expand(B, 1, d, d), Pf], 1)
        self.fwd_t = torch.cat([steps[:, :1], steps], 1)
        self.bwd_L = torch.cat([Lplus, Lplus.new_zeros(B, 1, d, d)], 1)
        self.bwd_e = torch.cat([eplus, eplus.new_zeros(B, 1, d)], 1)
        return observed

    def forward(self, t: Tensor):
        """m (B, T, dim_z), L (B, T, dim_z, dim_z) and None, None at the times t (B, T); keeps what marginal_drift
        needs there. Times before the first step count as the first step."""
        dtype, d, steps = t.dtype, self.dim_z, self.t
        tq = torch.maximum(t.to(steps.dtype), steps[:, :1])
        k = torch.searchsorted(steps.contiguous(), tq.contiguous())  # first step at or after t, N after the last
        F, b, D = _pick(self.lin_F, k), _pick(self.lin_b, k), _pick(self.lin_D, k)
        N = steps.shape[1]
        tau_b = torch.where(k < N, steps.gather(1, k.clamp(max=N - 1)) - tq, torch.zeros_like(tq))
        (A, Ab), (c, cb), (Q, Qb) = transition(F, b, D, torch.stack([tq - _pick(self.fwd_t, k), tau_b]))
        m_p = _mv(A, _pick(self.fwd_m, k)) + c
        P_p = A @ _pick(self.fwd_P, k) @ A.transpose(-1, -2) + Q
        Lp, ep = _pick(self.bwd_L, k), _pick(self.bwd_e, k)
        I = torch.eye(d, dtype=tq.dtype, device=tq.device)
        X = torch.linalg.solve(I + Lp @ Qb, torch.cat([Lp @ Ab, (ep - _mv(Lp, cb)).unsqueeze(-1)], -1))
        Lam_b, eta_b = _sym(Ab.transpose(-1, -2) @ X[..., :d]), _mv(Ab.transpose(-1, -2), X[..., d])
        sol = torch.linalg.solve(I + P_p @ Lam_b, torch.cat([P_p, (m_p + _mv(P_p, eta_b)).unsqueeze(-1)], -1))
        m, Sigma = sol[..., d], _sym(sol[..., :d])
        L = torch.linalg.cholesky(Sigma + self.jitter * I)
        self.q_F, self.q_b, self.q_D, self.Lam_b, self.eta_b, self.q_m, self.q_L = (
            v.to(dtype) for v in (F, b, D, Lam_b, eta_b, m, L))
        return m.to(dtype), L.to(dtype), None, None

    def marginal_drift(self, z: Tensor, steps: slice, G: Tensor, div: Tensor) -> Tensor:
        """
        f_q = F z + b + D (eta_b - Lambda_b z) + (D - Sigma(z)) S^-1 (z - m) / 2 + div Sigma(z) / 2 at the query times
        `steps` of the last forward call; z (..., B, len(steps), dim_z), G and div the prior's diffusion at z.
        """
        query = (v[:, steps] for v in (self.q_F, self.q_b, self.q_D, self.Lam_b, self.eta_b, self.q_m, self.q_L))
        return _drift_and_log_density(z, *query, G, div)[0]
