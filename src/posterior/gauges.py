"""
Drift f(z, t) = dm/dt + A(t) (z - m(t)) of an approximate posterior with Gaussian marginals N(m(t), S(t)) and
diffusion G G^T = Q.

Many drifts have the same marginals (Helmholtz-SDE, Smith et al. 2026); a gauge picks one. Every A of a gauge
satisfies A S + S A^T + Q = dS/dt, so the marginals are those of the posterior.

    sym   A symmetric. SVISE (reference/svise/svise/sde_learning/_marginal_sde.py, GPL-3.0) solves the
          Lyapunov equation; ROM-VISDE's drift residual is the diagonal case.
    sqrt  A = (d/dt S^1/2) S^-1/2 - Q S^-1 / 2, the drift of SDE matching (Bartosh et al. 2025) for an affine
          flow z = m + S^1/2 eps; written from the paper. It also allows a state-dependent diffusion.

For diagonal S both gauges give the same drift.
"""
from torch import Tensor

from ..core.linalg import solve_lyapunov_diag, solve_lyapunov_spectral


def dKdt_spectral(U: Tensor, dUdt: Tensor, D: Tensor, dDdt: Tensor) -> Tensor:
    """
    dS/dt for S = U diag(D) U^T (SVISE's SpectralMarginalSDE._dKdt_fast).

    Args:
        U, dUdt (Tensor): (bs, d, d) orthogonal matrices and their time derivatives
        D, dDdt (Tensor): (bs, d) eigenvalues and their time derivatives
    """
    F1 = (U.mul(D.unsqueeze(-2))) @ dUdt.transpose(-2, -1)
    F2 = dUdt.mul(D.unsqueeze(-2))
    F3 = U.mul(dDdt.unsqueeze(-2))
    return F1 + (F2 + F3) @ U.transpose(-2, -1)


def sym_drift_diagonal(z: Tensor, m: Tensor, dmdt: Tensor, K_diag: Tensor, dKdt_diag: Tensor, Q_diag: Tensor) -> Tensor:
    """
    sym gauge for diagonal S and Q: f = dm - (Q - dS) / (2 S) (z - m), elementwise.

    Args:
        z (Tensor): (..., bs, d) states
        m, dmdt, K_diag, dKdt_diag (Tensor): (bs, d) marginal mean, its derivative, variances, their derivative
        Q_diag (Tensor): (d,), or (n, 1, d) for n diffusion samples
    """
    A = solve_lyapunov_diag(K_diag, -dKdt_diag + Q_diag)
    return dmdt - (A * (z - m))


def sym_drift_spectral(z: Tensor, m: Tensor, dmdt: Tensor, U: Tensor, dUdt: Tensor, D: Tensor, dDdt: Tensor,
                       Q: Tensor) -> Tensor:
    """
    sym gauge for S = U diag(D) U^T: f = dm - X (z - m) with X S + S X = Q - dS/dt.

    Args:
        z (Tensor): (..., bs, d) states
        m, dmdt (Tensor): (bs, d)
        U, dUdt (Tensor): (bs, d, d); D, dDdt (Tensor): (bs, d)
        Q (Tensor): (d, d), or (n, 1, d, d) for n diffusion samples
    """
    dKdt = dKdt_spectral(U, dUdt, D, dDdt)
    X = solve_lyapunov_spectral(D, U, -dKdt + Q)
    return dmdt - (X @ (z - m).unsqueeze(-1)).squeeze(-1)


def sqrt_drift_spectral(z: Tensor, m: Tensor, dmdt: Tensor, U: Tensor, dUdt: Tensor, D: Tensor, dDdt: Tensor,
                        Q: Tensor) -> Tensor:
    """
    sqrt gauge for S = U diag(D) U^T: f = dm + A (z - m) with A = (d/dt S^1/2) S^-1/2 - Q S^-1 / 2.
    Arguments as for sym_drift_spectral.
    """
    Ut = U.transpose(-2, -1)
    sqrt_D = D.sqrt()
    dsqrt_S = (dUdt.mul(sqrt_D.unsqueeze(-2)) @ Ut + U.mul((0.5 * dDdt / sqrt_D).unsqueeze(-2)) @ Ut
               + U.mul(sqrt_D.unsqueeze(-2)) @ dUdt.transpose(-2, -1))
    inv_sqrt_S = U.mul((1.0 / sqrt_D).unsqueeze(-2)) @ Ut
    inv_S = U.mul((1.0 / D).unsqueeze(-2)) @ Ut
    A = dsqrt_S @ inv_sqrt_S - 0.5 * Q @ inv_S
    return dmdt + (A @ (z - m).unsqueeze(-1)).squeeze(-1)


GAUGES = {"sym": sym_drift_spectral, "sqrt": sqrt_drift_spectral}
