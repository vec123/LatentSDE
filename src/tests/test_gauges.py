import torch

from ..posterior import dKdt_spectral, sqrt_drift_spectral, sym_drift_diagonal, sym_drift_spectral
from ._reference import run_tests, use_float64

use_float64()
d, bs = 3, 4


def _marginals(t):
    """A smooth family of means m(t), rotations U(t) and eigenvalues D(t), for a batch of times t (bs,)."""
    k = torch.arange(1, d * (d - 1) // 2 + 1, dtype=t.dtype)
    i, j = torch.triu_indices(d, d, offset=1)
    basis = torch.zeros(len(k), d, d, dtype=t.dtype)
    basis[torch.arange(len(k)), j, i] = 1.0
    basis = basis - basis.transpose(-2, -1)
    U = torch.linalg.matrix_exp(torch.einsum("bk,kij->bij", torch.sin(t.unsqueeze(-1) * k), basis))
    D = 0.5 + torch.exp(torch.cos(t.unsqueeze(-1) * torch.arange(1, d + 1, dtype=t.dtype)))
    m = torch.sin(2 * t.unsqueeze(-1) + torch.arange(d, dtype=t.dtype))
    return m, U, D


def _with_derivatives(t):
    (m, U, D), (dm, dU, dD) = torch.func.jvp(_marginals, (t,), (torch.ones_like(t),))
    return m, dm, U, dU, D, dD


def _linear_part(drift, m):
    """A of an affine drift f(z) = f(m) + A (z - m), column by column."""
    f_m = drift(m)
    return torch.stack([drift(m + torch.eye(d)[i]) - f_m for i in range(d)], dim=-1)


def _assert_marginals_preserved(gauge, Q):
    t = torch.linspace(0.1, 2.0, bs)
    m, dm, U, dU, D, dD = _with_derivatives(t)
    S = U @ torch.diag_embed(D) @ U.transpose(-2, -1)
    A = _linear_part(lambda z: gauge(z, m, dm, U, dU, D, dD, Q), m)
    torch.testing.assert_close(A @ S + S @ A.transpose(-2, -1) + Q, dKdt_spectral(U, dU, D, dD))
    torch.testing.assert_close(gauge(m, m, dm, U, dU, D, dD, Q), dm)
    return A


def test_dKdt_spectral_matches_autograd():
    t = torch.linspace(0.1, 2.0, bs)
    _, _, U, dU, D, dD = _with_derivatives(t)
    _, dS = torch.func.jvp(lambda t_: (lambda m, U, D: U @ torch.diag_embed(D) @ U.transpose(-2, -1))(*_marginals(t_)),
                           (t,), (torch.ones_like(t),))
    torch.testing.assert_close(dKdt_spectral(U, dU, D, dD), dS)


def test_sym_gauge_preserves_marginals_and_is_symmetric():
    A = _assert_marginals_preserved(sym_drift_spectral, torch.diag(torch.tensor([0.3, 1.0, 2.0])))
    torch.testing.assert_close(A, A.transpose(-2, -1))


def test_sqrt_gauge_preserves_marginals():
    _assert_marginals_preserved(sqrt_drift_spectral, torch.diag(torch.tensor([0.3, 1.0, 2.0])))


def test_gauges_agree_for_diagonal_covariance():
    t = torch.linspace(0.1, 2.0, bs)
    m, dm, _, _, D, dD = _with_derivatives(t)
    U, dU = torch.eye(d).expand(bs, d, d), torch.zeros(bs, d, d)
    Q_diag = torch.tensor([0.3, 1.0, 2.0])
    z = m + torch.randn(5, bs, d)
    expected = sym_drift_diagonal(z, m, dm, D, dD, Q_diag)
    torch.testing.assert_close(sym_drift_spectral(z, m, dm, U, dU, D, dD, torch.diag(Q_diag)), expected)
    torch.testing.assert_close(sqrt_drift_spectral(z, m, dm, U, dU, D, dD, torch.diag(Q_diag)), expected)


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
