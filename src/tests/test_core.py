import numpy as np
import torch

from ..core.interpolation import BarycentricInterpolate, linterp
from ..core.linalg import SkewSymMatrixExp, solve_least_squares, solve_lyapunov_diag, solve_lyapunov_spectral
from ..core.nn import FCNN, check_nn_dims
from ..core.sdeint import sdeint_euler
from ._reference import run_tests, use_float64, visde_module

use_float64()


def _random_spd(d, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    return A @ A.T + d * torch.eye(d)


def test_lyapunov_spectral_solves_equation():
    K = _random_spd(4, 0)
    D, Q = torch.linalg.eigh(K)
    RHS = _random_spd(4, 1)
    X = solve_lyapunov_spectral(D, Q, RHS)
    torch.testing.assert_close(X @ K.T + K @ X.T, RHS)


def test_lyapunov_diag_solves_equation():
    K, RHS = torch.rand(5) + 0.5, torch.randn(5)
    X = solve_lyapunov_diag(K, RHS)
    torch.testing.assert_close(2 * X * K, RHS)


def test_least_squares_matches_lstsq():
    A, B = torch.randn(30, 6), torch.randn(30, 2)
    torch.testing.assert_close(solve_least_squares(A, B, gamma=1e-12), torch.linalg.lstsq(A, B).solution)


def test_skew_sym_exp_matches_matrix_exp_and_its_gradient():
    d = 4
    expm = SkewSymMatrixExp(d)
    v = torch.randn(d * (d - 1) // 2, requires_grad=True)

    def skew(v):
        S = torch.zeros(d, d, dtype=v.dtype)
        i, j = torch.triu_indices(d, d, offset=1)
        S = S.index_put((i, j), -v).index_put((j, i), v)
        return S

    R = expm(v)
    torch.testing.assert_close(R, torch.linalg.matrix_exp(skew(v)))
    torch.testing.assert_close(R @ R.T, torch.eye(d))
    w = torch.randn(d, d)
    g_fast, = torch.autograd.grad((expm(v) * w).sum(), v)
    g_ref, = torch.autograd.grad((torch.linalg.matrix_exp(skew(v)) * w).sum(), v)
    torch.testing.assert_close(g_fast, g_ref)


def test_linterp_matches_numpy_and_rom_visde():
    t = torch.tensor([0.0, 0.5, 1.5, 2.0])
    y = torch.randn(4, 3)
    tq = torch.tensor([-1.0, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
    out = linterp(y, t, tq)
    expected = np.stack([np.interp(tq.numpy(), t.numpy(), y[:, k].numpy()) for k in range(3)], axis=-1)
    np.testing.assert_allclose(out.numpy(), expected)
    reference = visde_module("utils").linterp(y, t, tq)
    torch.testing.assert_close(out, reference.to(out.dtype))


def test_barycentric_interpolation_reproduces_polynomial():
    x = torch.linspace(-1, 1, 6)
    interp = BarycentricInterpolate(x, x**5 - 2 * x**2)
    xq = torch.tensor([-0.9, 0.1, 0.33, x[2].item()])
    torch.testing.assert_close(interp(xq), xq**5 - 2 * xq**2)


def test_sdeint_euler_without_noise_is_euler():
    ts = torch.linspace(0, 1, 11)
    xs = sdeint_euler(lambda t, x: -x, lambda t, x: torch.zeros_like(x), torch.ones(2, 1), ts)
    assert xs.shape == (11, 2, 1)
    torch.testing.assert_close(xs[-1], torch.full((2, 1), 0.9**10))


def test_fcnn_shapes_and_dim_check():
    net = FCNN(3, 8, 2, num_layers=2, batch_norm=False)
    assert net(torch.randn(5, 7, 3)).shape == (5, 7, 2)
    check_nn_dims(net, ((3,),), ((2,),))
    try:
        check_nn_dims(net, ((3,),), ((4,),))
    except AssertionError:
        return
    raise AssertionError("check_nn_dims accepted a wrong output shape")


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
