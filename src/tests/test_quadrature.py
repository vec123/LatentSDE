import math

import torch

from ..quadrature import GaussLegendreQuad, StratifiedSampling, UnbiasedGaussLegendreQuad, gauss_legendre_vecs
from ._reference import run_tests, svise_module, use_float64

use_float64()


def test_gauss_legendre_matches_svise_fastgl():
    """Same nodes and weights in the same order (ascending up to 100 nodes, descending above)."""
    for n in (5, 26, 100, 101, 102, 128):
        w_ref, x_ref = svise_module("quadrature").gauss_legendre_vecs(n, 0.0, 20.0)
        w, x = gauss_legendre_vecs(n, 0.0, 20.0)
        torch.testing.assert_close(x, x_ref, rtol=0, atol=1e-12)
        torch.testing.assert_close(w, w_ref, rtol=0, atol=1e-12)


def test_gauss_legendre_is_exact_for_polynomials():
    quad = GaussLegendreQuad(0.0, 2.0, 6)
    torch.testing.assert_close(quad(lambda t: t**11), torch.tensor(2.0**12 / 12))


def test_unbiased_quadrature_is_unbiased():
    """Mean of many estimates of int_0^20 sin(t) + t^2 / 50 dt, with too few nodes for plain Gauss-Legendre."""
    def f(t):
        return torch.sin(3 * t) + t**2 / 50

    exact = (1 - math.cos(60)) / 3 + 20**3 / 150
    quad = UnbiasedGaussLegendreQuad(0.0, 20.0, 16, quad_percent=0.5)
    torch.manual_seed(0)
    estimates = torch.stack([quad(f) for _ in range(4000)])
    assert abs(estimates.mean() - exact) < 4 * estimates.std() / math.sqrt(len(estimates))


def test_single_call_quadrature_matches_two_calls():
    """For an f without random numbers, one call at all times gives the same estimate as two calls."""
    calls = []

    def f(t):
        calls.append(len(t))
        return torch.sin(3 * t) + t**2 / 50

    estimates = []
    for single_call in (False, True):
        torch.manual_seed(0)
        estimates.append(UnbiasedGaussLegendreQuad(0.0, 20.0, 16, quad_percent=0.5, single_call=single_call)(f))
    torch.testing.assert_close(estimates[1], estimates[0])
    assert calls == [8, 8, 16]


def test_stratified_sampling_estimates_integral():
    torch.manual_seed(1)
    quad = StratifiedSampling(0.0, 1.0, 2000)
    assert abs(quad(lambda t: t**2) - 1 / 3) < 1e-3


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
