import torch
import torch.nn as nn

from ..kernels import (DeepGaussianKernel, KumaraswamyWarping, Matern12, Matern32, Matern52, NeuralWarping,
                       SquaredExponential)
from ._reference import run_tests, svise_module, use_float64, visde_module

use_float64()

T = torch.linspace(0.0, 2.0, 15).unsqueeze(-1)
TAU = torch.linspace(-0.5, 2.5, 9).unsqueeze(-1)


def _warped_kumaraswamy():
    warp = KumaraswamyWarping((-0.5, 2.5))
    with torch.no_grad():
        warp.rawalpha.fill_(0.2)
        warp.rawbeta.fill_(1.1)
    return warp


def _check_time_derivative(kernel, t=T, tau=TAU):
    """dk/dt from return_grad against autograd: each row of k depends on its own t only."""
    t = t.clone().requires_grad_(True)
    k, dk = kernel(t, tau, return_grad=True)
    torch.testing.assert_close(k, kernel(t, tau))
    weights = torch.randn_like(k)
    expected, = torch.autograd.grad((k * weights).sum(), t)
    torch.testing.assert_close((dk * weights).sum(-1), expected.reshape(-1))


def test_matern_time_derivatives_with_warping():
    for kernel_class in (Matern52, Matern32, Matern12):
        _check_time_derivative(kernel_class(input_warping=_warped_kumaraswamy(), len_init=0.7))


def test_squared_exponential_time_derivative():
    _check_time_derivative(SquaredExponential(input_warping=_warped_kumaraswamy(), len_init=0.4))


def test_deep_kernel_time_derivative():
    for activation in (nn.Tanh, nn.ReLU):
        warping = NeuralWarping.for_window(15, 2.0 / 14, hidden_sizes=(16, 16), activation=activation)
        _check_time_derivative(DeepGaussianKernel(warping=warping))


def test_deep_kernel_gradient_reaches_warping_network():
    kernel = DeepGaussianKernel(batch_size=15, dt=2.0 / 14)
    _, dk = kernel(T, T, return_grad=True)
    dk.pow(2).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in kernel.warping.net.parameters())


def test_identity_init_and_window_rescaling():
    warping = NeuralWarping.for_window(15, 2.0 / 14, identity_init=True)
    u, du = warping(torch.tensor([0.0, 1.0, 2.0]), return_grad=True)
    torch.testing.assert_close(u, torch.tensor([-1.0, 0.0, 1.0]))
    torch.testing.assert_close(du, torch.full((3,), 1.0))


def test_matern52_matches_svise():
    reference = svise_module("kernels").Matern52(len_init=0.7)
    kernel = Matern52(len_init=0.7)
    torch.testing.assert_close(kernel(T, TAU, return_grad=True), reference(T, TAU, return_grad=True))


def test_deep_kernel_matches_rom_visde():
    torch.manual_seed(0)
    reference = visde_module("kernel").DeepGaussianKernel(
        nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)), 15, 2.0 / 14)
    with torch.no_grad():
        reference.rawlens.add_(0.3)
        reference.rawsigma.add_(-0.2)
    kernel = DeepGaussianKernel(batch_size=15, dt=2.0 / 14)
    kernel.warping.net.load_state_dict(reference.net.state_dict())
    kernel.load_state_dict({**kernel.state_dict(), "rawlens": reference.rawlens.detach(),
                            "rawsigma": reference.rawsigma.detach(), "rawvar": reference.rawvar.detach()})
    torch.testing.assert_close(kernel(T, TAU), reference(T, TAU))
    torch.testing.assert_close(kernel.var, reference.var)


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
