import numpy as np
import torch

from ..discovery import marginal_sde
from ..discovery.diffusion import ConstantDiagonalDiffusionPrior, SparseDiagonalDiffusionPrior
from ..discovery.marginal_sde import SpectralMarginalSDE
from ..discovery.marginals import MeanFCNN, MeanGLM, OrthogonalGLM, StrictlyPositiveGLM
from ._reference import run_tests, svise_module, use_float64

use_float64()
d = 2
T_SPAN = (0.0, 5.0)
T_OBS = torch.linspace(0.0, 5.0, 40)
Y_OBS = torch.stack([torch.sin(T_OBS), torch.cos(1.3 * T_OBS)], dim=-1)


def _seeded(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build(module, marginal_class_name, diffusion_class, **kwargs):
    diffusion = diffusion_class(d, torch.ones(d), 6, 1e-5)
    marginal_class = getattr(module, marginal_class_name)
    return marginal_class(d, T_SPAN, diffusion_prior=diffusion, model_form="GLM", n_tau=50,
                          learn_inducing_locations=False, train_x=T_OBS, train_y=Y_OBS, **kwargs)


def _prior_drift(t, z):
    return torch.stack([z[..., 1], -z[..., 0] - 0.1 * z[..., 1] ** 3], dim=-1)


def test_marginal_sdes_match_svise():
    svise = svise_module("sde_learning")
    t = torch.linspace(0.1, 4.9, 9)
    for name in ("SpectralMarginalSDE", "DiagonalMarginalSDE"):
        _seeded()
        reference = _build(svise, name, svise.SparseDiagonalDiffusionPrior)
        _seeded()
        ours = _build(marginal_sde, name, SparseDiagonalDiffusionPrior)
        torch.testing.assert_close(ours.mean(t, return_grad=True), reference.mean(t, return_grad=True))
        torch.testing.assert_close(ours.K(t), reference.K(t))
        _seeded(1)
        expected = reference(t, _prior_drift, 6)
        _seeded(1)
        torch.testing.assert_close(ours(t, _prior_drift, 6), expected)


def _check_time_derivative(model, t):
    t = t.clone().requires_grad_(True)
    out, dout = model(t, return_grad=True)
    weights = torch.randn_like(out)
    expected, = torch.autograd.grad((out * weights).sum(), t)
    dims = tuple(range(1, out.dim()))
    torch.testing.assert_close((dout * weights).sum(dims), expected)


def test_glm_time_derivatives_with_learned_warping():
    t = torch.linspace(0.2, 4.8, 11)
    for model in (MeanGLM(d, T_SPAN, 30), StrictlyPositiveGLM(d, T_SPAN, 30), OrthogonalGLM(3, T_SPAN, 30)):
        with torch.no_grad():
            model.K.input_warping.rawalpha.fill_(0.1)
            model.K.input_warping.rawbeta.fill_(0.9)
            model.raw_w.normal_()
        _check_time_derivative(model, t)


def test_fcnn_mean_time_derivative():
    torch.manual_seed(0)
    _check_time_derivative(MeanFCNN(d, T_SPAN, 16, 2), torch.linspace(0.2, 4.8, 11))


def test_spectral_gauges_preserve_marginals():
    t = torch.linspace(0.3, 4.7, 7)
    for gauge in ("sym", "sqrt"):
        _seeded()
        sde = SpectralMarginalSDE(d, T_SPAN, ConstantDiagonalDiffusionPrior(d, torch.tensor([0.4, 1.5])), n_tau=50,
                                  train_x=T_OBS, train_y=Y_OBS, gauge=gauge)
        U, dU = sde.orthogonal(t, return_grad=True)
        D, dD = sde.eigenvals(t, return_grad=True)
        S, dS = sde.K(t), sde._dKdt_fast(U, dU, D, dD)
        m = sde.mean(t)
        f_m = sde.drift(t, m)
        A = torch.stack([sde.drift(t, m + torch.eye(d)[i]) - f_m for i in range(d)], dim=-1)
        Q = torch.diag(torch.tensor([0.4, 1.5]))
        torch.testing.assert_close(A @ S + S @ A.transpose(-2, -1) + Q, dS)


def test_sparse_diffusion_prior_parameterizations():
    for parameterization in ("noncentered", "centered"):
        prior = SparseDiagonalDiffusionPrior(3, torch.ones(3), 5, 1e-5, parameterization=parameterization)
        assert prior.process_noise.shape == (5, 3, 3)
        assert torch.isfinite(prior.kl_divergence)


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
