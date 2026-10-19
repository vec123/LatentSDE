"""MixtureKalmanPosterior (src/posterior/mixture_kalman.py) and LatentSDE._mixture_loss."""
import io
import math

import numpy as np
import torch
from torch import nn

from ..latent import ConstantDiagonalDiffusion, GaussianDecoder, LatentSDE, fit, make_batch
from ..posterior.kalman import GaussianPotentials
from ..posterior.marginal_kalman import MarginalKalmanPosterior
from ..posterior.mixture_kalman import MixtureKalmanPosterior
from ._reference import run_tests, use_float64
from .test_marginal_kalman import D, Z, _affine, _batch, _model, _query_times

use_float64()

OFFSETS = [[0.0, 0.0, 0.4], [0.0, 0.0, -0.4]]


def _gauss_hermite(m, L, n=9):
    """Nodes (n^Z, B, T, Z) and weights (n^Z,) of N(m, L L^T): a product Gauss-Hermite rule."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    grid = torch.cartesian_prod(*[torch.tensor(x)] * Z)
    weights = torch.cartesian_prod(*[torch.tensor(w / w.sum())] * Z).prod(-1)
    return m + torch.einsum("btij,kj->kbti", L, grid), weights


class _QuadraticDrift(nn.Module):
    """dx = (y1 + 3 (y2^2 - 0.09)) dt, dy_j = -y_j / 5 dt: symmetric under y2 -> -y2."""

    def forward(self, z, mu, u):
        x, y1, y2 = z.unbind(-1)
        return torch.stack([y1 + 3 * (y2 ** 2 - 0.09), -y1 / 5, -y2 / 5], -1)


def test_identical_components_give_the_single_posterior():
    batch, out = _batch(), []
    single = MarginalKalmanPosterior(Z, n_passes=3)
    mixture = MixtureKalmanPosterior(Z, [[0.0] * Z] * 2, n_passes=3)
    for posterior in (single, mixture):
        model = _model(posterior, _affine())
        posterior.condition(batch)
        out.append(posterior(_query_times(batch))[:2])
    torch.testing.assert_close(out[0], out[1])
    m, L = mixture.components(_query_times(batch))
    for k in range(2):
        torch.testing.assert_close(m[k], out[0][0])
    single(_query_times(batch))
    z = out[0][0] + 0.1 * torch.randn(4, *out[0][0].shape)
    G, div = model.diffusion.with_divergence(z, None, None)
    torch.testing.assert_close(mixture.mixture_drift(z, slice(None), G, div), single.marginal_drift(z, slice(None), G, div))
    log_q = mixture.mixture_log_density(z[:, :, -1], -1)
    expected = torch.distributions.MultivariateNormal(out[0][0][:, -1], scale_tril=out[0][1][:, -1]).log_prob(z[:, :, -1])
    torch.testing.assert_close(log_q, expected)


def test_mixture_drift_keeps_the_mixture_marginals():
    """
    With two different components and a state-dependent G, dz = f_q dt + G(z) dW keeps the mixture marginals:
    d/dt E[z] = E[f_q], d/dt E[z z^T] = E[f_q z^T + z f_q^T + G G^T], expectations sum_k pi_k E_{N_k} by product
    Gauss-Hermite rules, time derivatives by central differences of the components' m(t), S(t).
    """
    batch = _batch()
    posterior = MixtureKalmanPosterior(Z, OFFSETS, n_passes=3)
    model = _model(posterior, _affine(), hidden=(8,))  # nonlinear drift: the components differ
    posterior.condition(batch)
    t = batch["t"]
    tq, h = (0.5 * (t[:, :-1] + t[:, 1:]))[:, :5], 1e-5

    def moments(times):
        m, L = posterior.components(times)
        second = L @ L.transpose(-1, -2) + m.unsqueeze(-1) * m.unsqueeze(-2)
        return m.mean(0), second.mean(0)  # equal weights

    (m_plus, M_plus), (m_minus, M_minus) = moments(tq + h), moments(tq - h)
    m, L = posterior.components(tq)  # mixture_drift reads the last call
    assert (m[0] - m[1]).abs().max() > 1e-2  # the components differ
    first, second = 0.0, 0.0
    for k in range(2):
        nodes, weights = _gauss_hermite(m[k], L[k])
        G, div = model.diffusion.with_divergence(nodes, None, None)
        f = posterior.mixture_drift(nodes, slice(None), G, div)
        E = lambda v: torch.einsum("n,n...->...", weights, v) / 2
        first = first + E(f)
        second = second + E(f.unsqueeze(-1) * nodes.unsqueeze(-2) + nodes.unsqueeze(-1) * f.unsqueeze(-2)
                            + G @ G.transpose(-1, -2))
    torch.testing.assert_close(first, (m_plus - m_minus) / (2 * h), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(second, (M_plus - M_minus) / (2 * h), rtol=1e-5, atol=1e-5)


def test_mirrored_offsets_give_mirrored_components_when_only_x_is_seen():
    """
    A drift symmetric under y2 -> -y2 and observations of x only: the components with offsets +-0.4 in y2 are
    mirror images (y2 -> -y2), and the offsets break the symmetry (y2 away from 0), where one Gaussian keeps y2 at 0.
    """
    torch.manual_seed(3)
    dt, n = 0.01, 2000
    y = torch.zeros(n, 2)
    for i in range(1, n):
        y[i] = y[i - 1] - y[i - 1] / 5 * dt + 0.19 * math.sqrt(dt) * torch.randn(2)
    xs = torch.cumsum((y[:, 0] + 3 * (y[:, 1] ** 2 - 0.09)) * dt, 0)
    idx = torch.arange(0, n, 50)
    t = (idx * dt)[None]
    x = torch.zeros(1, len(idx), D)
    x[0, :, 0] = xs[idx] + 0.05 * torch.randn(len(idx))
    mask = torch.zeros_like(x)
    mask[..., 0] = 1.0
    batch = make_batch(t, x, mask)
    means = []
    for posterior in (MixtureKalmanPosterior(Z, OFFSETS, n_passes=4), MarginalKalmanPosterior(Z, n_passes=4)):
        decoder = GaussianDecoder(Z, D, std_fixed=[0.05, 0.05, 0.05])
        with torch.no_grad():
            decoder.mean[0].weight.copy_(torch.eye(D))
            decoder.mean[0].bias.zero_()
        model = LatentSDE(posterior, _QuadraticDrift(), decoder, Z,
                          diffusion=ConstantDiagonalDiffusion(Z, 0.1, fixed=[0.2, 0.19, 0.19]))
        with torch.no_grad():
            model.z0_logvar.copy_(2 * torch.tensor([1.0, 0.3, 0.3]).log())
        posterior.condition(batch)
        times = torch.linspace(0.5, 19.5, 40)[None]
        means.append(posterior.components(times)[0] if hasattr(posterior, "components") else posterior(times)[0])
    m = means[0]
    torch.testing.assert_close(m[1, ..., :2], m[0, ..., :2])
    torch.testing.assert_close(m[1, ..., 2], -m[0, ..., 2])
    assert m[0, ..., 2].abs().mean() > 0.05
    assert means[1][..., 2].abs().max() < 1e-6


def _terms_over_seeds(model, batch, n_seeds=8):
    runs = []
    with torch.no_grad():
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            runs.append(model.loss(batch)[1])
    values = {k: torch.stack([r[k] for r in runs]) for k in runs[0]}
    return {k: (v.mean().item(), v.std().item() / math.sqrt(n_seeds)) for k, v in values.items()}


def test_identical_components_give_the_same_loss_in_expectation():
    batch, terms = _batch(), []
    for posterior in (MarginalKalmanPosterior(Z), MixtureKalmanPosterior(Z, [[0.0] * Z] * 2)):
        terms.append(_terms_over_seeds(_model(posterior, _affine()), batch))
    for key, (mean, err) in terms[0].items():
        other, other_err = terms[1][key]
        assert abs(mean - other) < 4 * math.hypot(err, other_err) + 1e-8, (key, mean, other, err, other_err)


def test_gradients_reach_every_part_and_the_model_survives_a_reload():
    batch = _batch()
    posterior = MixtureKalmanPosterior(Z, OFFSETS, GaussianPotentials(D, Z, hidden=(8,)))
    model = _model(posterior, _affine(), hidden=(8,))
    loss, terms = model.loss(batch)
    assert all(math.isfinite(v.item()) for v in terms.values())
    loss.backward()
    pot = posterior.potentials
    for p in (model.drift.net[0].weight, model.diffusion.raw_g, model.diffusion.C, model.z0_mean, model.z0_logvar,
              pot.W, pot.log_s, pot.net[-1].weight, model.decoder.mean[0].weight):
        assert p.grad is not None and p.grad.abs().sum() > 0
    losses = fit(model, batch, n_iter=3, lr=1e-2, n_warmup=0, log_every=0)
    assert all(math.isfinite(v) for v in losses)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    loaded.posterior.condition(batch)
    model.posterior.condition(batch)
    torch.testing.assert_close(loaded.posterior(batch["t"])[0], model.posterior(batch["t"])[0])


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
