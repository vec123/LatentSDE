"""MarginalKalmanPosterior (src/posterior/marginal_kalman.py) and AffineCholeskyDiffusion (src/latent/diffusion.py)."""
import io
import math

import numpy as np
import torch

from ..latent import AffineCholeskyDiffusion, ConstantDiagonalDiffusion, DriftNet, GaussianDecoder, LatentSDE, fit
from ..latent import make_batch
from ..posterior.kalman import GaussianPotentials, KalmanPosterior
from ..posterior.marginal_kalman import MarginalKalmanPosterior
from ._reference import run_tests, use_float64

use_float64()

D, Z = 3, 3


def _batch():
    """Two trajectories at irregular times, some values unobserved; the second row padded (last time repeated)."""
    torch.manual_seed(1)
    t = torch.cumsum(torch.rand(2, 8) * 0.8 + 0.05, 1)
    t[1, 6:] = t[1, 5]
    x = torch.randn(2, 8, D)
    mask = (torch.rand(2, 8, D) < 0.6).double()
    mask[:, 0] = 1.0
    mask[1, 6:] = 0.0
    return make_batch(t, x * mask, mask)


def _affine(scale=0.3):
    """An affine G with random entries below the diagonal: the noise depends on the state."""
    torch.manual_seed(5)
    diffusion = AffineCholeskyDiffusion(Z, 0.4)
    with torch.no_grad():
        diffusion.C0.normal_(0, scale)
        diffusion.C.normal_(0, scale)
    return diffusion


def _model(posterior, diffusion=None, hidden=()):
    torch.manual_seed(2)
    drift = DriftNet(Z, hidden=hidden)
    decoder = GaussianDecoder(Z, D, std_fixed=[0.3, 0.2, 0.4])
    with torch.no_grad():
        decoder.mean[0].weight.copy_(torch.eye(D) + 0.3 * torch.randn(D, Z))
        decoder.mean[0].bias.copy_(0.1 * torch.randn(D))
        drift.net[0].weight.mul_(0.5)
    model = LatentSDE(posterior, drift, decoder, Z, diffusion=diffusion or ConstantDiagonalDiffusion(Z, 0.4),
                      n_samples=4, n_quad=16, closed_form=False)
    with torch.no_grad():
        model.z0_mean.copy_(torch.tensor([0.2, -0.1, 0.3]))
        model.z0_logvar.copy_(torch.tensor([0.1, -0.5, 0.2]))
    return model


def _query_times(batch, n=30):
    torch.manual_seed(0)
    t = batch["t"]
    return torch.sort(t[:, :1] + torch.rand(2, n) * (t[:, -1:] - t[:, :1]), 1).values


def _gauss_hermite(m, L, n=5):
    """Nodes (n^Z, B, T, Z) and weights (n^Z,) of N(m, L L^T): a product Gauss-Hermite rule, exact to degree 2n - 1."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    grid = torch.cartesian_prod(*[torch.tensor(x)] * Z)
    weights = torch.cartesian_prod(*[torch.tensor(w / w.sum())] * Z).prod(-1)
    return m + torch.einsum("btij,kj->kbti", L, grid), weights


def test_affine_diffusion_starts_constant_and_its_divergence_is_exact():
    z = torch.randn(5, Z)
    G, div = AffineCholeskyDiffusion(Z, 0.4).with_divergence(z, z[:, :0], torch.zeros(()))
    torch.testing.assert_close(G, 0.4 * torch.eye(Z).expand(5, Z, Z))
    torch.testing.assert_close(div, torch.zeros(5, Z))
    diffusion = _affine()
    G, div = diffusion.with_divergence(z, z[:, :0], torch.zeros(()))
    assert (G.triu(1) == 0).all() and (G.diagonal(dim1=-2, dim2=-1) > 0).all()
    sigma = lambda zz: (lambda g: g @ g.T)(diffusion(zz, zz[:0], torch.zeros(())))
    for i in range(len(z)):
        J = torch.func.jacfwd(sigma)(z[i])  # J[a, b, c] = d Sigma_ab / d z_c
        torch.testing.assert_close(div[i], torch.einsum("acc->a", J))


def test_constant_noise_gives_the_kalman_posterior():
    """With a constant diagonal G and a linear drift, the marginals and the drift are those of KalmanPosterior."""
    batch, out = _batch(), []
    for posterior in (KalmanPosterior(Z), MarginalKalmanPosterior(Z, n_passes=2)):
        model = _model(posterior)
        posterior.condition(batch)
        m, L, _, _ = posterior(_query_times(batch))
        z = m + 0.1
        if posterior.exact_drift is not None:
            drift = posterior.exact_drift(z, slice(None))
        else:
            drift = posterior.marginal_drift(z, slice(None), *model.diffusion.with_divergence(z, None, None))
        out.append((m, L, drift))
    for a, b in zip(*out):
        torch.testing.assert_close(a, b)


def test_constant_noise_gives_the_same_monte_carlo_loss():
    batch, terms = _batch(), []
    for posterior in (KalmanPosterior(Z), MarginalKalmanPosterior(Z)):
        model = _model(posterior)
        torch.manual_seed(0)
        terms.append(model.loss(batch)[1])
    for key in terms[0]:
        torch.testing.assert_close(terms[0][key], terms[1][key])


def test_statistical_linearization_of_a_linear_drift_is_exact():
    batch, out = _batch(), []
    for extended in (None, True):
        posterior = MarginalKalmanPosterior(Z, n_passes=2, extended=extended)
        _model(posterior, _affine())
        posterior.condition(batch)
        m, L, _, _ = posterior(_query_times(batch))
        out.append((m, L, posterior.q_F, posterior.q_b, posterior.q_D))
    for a, b in zip(*out):
        torch.testing.assert_close(a, b)


def test_marginal_drift_keeps_the_marginals():
    """
    With a state-dependent G, the SDE dz = f_q dt + G(z) dW has the smoother's marginals (Fokker-Planck):
    dm/dt = E[f_q] and dS/dt = E[f_q (z - m)^T + (z - m) f_q^T + G G^T]. Expectations from a product Gauss-Hermite
    rule (exact: f_q is cubic in z), time derivatives from central differences of m(t), S(t) inside the gaps.
    """
    batch = _batch()
    posterior = MarginalKalmanPosterior(Z, n_passes=2)
    model = _model(posterior, _affine())
    posterior.condition(batch)
    t = batch["t"]
    tq, h = (0.5 * (t[:, :-1] + t[:, 1:]))[:, :5], 1e-5  # gap midpoints, away from the padded end

    def moments(times):
        m, L, _, _ = posterior(times)
        return m, L @ L.transpose(-1, -2)

    (m_plus, S_plus), (m_minus, S_minus) = moments(tq + h), moments(tq - h)
    m, L, _, _ = posterior(tq)  # marginal_drift reads the last call
    nodes, weights = _gauss_hermite(m, L)
    G, div = model.diffusion.with_divergence(nodes, None, None)
    f = posterior.marginal_drift(nodes, slice(None), G, div)
    E = lambda v: torch.einsum("k,k...->...", weights, v)
    dz = nodes - m
    torch.testing.assert_close(E(f), (m_plus - m_minus) / (2 * h), rtol=1e-6, atol=1e-6)
    flux = E(f.unsqueeze(-1) * dz.unsqueeze(-2) + dz.unsqueeze(-1) * f.unsqueeze(-2) + G @ G.transpose(-1, -2))
    torch.testing.assert_close(flux, (S_plus - S_minus) / (2 * h), rtol=1e-6, atol=1e-6)


def test_gradients_reach_every_part_and_the_model_survives_a_reload():
    batch = _batch()
    posterior = MarginalKalmanPosterior(Z, GaussianPotentials(D, Z, hidden=(8,)))
    model = _model(posterior, _affine(), hidden=(8,))  # nonlinear drift: statistical linearization
    loss, terms = model.loss(batch)
    assert all(math.isfinite(v.item()) for v in terms.values())
    loss.backward()
    pot = posterior.potentials
    for p in (model.drift.net[0].weight, model.diffusion.raw_g, model.diffusion.C0, model.diffusion.C, model.z0_mean,
              model.z0_logvar, pot.W, pot.log_s, pot.net[-1].weight, model.decoder.mean[0].weight):
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
