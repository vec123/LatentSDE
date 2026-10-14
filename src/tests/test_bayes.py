import math

import numpy as np
import torch
import torch.distributions as D
from scipy.integrate import quad

from ..bayes import (CenteredHorseshoePrior, CollapsedHorseshoe, LinearCollapsedHorseshoe, SVIHalfCauchyPrior,
                     kl_lognormal_gamma, kl_lognormal_inverse_gamma)
from ..bayes.collapsed_horseshoe import log_horseshoe_density
from ..bayes.fixed_priors import KINDS, sample_observations
from ._reference import run_tests, svise_module, use_float64

use_float64()
NUM_MC = 400_000


def _randomise(module, seed):
    """Moderate random values for all means and log standard deviations of a variational module."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "log_sigma" in name:
                p.copy_(-1.5 + torch.rand(p.shape, generator=g))
            else:
                p.copy_(0.5 * torch.randn(p.shape, generator=g))


def _mc_kl(pairs):
    """Monte Carlo KL of independent factors: sum over (q, p) of the mean of log q - log p, and its std error."""
    total, var = 0.0, 0.0
    for q, p in pairs:
        x = q.sample((NUM_MC,))
        diff = (q.log_prob(x) - p.log_prob(x)).reshape(NUM_MC, -1).sum(-1)
        total, var = total + diff.mean(), var + diff.var() / NUM_MC
    return total, math.sqrt(var)


def _lognormal(factor):
    return D.LogNormal(factor.mu, torch.exp(factor.log_sigma))


def test_lognormal_kls_match_monte_carlo():
    mu, log_sigma = torch.tensor([0.3, -1.0]), torch.tensor([-0.7, -1.2])
    q = D.LogNormal(mu, log_sigma.exp())
    for kl_fn, p in ((kl_lognormal_gamma, D.Gamma(0.5, 2.0)), (kl_lognormal_inverse_gamma, D.InverseGamma(0.5, 2.0))):
        expected, se = _mc_kl([(q, p)])
        assert abs(kl_fn(mu, log_sigma, 0.5, 2.0).sum() - expected) < 6 * se


def test_noncentered_kl_matches_monte_carlo():
    tau = 1e-2  # SVISE convention: tau = tau_0^2, so s_a ~ Gamma(1/2, rate 1 / tau)
    prior = SVIHalfCauchyPrior(5, tau)
    _randomise(prior, 0)
    w = prior.w_tilde
    expected, se = _mc_kl([
        (D.Normal(w.mu, torch.exp(w.log_sigma)), D.Normal(0.0, 1.0)),
        (_lognormal(prior.gamma_a), D.Gamma(0.5, 1.0)),
        (_lognormal(prior.gamma_b), D.InverseGamma(0.5, 1.0)),
        (_lognormal(prior.s_a), D.Gamma(0.5, 1.0 / tau)),
        (_lognormal(prior.s_b), D.InverseGamma(0.5, 1.0)),
    ])
    assert abs(prior.kl_divergence() - expected) < 6 * se


def test_centered_kl_matches_monte_carlo():
    tau = 0.3
    prior = CenteredHorseshoePrior(4, tau=tau)
    _randomise(prior, 1)
    factors = {name: _lognormal(getattr(prior, name)) for name in ("z_a", "z_b", "s_a", "s_b")}
    q_theta = D.Normal(prior.theta.mu, torch.exp(prior.theta.log_sigma))
    samples = {name: q.sample((NUM_MC,)) for name, q in factors.items()}
    theta = q_theta.sample((NUM_MC,))
    variance = samples["z_a"] * samples["z_b"] * samples["s_a"] * samples["s_b"]
    log_q = q_theta.log_prob(theta).sum(-1) + sum(q.log_prob(samples[n]).sum(-1) for n, q in factors.items())
    log_p = (D.Normal(0.0, variance.sqrt()).log_prob(theta).sum(-1)
             + D.Gamma(0.5, 1.0).log_prob(samples["z_a"]).sum(-1)
             + D.InverseGamma(0.5, 1.0).log_prob(samples["z_b"]).sum(-1)
             + D.Gamma(0.5, 1.0 / tau).log_prob(samples["s_a"]).sum(-1)
             + D.InverseGamma(0.5, 1.0).log_prob(samples["s_b"]).sum(-1))
    diff = log_q - log_p
    assert abs(prior.kl_divergence() - diff.mean()) < 6 * diff.std() / math.sqrt(NUM_MC)


def test_noncentered_prior_matches_svise():
    torch.manual_seed(3)
    reference = svise_module("variationalsparsebayes").SVIHalfCauchyPrior(7, 1e-5, w_init=torch.randn(7))
    torch.manual_seed(3)
    prior = SVIHalfCauchyPrior(7, 1e-5, w_init=torch.randn(7))
    torch.testing.assert_close(prior.kl_divergence(), reference.kl_divergence())
    torch.manual_seed(4)
    expected = reference.get_reparam_weights(3)
    torch.manual_seed(4)
    torch.testing.assert_close(prior.get_reparam_weights(3), expected)


def test_get_uncertainty_matches_samples():
    prior = SVIHalfCauchyPrior(3, 1e-2)
    _randomise(prior, 2)
    mean, std = prior.get_uncertainty()
    samples = prior.get_reparam_weights(NUM_MC)
    torch.testing.assert_close(mean, samples.mean(0), rtol=0.02, atol=0.01)
    torch.testing.assert_close(std, samples.std(0), rtol=0.02, atol=0.01)


def test_centered_pruning_masks_small_weights():
    prior = CenteredHorseshoePrior(4, w_init=torch.tensor([1.0, 1e-3, -2.0, 0.0]))
    prior.update_sparse_index()
    assert prior.sparse_index.tolist() == [True, False, True, False]
    assert prior.get_reparam_weights(5).shape == (5, 2)
    assert torch.isfinite(prior.kl_divergence())
    prior.reset_sparse_index()
    assert prior.get_reparam_weights(5).shape == (5, 4)


def _log_horseshoe_by_quadrature(theta, tau):
    """log int N(theta | 0, lambda^2 tau^2) HalfCauchy(lambda | 0, 1) dlambda, split at the peak lambda = theta / tau."""
    f = lambda lam: math.exp(-theta**2 / (2 * lam**2 * tau**2)) / math.sqrt(2 * math.pi * lam**2 * tau**2) \
        * 2 / (math.pi * (1 + lam**2))  # noqa: E731
    return math.log(quad(f, 0, theta / tau, limit=500)[0] + quad(f, theta / tau, np.inf, limit=500)[0])


def test_horseshoe_density_matches_quadrature_and_has_exact_gradient():
    for theta in (1e-3, 0.1, 1.0, 3.0, 30.0):
        for tau in (0.01, 1.0):
            value = log_horseshoe_density(torch.tensor(theta**2), torch.tensor(tau)).item()
            assert abs(value - _log_horseshoe_by_quadrature(theta, tau)) < 1e-6, (theta, tau)
    theta_sq = torch.logspace(-3, 4, 40).requires_grad_(True)  # both branches of e^u E1(u), above the FD step
    assert torch.autograd.gradcheck(lambda x: log_horseshoe_density(x, torch.tensor(0.3)), (theta_sq,))


def test_collapsed_prior_is_at_least_the_mean_field_prior():
    """For fixed q(theta) and tau, the collapsed term is the maximum over q(lambda): no log-normal q does better."""
    model = CollapsedHorseshoe(1, tau_0=0.1)
    with torch.no_grad():
        model.log_tau_log_sigma.fill_(-10.0)  # q(tau) a point mass at tau = 0.1
    z, w = (torch.tensor(a) for a in np.polynomial.hermite_e.hermegauss(60))
    w = w / math.sqrt(2 * math.pi)
    for mu, sigma in ((0.0, 0.01), (0.05, 0.02), (1.0, 0.1)):
        collapsed = model.collapsed_log_prior(torch.tensor([mu]), torch.tensor([sigma])).item()
        best = -math.inf
        for m in torch.linspace(-6, 4, 81):
            for s in (0.05, 0.2, 0.5, 1.0, 2.0):
                lam = torch.exp(m + s * z)
                e_log_p = (w * (-0.5 * torch.log(2 * math.pi * lam**2 * 0.01) - (mu**2 + sigma**2) / (2 * lam**2 * 0.01)
                                + math.log(2 / math.pi) - torch.log1p(lam**2))).sum()
                entropy = m + 0.5 * math.log(2 * math.pi * math.e * s**2)
                best = max(best, (e_log_p + entropy).item())
        assert best <= collapsed + 1e-6 and collapsed - best < 0.2, (mu, sigma, best, collapsed)


def test_collapsed_global_kl_is_bounded_below_and_exact():
    model = CollapsedHorseshoe(3, tau_0=0.1)
    values = []
    with torch.no_grad():
        for mu in (-2.0, -6.0, -10.0):  # the JAX version tended to minus infinity here
            model.log_tau_mu.fill_(mu)
            values.append(model.kl_global_scale().item())
        assert values[0] < values[1] < values[2]
        model.log_tau_mu.fill_(math.log(0.05))
        model.log_tau_log_sigma.fill_(math.log(0.8))  # a wide q(tau)
        q = D.LogNormal(model.log_tau_mu, model.log_tau_log_sigma.exp())
        cross = quad(lambda t: math.exp(q.log_prob(torch.tensor(t)).item() + D.HalfCauchy(0.1).log_prob(
            torch.tensor(t)).item()), 0, np.inf, limit=500)[0]
        expected_log_p = quad(lambda t: math.exp(q.log_prob(torch.tensor(t)).item())
                              * D.HalfCauchy(0.1).log_prob(torch.tensor(t)).item(), 0, np.inf, limit=500)[0]
        assert cross > 0
        assert abs(model.kl_global_scale().item() - (-q.entropy().item() - expected_log_p)) < 1e-6


def test_linear_collapsed_horseshoe_recovers_sparse_signal():
    torch.manual_seed(5)
    X = torch.randn(200, 10)
    signal = torch.tensor([3.0, 0, 0, -2.0, 0, 0, 0, 0, 0, 0])
    y = X @ signal + 0.1 * torch.randn(200)
    model = LinearCollapsedHorseshoe(10, tau_0=1e-2, noise_std=0.1, X=X, y=y)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(1500):
        optimizer.zero_grad()
        model.loss_regression(X, y).backward()
        optimizer.step()
    mean, _ = model.get_uncertainty()
    torch.testing.assert_close(mean.detach(), signal, atol=0.05, rtol=0.0)


def test_fixed_priors_sample_shapes():
    for kind in KINDS:
        betas, y = sample_observations(kind, 6, num_samples=4)
        assert betas.shape == y.shape == (4, 6)


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
