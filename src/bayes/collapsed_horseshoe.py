"""
Collapsed variational horseshoe (standard collapsed VI): the local scales are integrated out with their optimal
variational distribution.

Model: theta_j | lambda_j, tau ~ N(0, lambda_j^2 tau^2), lambda_j ~ HalfCauchy(0, 1), tau ~ HalfCauchy(0, tau_0).
Variational family: q(theta) = prod_j N(mu_j, sigma_j^2), q(tau) = LogNormal, q(lambda | tau) optimal. For a
fixed q(theta), the best q(lambda_j) gives (King & Lawrence 2006; Hensman, Rattray & Lawrence 2012)

    max_q(lambda) E_q[log p(theta_j | lambda_j, tau) + log p(lambda_j) - log q(lambda_j)]
        = log int p(lambda) exp(E_q(theta) log N(theta_j | 0, lambda^2 tau^2)) dlambda
        = log p_HS(sqrt(E[theta_j^2]) | tau),

with the horseshoe marginal p_HS(theta | tau) = (2 pi^3)^(-1/2) tau^(-1) e^u E1(u), u = theta^2 / (2 tau^2).
The result is a lower bound on the evidence, at least as tight as any mean-field q(theta) q(lambda) with the same
q(theta) and q(tau). Gauss-Hermite quadrature gives the expectations over q(tau).

Port of the old JAX src/distributions/collapsed_variationalHorsehoe.py. Changes: the exact marginal in place of
the upper bound log(1 + 2 tau^2 / theta^2) of Carvalho et al. (2010); quadrature over q(tau) in place of E[tau]
put into the bound; the corrected KL of the global scale (reference/svise_adaptations.tex); no slab, because the
regularised horseshoe has no closed-form marginal.

CollapsedNonCenteredHorseshoe has the same model, but the local scales are integrated out exactly (q(lambda |
theta) = p(lambda | theta, tau)) and q(theta) has the shape of SVISE's non-centred q; it is a SparsityPrior for
the sparse drift ("collapsed" in make_sparsity_prior).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from .divergences import kl_lognormal_half_cauchy
from .horseshoe import SparsityPrior, _scatter
from .mean_field import LogNormalMeanFieldVariational, NormalMeanFieldVariational


def _log_scaled_exp1(u: Tensor) -> Tensor:
    """log(e^u E1(u)) for u > 0: power series up to u = 2, continued fraction above (error below 1e-13)."""
    small = u <= 2.0
    us = torch.where(small, u, torch.ones_like(u))
    k = torch.arange(1, 31, dtype=u.dtype, device=u.device)
    e1 = -0.5772156649015329 - torch.log(us) - ((-us.unsqueeze(-1)) ** k / (k * torch.exp(torch.lgamma(k + 1)))).sum(-1)
    ul = torch.where(small, torch.ones_like(u), u)
    t = torch.zeros_like(ul)
    for n in range(40, 0, -1):  # E1(u) = e^-u / (u + 1 - 1 / (u + 3 - 4 / (u + 5 - ...)))
        t = n**2 / (ul + 2 * n + 1 - t)
    return torch.where(small, us + torch.log(e1), -torch.log(ul + 1 - t))


class _LogScaledExp1(torch.autograd.Function):
    """log(e^u E1(u)) with the exact derivative 1 - 1 / (u e^u E1(u)), from dE1/du = -e^-u / u."""

    @staticmethod
    def forward(ctx, u: Tensor) -> Tensor:
        f = _log_scaled_exp1(u)
        ctx.save_for_backward(u, f)
        return f

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        u, f = ctx.saved_tensors
        return grad * (1 - torch.exp(-f) / u)


def log_horseshoe_density(theta_sq: Tensor, tau: Tensor) -> Tensor:
    """log p_HS(theta | tau) of the horseshoe marginal, from theta^2."""
    u = theta_sq / (2 * tau**2)
    return -0.5 * math.log(2 * math.pi**3) - torch.log(tau) + _LogScaledExp1.apply(u)


class CollapsedHorseshoe(nn.Module):
    """
    Collapsed horseshoe over D weights. The ELBO is E_q log p(y | theta) - kl_divergence().

    Args:
        D (int): number of weights
        tau_0 (float): scale of the global HalfCauchy prior
        mu_init (Tensor): initial means of theta; default small random values
        init_log_sigma (float): initial log standard deviation of theta
        n_quad (int): Gauss-Hermite nodes for the expectations over q(tau)
    """

    def __init__(self, D: int, tau_0: float = 1e-2, log_sigma_min: float = -10.0, log_sigma_max: float = 2.0,
                 mu_init: Tensor = None, init_log_sigma: float = -2.0, n_quad: int = 20):
        super().__init__()
        self.D = D
        self.tau_0 = tau_0
        self.log_sigma_min, self.log_sigma_max = log_sigma_min, log_sigma_max
        mu_init = torch.randn(D) * 0.01 if mu_init is None else mu_init.clone().detach()
        self.theta_mu = Parameter(mu_init)
        self.theta_log_sigma = Parameter(torch.full((D,), float(init_log_sigma)))
        self.log_tau_mu = Parameter(torch.tensor(math.log(tau_0)))
        self.log_tau_log_sigma = Parameter(torch.tensor(-2.0))
        z, w = np.polynomial.hermite_e.hermegauss(n_quad)
        self.register_buffer("quad_z", torch.tensor(z, dtype=torch.get_default_dtype()))
        self.register_buffer("quad_w", torch.tensor(w / math.sqrt(2 * math.pi), dtype=torch.get_default_dtype()))

    def _clip_log_sigma(self, log_sigma: Tensor) -> Tensor:
        return torch.clamp(log_sigma, self.log_sigma_min, self.log_sigma_max)

    def _tau_nodes(self):
        """Gauss-Hermite nodes tau_k of q(tau), and their weights."""
        sigma_tau = torch.exp(self._clip_log_sigma(self.log_tau_log_sigma))
        return torch.exp(self.log_tau_mu + sigma_tau * self.quad_z), self.quad_w

    def collapsed_log_prior(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """E_q(tau) sum_j log p_HS(sqrt(E[theta_j^2]) | tau): the prior term with the optimal q(lambda)."""
        tau, w = self._tau_nodes()
        return (w * log_horseshoe_density((mu.pow(2) + sigma.pow(2)).unsqueeze(-1), tau).sum(0)).sum()

    def kl_global_scale(self) -> Tensor:
        """KL(q(tau) || HalfCauchy(0, tau_0))."""
        tau, w = self._tau_nodes()
        expected_log1p = (w * torch.log1p((tau / self.tau_0) ** 2)).sum()
        return kl_lognormal_half_cauchy(self.log_tau_mu, self._clip_log_sigma(self.log_tau_log_sigma), self.tau_0,
                                        expected_log1p)

    def kl_divergence(self) -> Tensor:
        """E_q log q(theta) - collapsed_log_prior + KL(q(tau) || p(tau))."""
        ls_t = self._clip_log_sigma(self.theta_log_sigma)
        entropy_theta = 0.5 * torch.sum(1.0 + math.log(2.0 * math.pi) + 2.0 * ls_t)
        return -entropy_theta - self.collapsed_log_prior(self.theta_mu, torch.exp(ls_t)) + self.kl_global_scale()

    def get_uncertainty(self):
        """Posterior mean and standard deviation of theta."""
        return self.theta_mu, torch.exp(self._clip_log_sigma(self.theta_log_sigma))

    def sample(self, n: int = 1) -> Tensor:
        mu, std = self.get_uncertainty()
        return mu + std * torch.randn(n, self.D, dtype=mu.dtype, device=mu.device)


def _log_product_density(theta: Tensor, mu: Tensor, sigma: Tensor, m: Tensor, v: Tensor, z: Tensor,
                         w: Tensor) -> Tensor:
    """
    log q(theta) of theta = w r, w ~ N(mu, sigma^2), r ~ LogNormal(m, v^2), for theta (d, n) and (d,) parameters.

    q(theta) = int exp(-f(a)) da / (2 pi sigma v), f(a) = a + (theta e^-a - mu)^2 / (2 sigma^2) + (a - m)^2 / (2 v^2),
    with a = log r. f is convex if v |mu| <= sigma (f'' >= 1 / v^2 - mu^2 / (8 sigma^2) > 0) or if theta and mu
    have opposite signs; then Gauss-Hermite (nodes z, weights w) around its minimum (Laplace) is accurate. Else the
    N(w) factor is the narrow one, and Gauss-Hermite over w is accurate.
    """
    mu, sigma, m, v = (p.unsqueeze(-1) for p in (mu, sigma, m, v))
    use_a = (v * mu.abs() <= sigma) | (theta * mu <= 0)
    with torch.no_grad():  # minimum and curvature of f; any centre and scale give a valid rule
        a = torch.maximum(m, torch.log(theta.abs() / sigma))
        for _ in range(40):
            e = theta * torch.exp(-a)
            step = (1 - e * (e - mu) / sigma**2 + (a - m) / v**2) / (e * (2 * e - mu) / sigma**2 + 1 / v**2)
            a = a - torch.where(use_a, step, torch.zeros_like(step)).clamp(-3.0, 3.0)
        e = theta * torch.exp(-a)
        scale = torch.where(use_a, e * (2 * e - mu) / sigma**2 + 1 / v**2, torch.ones_like(a)).rsqrt()
    a_k = a.unsqueeze(-1) + scale.unsqueeze(-1) * z
    f = (a_k + (theta.unsqueeze(-1) * torch.exp(-a_k) - mu.unsqueeze(-1)).pow(2) / (2 * sigma.unsqueeze(-1) ** 2)
         + (a_k - m.unsqueeze(-1)).pow(2) / (2 * v.unsqueeze(-1) ** 2))
    over_a = (torch.logsumexp(torch.log(w) + 0.5 * z**2 - f, -1) + torch.log(scale)
              - 0.5 * math.log(2 * math.pi) - torch.log(sigma * v))
    w_k = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * z  # q(theta) = E_w LogNormal(theta / w) / |w|, w of theta's sign
    ratio = theta.unsqueeze(-1) / w_k
    valid = ratio > 0
    log_ratio = torch.log(torch.where(valid, ratio, torch.ones_like(ratio)))
    over_w = -(log_ratio - m.unsqueeze(-1)).pow(2) / (2 * v.unsqueeze(-1) ** 2)
    over_w = torch.where(valid, over_w, torch.full_like(over_w, -1e30))
    over_w = torch.logsumexp(torch.log(w) + over_w, -1) - torch.log(theta.abs() * v * math.sqrt(2 * math.pi))
    return torch.where(use_a, over_a, over_w)


class CollapsedNonCenteredHorseshoe(SparsityPrior):
    """
    Collapsed horseshoe with q(theta) of non-centred shape, as a SparsityPrior over d weights.

    Model: theta_j | tau ~ p_HS(theta_j | tau) (local scales integrated out), tau ~ HalfCauchy(0, tau_0).
    Variational family: theta_j = w_j r_j with w_j ~ N(mu_j, sigma_j^2) and r_j ~ LogNormal(m_j, v_j^2), where r_j
    is an auxiliary scale of q only; q(tau) = LogNormal. This q(theta_j) has the shape of SVISE's q: a peak at 0
    with heavy tails, or a skewed bump. For the same q(theta), the bound is the non-centred mean-field bound plus
    E_q KL(q(lambda | theta) || p(lambda | theta, tau)), so never lower.

    Monte Carlo with reparameterised samples gives E_q[log p_HS(theta | tau) - log q(theta)] (log p_HS is singular
    at 0, where quadrature converges slowly), _log_product_density gives log q(theta), Gauss-Hermite the KL of tau.
    Pruning keeps the weights with log E[theta^2] above the mid-range of all, as SVISE does with its scales.

    Args:
        d (int): number of weights
        tau (float): tau_0^2, as for SVIHalfCauchyPrior
        w_init (Tensor): initial weights; then q(theta) starts nearly at w_init, as in SVIHalfCauchyPrior
        n_samples (int): Monte Carlo samples for the prior terms
        n_quad (int): Gauss-Hermite nodes for log q(theta) and for the KL of tau
    """

    def __init__(self, d: int, tau: float, w_init: Tensor = None, n_samples: int = 32, n_quad: int = 20):
        super().__init__()
        self.tau_0 = math.sqrt(float(tau))
        if w_init is None:
            self.w = NormalMeanFieldVariational(torch.zeros(d), torch.zeros(d))
            self.r = LogNormalMeanFieldVariational(torch.full((d,), math.log(self.tau_0)), torch.zeros(d))
            log_tau_init = math.log(self.tau_0)
        else:
            self.w = NormalMeanFieldVariational(w_init.clone().detach(), torch.full((d,), -6.0))
            self.r = LogNormalMeanFieldVariational(torch.zeros(d), torch.full((d,), -5.0))
            log_tau_init = 0.0
        self.log_tau_mu = Parameter(torch.tensor(log_tau_init))
        self.log_tau_log_sigma = Parameter(torch.tensor(-2.0))
        z, w = np.polynomial.hermite_e.hermegauss(n_quad)
        self.register_buffer("quad_z", torch.tensor(z, dtype=torch.get_default_dtype()))
        self.register_buffer("quad_w", torch.tensor(w / math.sqrt(2 * math.pi), dtype=torch.get_default_dtype()))
        self.register_buffer("sparse_index", torch.ones(d, dtype=torch.bool))
        self.n_samples = n_samples

    def get_reparam_weights(self, n: int) -> Tensor:
        return self.w(n) * self.r(n)

    def kl_divergence(self) -> Tensor:
        """E_q log q(theta) - E_q log p_HS(theta | tau) + KL(q(tau) || HalfCauchy(0, tau_0)), active weights."""
        theta = self.get_reparam_weights(self.n_samples)  # (n, d)
        sigma_tau = torch.exp(self.log_tau_log_sigma)
        noise = torch.randn(self.n_samples, 1, dtype=theta.dtype, device=theta.device)
        log_p = log_horseshoe_density(theta.pow(2), torch.exp(self.log_tau_mu + sigma_tau * noise))
        log_q = _log_product_density(theta.T, self.w.mu, torch.exp(self.w.log_sigma), self.r.mu,
                                     torch.exp(self.r.log_sigma), self.quad_z, self.quad_w).T
        tau = torch.exp(self.log_tau_mu + sigma_tau * self.quad_z)
        expected_log1p = (self.quad_w * torch.log1p((tau / self.tau_0) ** 2)).sum()
        kl_tau = kl_lognormal_half_cauchy(self.log_tau_mu, self.log_tau_log_sigma, self.tau_0, expected_log1p)
        return (log_q - log_p).sum(-1).mean() + kl_tau

    def _moments(self):
        mu, var, m, v2 = self.w.mu, self.w.var(), self.r.mu, self.r.var()
        mean = mu * torch.exp(m + 0.5 * v2)
        second_moment = (mu.pow(2) + var) * torch.exp(2 * m + 2 * v2)
        return mean, second_moment

    def update_sparse_index(self) -> Tensor:
        score = torch.log(self._moments()[1])
        score = score - 0.5 * (score.max() + score.min())
        sparse_index = self.sparse_index.clone()
        sparse_index[self.sparse_index] = (score >= 0).to(sparse_index.device)
        self.sparse_index = sparse_index
        self._propagate_sparse_index()
        return score

    def reset_sparse_index(self) -> None:
        self.sparse_index = torch.ones(len(self.sparse_index), dtype=torch.bool, device=self.sparse_index.device)
        self._propagate_sparse_index()

    def _propagate_sparse_index(self) -> None:
        self.w.update_sparse_index(self.sparse_index)
        self.r.update_sparse_index(self.sparse_index)

    def get_uncertainty(self):
        mean, second_moment = self._moments()
        std = (second_moment - mean.pow(2)).clamp_min(0.0).sqrt()
        return _scatter(mean, self.sparse_index), _scatter(std, self.sparse_index)


class LinearCollapsedHorseshoe(CollapsedHorseshoe):
    """Collapsed horseshoe for linear regression y = X theta + noise with known noise_std."""

    def __init__(self, D: int, tau_0: float = 1e-2, noise_std: float = 5.0, log_sigma_min: float = -10.0,
                 log_sigma_max: float = 3.0, X: Tensor = None, y: Tensor = None):
        mu_init = X.T @ y / (X.shape[0] + 10.0) if X is not None and y is not None else None
        super().__init__(D, tau_0, log_sigma_min, log_sigma_max, mu_init=mu_init)
        self.noise_std = noise_std

    def loss_regression(self, X: Tensor, y: Tensor) -> Tensor:
        return self.loss_annealed(X, y, kl_weight=1.0)

    def loss_annealed(self, X: Tensor, y: Tensor, kl_weight: float = 1.0) -> Tensor:
        """Expected negative log-likelihood (closed form) plus kl_weight times the KL."""
        sig_t = torch.exp(self._clip_log_sigma(self.theta_log_sigma))
        residual_sq = torch.sum((y - X @ self.theta_mu) ** 2)
        variance_penalty = torch.sum(torch.sum(X**2, dim=0) * sig_t.pow(2))
        nll = (0.5 / self.noise_std**2) * (residual_sq + variance_penalty)
        return nll + kl_weight * self.kl_divergence()


class NonLinearCollapsedHorseshoe(CollapsedHorseshoe):
    """
    Collapsed horseshoe over all parameters of a network, centred on its current weights; the network itself is
    only a template, its parameters are replaced by samples through torch.func.functional_call.
    """

    def __init__(self, module: nn.Module, tau_0: float = 1e-2, log_sigma_min: float = -10.0,
                 log_sigma_max: float = 2.0, init_log_sigma: float = -5.0):
        self._shapes = {name: p.shape for name, p in module.named_parameters()}
        flat = torch.cat([p.detach().reshape(-1) for p in module.parameters()])
        super().__init__(flat.numel(), tau_0, log_sigma_min, log_sigma_max, mu_init=flat,
                         init_log_sigma=init_log_sigma)

    def unflatten(self, flat: Tensor) -> dict:
        params, start = {}, 0
        for name, shape in self._shapes.items():
            n = math.prod(shape)
            params[name] = flat[start:start + n].reshape(shape)
            start += n
        return params

    def loss_elbo(self, module: nn.Module, X: Tensor, y: Tensor, num_total_samples: int, kl_weight: float = 1.0,
                  num_mc_samples: int = 1) -> Tensor:
        """Monte Carlo ELBO loss of a classifier with logits module(X) and integer labels y."""
        nll = torch.stack([
            F.cross_entropy(torch.func.functional_call(module, self.unflatten(w), (X,)), y)
            for w in self.sample(num_mc_samples)
        ]).mean()
        return nll * num_total_samples + kl_weight * self.kl_divergence()

    def extract_sparse_parameters(self, threshold_std_ratio: float = 0.1):
        """Posterior-mean parameters with |mean| <= threshold_std_ratio * std set to zero, and the mask."""
        mu, std = self.get_uncertainty()
        mask = mu.abs() > threshold_std_ratio * std
        return self.unflatten((mu * mask).detach()), mask
