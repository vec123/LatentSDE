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
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from .divergences import kl_lognormal_half_cauchy


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
