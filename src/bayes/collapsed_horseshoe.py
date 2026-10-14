"""
Collapsed variational horseshoe: the local scales are integrated out with the bound

    log p(theta | tau) >= log[(2 pi^3)^(-1/2) tau^(-1) log(1 + 2 tau^2 / theta^2)],

evaluated at E[theta^2]; q(theta) = N(mu, sigma^2), q(tau) = LogNormal, and a global tau ~ HalfCauchy(0, tau_0).
With slab_scale, theta^2 is replaced by theta^2 / (1 + theta^2 / slab_scale^2) (regularised, "Finnish" horseshoe).

Port of the old JAX src/distributions/collapsed_variationalHorsehoe.py. Changes: the KL of the global scale
is corrected (-mu_tau where the JAX version had +mu_tau, and the -1/2 log 2 pi term), as derived in
reference/svise_adaptations.tex; the four JAX classes are two classes with an optional slab_scale, plus aliases.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from .divergences import kl_lognormal_half_cauchy


class CollapsedHorseshoe(nn.Module):
    """
    Collapsed horseshoe over D weights.

    Args:
        D (int): number of weights
        tau_0 (float): scale of the global HalfCauchy prior
        slab_scale (float): None for the plain horseshoe, else the slab of the regularised horseshoe
        mu_init (Tensor): initial means of theta; default small random values
        init_log_sigma (float): initial log standard deviation of theta
    """

    def __init__(self, D: int, tau_0: float = 1e-2, slab_scale: float = None, log_sigma_min: float = -10.0,
                 log_sigma_max: float = 2.0, mu_init: Tensor = None, init_log_sigma: float = -2.0):
        super().__init__()
        self.D = D
        self.tau_0 = tau_0
        self.slab_scale = slab_scale
        self.log_sigma_min, self.log_sigma_max = log_sigma_min, log_sigma_max
        mu_init = torch.randn(D) * 0.01 if mu_init is None else mu_init.clone().detach()
        self.theta_mu = Parameter(mu_init)
        self.theta_log_sigma = Parameter(torch.full((D,), float(init_log_sigma)))
        self.log_tau_mu = Parameter(torch.tensor(math.log(tau_0)))
        self.log_tau_log_sigma = Parameter(torch.tensor(-2.0))

    def _clip_log_sigma(self, log_sigma: Tensor) -> Tensor:
        return torch.clamp(log_sigma, self.log_sigma_min, self.log_sigma_max)

    def _expected_log_prior_theta(self, mu: Tensor, sigma: Tensor, tau: Tensor) -> Tensor:
        e_sq = mu.pow(2) + sigma.pow(2)
        if self.slab_scale is not None:
            e_sq = e_sq / (1.0 + e_sq / self.slab_scale**2)
        tau_sq = torch.clamp(tau.pow(2), min=1e-12)
        ratio = 2.0 * tau_sq / (e_sq + 1e-8)
        return torch.sum(-0.5 * torch.log(2.0 * math.pi**3 * tau_sq) + torch.log(torch.log1p(ratio) + 1e-8))

    def _tau_eff(self) -> Tensor:
        sig_tau = torch.exp(self._clip_log_sigma(self.log_tau_log_sigma))
        return torch.clamp(torch.exp(self.log_tau_mu + 0.5 * sig_tau.pow(2)), min=1e-6)  # prevents tau = 0

    def kl_global_scale(self) -> Tensor:
        """KL(q(tau) || HalfCauchy(0, tau_0)), with E[log(1 + tau^2 / tau_0^2)] taken at E[tau]."""
        expected_log1p = torch.log1p((self._tau_eff() / self.tau_0) ** 2)
        return kl_lognormal_half_cauchy(self.log_tau_mu, self._clip_log_sigma(self.log_tau_log_sigma), self.tau_0,
                                        expected_log1p)

    def kl_divergence(self) -> Tensor:
        ls_t = self._clip_log_sigma(self.theta_log_sigma)
        entropy_theta = 0.5 * torch.sum(1.0 + math.log(2.0 * math.pi) + 2.0 * ls_t)
        exp_log_p_theta = self._expected_log_prior_theta(self.theta_mu, torch.exp(ls_t), self._tau_eff())
        return -entropy_theta - exp_log_p_theta + self.kl_global_scale()

    def get_uncertainty(self):
        """Posterior mean and standard deviation of theta."""
        return self.theta_mu, torch.exp(self._clip_log_sigma(self.theta_log_sigma))

    def sample(self, n: int = 1) -> Tensor:
        mu, std = self.get_uncertainty()
        return mu + std * torch.randn(n, self.D, dtype=mu.dtype, device=mu.device)


class LinearCollapsedHorseshoe(CollapsedHorseshoe):
    """Collapsed horseshoe for linear regression y = X theta + noise with known noise_std."""

    def __init__(self, D: int, tau_0: float = 1e-2, noise_std: float = 5.0, slab_scale: float = None,
                 log_sigma_min: float = -10.0, log_sigma_max: float = 3.0, X: Tensor = None, y: Tensor = None):
        mu_init = X.T @ y / (X.shape[0] + 10.0) if X is not None and y is not None else None
        super().__init__(D, tau_0, slab_scale, log_sigma_min, log_sigma_max, mu_init=mu_init)
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

    def __init__(self, module: nn.Module, tau_0: float = 1e-2, slab_scale: float = None,
                 log_sigma_min: float = -10.0, log_sigma_max: float = 2.0, init_log_sigma: float = -5.0):
        self._shapes = {name: p.shape for name, p in module.named_parameters()}
        flat = torch.cat([p.detach().reshape(-1) for p in module.parameters()])
        super().__init__(flat.numel(), tau_0, slab_scale, log_sigma_min, log_sigma_max, mu_init=flat,
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


class LinearCollapsedVariationalHorseshoe(LinearCollapsedHorseshoe):
    """Name of the old JAX class."""


class LinearCollapsedVariationalHorseshoeRegularized(LinearCollapsedHorseshoe):
    """Name of the old JAX class: slab_scale 15 and log_sigma_max 2 by default."""

    def __init__(self, D: int, tau_0: float = 1e-2, noise_std: float = 5.0, slab_scale: float = 15.0,
                 log_sigma_min: float = -10.0, log_sigma_max: float = 2.0, **kwargs):
        super().__init__(D, tau_0, noise_std, slab_scale, log_sigma_min, log_sigma_max, **kwargs)


class NonLinearCollapsedVariationalHorseshoe(NonLinearCollapsedHorseshoe):
    """Name of the old JAX class."""


class NonLinearCollapsedVariationalHorseshoeRegularized(NonLinearCollapsedHorseshoe):
    """Name of the old JAX class: slab_scale 15 by default."""

    def __init__(self, module: nn.Module, tau_0: float = 1e-2, slab_scale: float = 15.0, **kwargs):
        super().__init__(module, tau_0, slab_scale, **kwargs)
