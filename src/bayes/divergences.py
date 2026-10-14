"""
Closed-form KL divergences from a log-normal q to the scale priors of the horseshoe.

Port of the old JAX src/distributions/divergences.py with the corrected log-normal entropy
(reference/svise_adaptations.tex, "Correctness fixes"). Rates, not scales, parameterise Gamma and InvGamma.
"""
import math

import torch
from torch import Tensor


def kl_lognormal_gamma(mu: Tensor, log_sigma: Tensor, shape, rate) -> Tensor:
    """KL(LogNormal(mu, sigma^2) || Gamma(shape, rate)), elementwise."""
    shape, rate = torch.as_tensor(shape, dtype=mu.dtype), torch.as_tensor(rate, dtype=mu.dtype)
    sigma2 = torch.exp(2 * log_sigma)
    return (-0.5 - log_sigma - 0.5 * math.log(2 * math.pi) + torch.lgamma(shape) - shape * torch.log(rate)
            - shape * mu + rate * torch.exp(mu + 0.5 * sigma2))


def kl_lognormal_inverse_gamma(mu: Tensor, log_sigma: Tensor, shape, rate) -> Tensor:
    """KL(LogNormal(mu, sigma^2) || InvGamma(shape, rate)), elementwise."""
    shape, rate = torch.as_tensor(shape, dtype=mu.dtype), torch.as_tensor(rate, dtype=mu.dtype)
    sigma2 = torch.exp(2 * log_sigma)
    return (-0.5 - log_sigma - 0.5 * math.log(2 * math.pi) + torch.lgamma(shape) - shape * torch.log(rate)
            + shape * mu + rate * torch.exp(-mu + 0.5 * sigma2))


def kl_lognormal_half_cauchy(mu: Tensor, log_sigma: Tensor, scale, expected_log1p: Tensor) -> Tensor:
    """
    KL(LogNormal(mu, sigma^2) || HalfCauchy(0, scale)) =
    -1/2 - log sigma - 1/2 log 2 pi - log(2 / pi) + log scale - mu + E[log(1 + tau^2 / scale^2)].

    expected_log1p: the expectation E[log(1 + tau^2 / scale^2)] under q, or an approximation of it.
    """
    return (-0.5 - log_sigma - 0.5 * math.log(2 * math.pi) - math.log(2 / math.pi) + math.log(scale) - mu
            + expected_log1p)
