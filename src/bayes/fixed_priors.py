"""
Basic shrinkage priors as scale mixtures, beta_i ~ N(0, (lambda_i tau)^2), for simulation and comparison:

    horseshoe   lambda_i ~ HalfCauchy(0, 1)
    laplace     lambda_i^2 ~ Exponential(rate 1/2)
    student     lambda_i^2 ~ InvGamma(1, 1)

Port of the old JAX src/distributions/fixed/ (NumPyro models); without a probabilistic programming library
here, the models are samplers, and sample_observations adds the observation noise y = beta + N(0, sigma^2).
"""
import torch
import torch.distributions as D
from torch import Tensor

KINDS = ("horseshoe", "laplace", "student")


def local_scales(kind: str, n: int, num_samples: int = 1) -> Tensor:
    """(num_samples, n) local scales lambda."""
    ones = torch.ones(n)
    if kind == "horseshoe":
        return D.HalfCauchy(ones).sample((num_samples,))
    if kind == "laplace":
        return D.Exponential(0.5 * ones).sample((num_samples,)).sqrt()
    if kind == "student":
        return D.InverseGamma(ones, ones).sample((num_samples,)).sqrt()
    raise ValueError(f"Unknown kind '{kind}', choose from {KINDS}")


def sample_betas(kind: str, n: int, num_samples: int = 1, tau: float = 1.0) -> Tensor:
    """(num_samples, n) draws of beta from the prior."""
    return torch.randn(num_samples, n) * local_scales(kind, n, num_samples) * tau


def sample_observations(kind: str, n: int, num_samples: int = 1, tau: float = 1.0, sigma: float = 1.0):
    """(beta, y) with y = beta + N(0, sigma^2), each (num_samples, n)."""
    betas = sample_betas(kind, n, num_samples, tau)
    return betas, betas + sigma * torch.randn_like(betas)
