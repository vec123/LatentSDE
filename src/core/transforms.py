"""
Elementwise output transforms and affine data transforms.

Adapted from SVISE, reference/svise/svise/utils.py (GPL-3.0). Changes: MultivariateStandardizeTransform
uses torch.linalg.solve_triangular in place of the deprecated torch.triangular_solve.
"""
import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor, sigmoid
import torch.nn as nn


def softplus(x: Tensor) -> Tensor:
    return torch.log(1 + torch.exp(x))


def inverse_softplus(x: Tensor) -> Tensor:
    return torch.log(torch.exp(x) - 1)


def inverse_softplus_scalar(value: float) -> float:
    """Inverse of softplus, for a python float (e.g. a Parameter's init value)."""
    return math.log(math.expm1(value))


class Identity(nn.Module):
    def forward(self, x):
        return x

    def inverse(self, x):
        return x


class Positive(nn.Module):
    """softplus(beta x) / beta + 1e-6; with return_grad also the elementwise derivative."""

    def __init__(self, beta=1.0):
        super().__init__()
        self.output_transform = nn.Softplus(beta=beta)
        self.minimum = 1e-6

    def forward(self, x, return_grad=False):
        res = self.output_transform(x) + self.minimum
        if not return_grad:
            return res
        grad_res = sigmoid(x * self.output_transform.beta)
        return res, grad_res


class StandardizeTransform(nn.Module):
    """x -> (x - mean) / std, with mean and std per dimension of the data x (n, d)."""

    def __init__(self, x: Tensor):
        super().__init__()
        self.register_buffer("mu", x.mean(0))
        self.register_buffer("sigma", x.std(0))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mu) / self.sigma

    def transform(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def inverse(self, x: Tensor) -> Tensor:
        return x * self.sigma + self.mu

    def scale(self, dx: Tensor) -> Tensor:
        """Transform of a derivative dx/dt."""
        return dx / self.sigma


class MultivariateStandardizeTransform(nn.Module):
    """x -> L^-1 (x - mean) with L L^T = R (default: the sample covariance of x)."""

    def __init__(self, x: Tensor, R: Tensor = None):
        super().__init__()
        self.register_buffer("mu", x.mean(0))
        if R is None:
            from .linalg import sample_covariance

            R = sample_covariance(x)
        self.register_buffer("L", torch.linalg.cholesky(R))

    def forward(self, x: Tensor) -> Tensor:
        return torch.linalg.solve_triangular(self.L, (x - self.mu).t(), upper=False).t()

    def inverse(self, x: Tensor) -> Tensor:
        return x @ self.L.t() + self.mu


class ScaleTransform(nn.Module, ABC):
    """Scales data by a factor per dimension: forward(x) = x / scale."""

    @property
    @abstractmethod
    def scale(self) -> Tensor:
        """(d,) scaling factor"""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """x / scale"""

    @abstractmethod
    def inverse(self, x: Tensor) -> Tensor:
        """x * scale"""


class StdevScaleTransform(ScaleTransform):
    def __init__(self, x: Tensor):
        super().__init__()
        self.register_buffer("sigma", x.std(0))

    @property
    def scale(self) -> Tensor:
        return self.sigma

    def forward(self, x: Tensor) -> Tensor:
        return x / self.scale

    def inverse(self, x: Tensor) -> Tensor:
        return x * self.scale


class IdentityScaleTransform(ScaleTransform):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.register_buffer("ones", torch.ones(d))

    @property
    def scale(self) -> Tensor:
        return self.ones

    def forward(self, x: Tensor) -> Tensor:
        return x

    def inverse(self, x: Tensor) -> Tensor:
        return x
