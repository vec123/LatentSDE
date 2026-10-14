"""
Mean-field variational families.

NormalMeanFieldVariational, LogNormalMeanFieldVariational: adapted from SVISE,
reference/svise/svise/variationalsparsebayes/svi_half_cauchy.py (GPL-3.0), unchanged; a sparse_index masks
pruned weights.
GaussianVariational: port of the old JAX src/distributions/learnable.py; the full covariance now starts at the
identity (the JAX version started at softplus(1) I).
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softplus
from torch.nn.parameter import Parameter

from ..core.transforms import inverse_softplus


class NormalMeanFieldVariational(nn.Module):
    """
    Mean field parameterization for normal variational distribution

    Args:
        mu_init (Tensor): mean initialization
        log_sigma_init (Tensor): log stdev initialization
    """

    def __init__(self, mu_init: Tensor, log_sigma_init: Tensor) -> None:
        super().__init__()
        # check that the input shapes are the same
        assert len(mu_init) == len(log_sigma_init), "init shapes must be equal."
        assert mu_init.shape == log_sigma_init.shape, "init shapes must be equal."
        self.total_weights = len(mu_init)
        # saving parameters
        self.register_buffer("sparse_index", torch.ones(len(mu_init)).bool())

        self.mu = mu_init
        self.log_sigma = log_sigma_init

    @property
    def mu(self) -> Tensor:
        return self.__mu[self.sparse_index]

    @mu.setter
    def mu(self, value: Tensor):
        self.__mu = Parameter(value)

    @property
    def log_sigma(self) -> Tensor:
        return self.__log_sigma[self.sparse_index]

    @log_sigma.setter
    def log_sigma(self, value: Tensor):
        self.__log_sigma = Parameter(value)

    @property
    def d(self) -> int:
        return int(self.sparse_index.sum())

    def var(self) -> Tensor:
        return torch.exp(self.log_sigma).pow(2)

    def update_sparse_index(self, sparse_index: Tensor) -> None:
        assert (
            len(sparse_index) == self.total_weights
        ), "Sparse index should be a bool array masking unimportant weights."
        self.sparse_index = sparse_index

    def forward(self, n: int) -> Tensor:
        """
        Generates reparameterized samples

        Args:
            n (int): number of reparam samples

        Returns:
            Tensor: (n,d) reparameterized samples from variational distribution
        """
        sigma = torch.exp(self.log_sigma)
        return self.mu + torch.randn(n, len(sigma), device=sigma.device) * sigma


class LogNormalMeanFieldVariational(NormalMeanFieldVariational):
    """
    Mean field parameterization for log normal variational distribution

    Args:
        mu_init (Tensor): mean initialization
        log_sigma_init (Tensor): log stdev initialization
    """

    def __init__(self, mu_init: Tensor, log_sigma_init: Tensor) -> None:
        super().__init__(mu_init, log_sigma_init)

    def forward(self, n: int) -> Tensor:
        """
        Generates reparameterized samples

        Args:
            n (int): number of reparam samples

        Returns:
            Tensor: (n,d) reparameterized samples from variational distribution
        """
        return torch.exp(super().forward(n))


class GaussianVariational(nn.Module):
    """
    Gaussian q(x) = N(mu, Sigma) with mu = 0 and Sigma = I at the start.

    Args:
        dim (int): dimension of x
        type (str): "diagonal" (parameter log_var) or "full" (Cholesky factor L with a softplus diagonal)
    """

    def __init__(self, dim: int, type: str = "full") -> None:
        super().__init__()
        if type not in ("diagonal", "full"):
            raise ValueError("type must be 'diagonal' or 'full'")
        self.type = type
        self.mu = Parameter(torch.zeros(dim))
        if type == "diagonal":
            self.log_var = Parameter(torch.zeros(dim))
        else:
            self.L = Parameter(torch.eye(dim) * inverse_softplus(torch.tensor(1.0)))

    def scale_tril(self) -> Tensor:
        if self.type == "diagonal":
            return torch.diag(torch.exp(0.5 * self.log_var))
        L = torch.tril(self.L, diagonal=-1)
        return L + torch.diag(softplus(torch.diagonal(self.L)))

    def rsample(self, n: int = 1) -> Tensor:
        """(n, dim) reparameterised samples."""
        noise = torch.randn(n, self.mu.shape[0], dtype=self.mu.dtype, device=self.mu.device)
        if self.type == "diagonal":
            return self.mu + noise * torch.exp(0.5 * self.log_var)
        return self.mu + noise @ self.scale_tril().T
