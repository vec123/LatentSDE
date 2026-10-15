"""
Quadrature rules for integrals over [a, b] of functions evaluated on batches of times.

Adapted from SVISE, reference/svise/svise/quadrature/_unbiased_quadrature.py (GPL-3.0). Changes: imports,
the class StartifiedSampling is renamed StratifiedSampling (the old name stays as an alias), and
UnbiasedGaussLegendreQuad keeps the barycentric weights of its nodes and can evaluate f once at all times
(single_call).
"""
from abc import ABC, abstractmethod
from typing import Callable, Union
import warnings

import torch
from torch import Tensor
import torch.nn as nn

from ..core.interpolation import BarycentricInterpolate, barycentric_weights
from .gauss_legendre import gauss_legendre_vecs


def stratified_sample1d(
    t_0: Union[float, Tensor],
    t_1: Union[float, Tensor],
    M: int,
    L: int,
    dtype,
    device="cpu",
) -> Tensor:
    """Stratified sampling in 1D
        dtype ([TODO:parameter]): [TODO:description]
        device ([TODO:parameter]): [TODO:description]
        t_0 (Union[float, Tensor]): lower bound
        t_1 (Union[float, Tensor]): upper bound
        M (int): number of intervals
        L (int): number of samples / interval

    Returns:
        Tensor: (L, M) stratified samples
    """
    t = torch.linspace(t_0, t_1, M + 1, dtype=dtype, device=device)
    # compute interval width
    w = t[1] - t[0]
    samples = torch.rand(L, M, dtype=dtype, device=device) * w + t[:-1]
    return samples.flatten()


class QuadRule1D(ABC, nn.Module):
    """Abstract base class for 1D quad rules.

    Args:
        a (Union[float, Tensor]): lower bound of integration
        b (Union[float, Tensor]): upper bound of integration
        N (int): number of quadrature nodes
    """

    @abstractmethod
    def __init__(
        self, a: Union[float, Tensor], b: Union[float, Tensor], N: int, *args, **kwargs
    ) -> None:
        super().__init__()
        self.dtype = torch.get_default_dtype()
        # saving some constants
        self.a = a
        self.b = b
        self.N = N

    @abstractmethod
    def forward(self, f: Callable) -> Tensor:
        """Estimate for integral
        Args:
            f (Callable): function to be integrated

        Returns:
            Tensor: Iq integral estimate
        """
        pass


class MonteCarloQuad(QuadRule1D):
    def __init__(self, a: Union[float, Tensor], b: Union[float, Tensor], N: int):
        super().__init__(a, b, N)

    def forward(self, f: Callable):
        xmc = torch.rand(self.N, dtype=self.dtype) * (self.b - self.a) + self.a
        dtype = xmc.dtype
        # evaluating function at quadrature nodes + monte-carlo
        fmc = f(xmc)
        Imc = fmc.mean() * (self.b - self.a)
        return Imc


class StratifiedSampling(QuadRule1D):
    """Estimates integral with stratified sampling

    Args:
        a (Union[float, Tensor]): lower bound of integration
        b (Union[float, Tensor]): upper bound of integration
        N (int): number of quadrature nodes
    """

    def __init__(
        self, a: Union[float, Tensor], b: Union[float, Tensor], N: int
    ) -> None:
        super().__init__(a, b, N)

    def forward(self, f: Callable):
        # evaluating function at quadrature nodes + monte-carlo
        # xstrat = self.stratified_sample1d(self.a, self.b, self.N, 1)
        xstrat = stratified_sample1d(self.a, self.b, self.N, 1, dtype=self.dtype)
        Istrat = f(xstrat).mean() * (self.b - self.a)
        # estimating integral
        return Istrat


class GaussLegendreQuad(QuadRule1D):
    """Estimates integral with Gauss Legendre quad.

    Args:
        a (Union[float, Tensor]): lower bound of integration
        b (Union[float, Tensor]): upper bound of integration
        N (int): number of quadrature nodes
    """

    def __init__(self, a: Union[float, Tensor], b: Union[float, Tensor], N: int):
        super().__init__(a, b, N)
        wq, xq = gauss_legendre_vecs(N, a, b, dtype=self.dtype)
        self.register_buffer("wq", wq)
        self.register_buffer("xq", xq)

    def forward(self, f: Callable):
        # evaluating function at quadrature nodes + monte-carlo
        fxq = f(self.xq)
        Iq = (fxq @ self.wq).mean()
        # estimating integral
        return Iq


class UnbiasedGaussLegendreQuad(QuadRule1D):
    """Estimates integral using the unbiased quadrature scheme
    based on Gauss Legendre quadrature from L.-f. Lee, Interpolation,
    quadrature, and stochastic integration, Econometric Theory  17,  (2001).

    Args:
        a (Union[float, Tensor]): lower bound of integration
        b (Union[float, Tensor]): upper bound of integration
        N (int): number of quadrature nodes
        quad_percent (float): percentage of quad nodes to use for Legendre
        single_call (bool): evaluate f once at the nodes and the Monte Carlo times together; fewer operations,
            but random numbers drawn inside f come in another order than in SVISE
    """

    def __init__(
        self,
        a: Union[float, Tensor],
        b: Union[float, Tensor],
        N: int,
        quad_percent: float,
        gamma: Union[float, Tensor] = 1.0,
        single_call: bool = False,
    ):
        super().__init__(a, b, N)
        err_msg = "Percent of quadrature points must be between 0 and 1"
        assert quad_percent <= 1.0, err_msg
        assert quad_percent >= 0.0, err_msg
        # calculating some convenient quantities
        nq = int(N * quad_percent)
        nmc = N - nq
        self.nq = nq
        self.nmc = nmc
        wq, xq = gauss_legendre_vecs(nq, a, b, dtype=self.dtype)
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("wq", wq)
        self.register_buffer("xq", xq)
        self.register_buffer("interp_weights", barycentric_weights(xq))
        self.single_call = single_call

    def forward(self, f: Callable):
        # xmc = torch.rand(self.nmc, dtype=self.dtype) * (self.b - self.a) + self.a
        xmc = stratified_sample1d(
            self.a, self.b, self.nmc, 1, dtype=self.dtype, device=self.xq.device
        )
        dtype = xmc.dtype
        # evaluating function at quadrature nodes + monte-carlo
        if getattr(self, "single_call", False):
            f_all = f(torch.cat([self.xq, xmc]))
            fxq, fmc = f_all[: self.nq], f_all[self.nq :]
        else:
            fxq = f(self.xq)
            fmc = f(xmc)
        Iq = (fxq @ self.wq).mean()
        Imc = fmc.mean() * (self.b - self.a)
        # interpolating; models saved before interp_weights existed compute the weights here
        interp = BarycentricInterpolate(self.xq, fxq, getattr(self, "interp_weights", None))
        fip = interp(xmc)  # .to(dtype)
        divisor = 1
        while torch.isnan(fip).any():
            warnings.warn("Interp is nan, reducing interp degree.")
            divisor *= 2
            interp = BarycentricInterpolate(self.xq[::divisor], fxq[::divisor])
            fip = interp(xmc).to(dtype)
        Iip = fip.mean() * (self.b - self.a)
        # estimating integral
        return Imc - self.gamma * (Iip - Iq)


StartifiedSampling = StratifiedSampling
