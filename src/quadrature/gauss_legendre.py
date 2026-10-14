"""
Quadrature nodes and weights.

Adapted from SVISE, reference/svise/svise/quadrature/_quadrature.py (GPL-3.0). Change: the Gauss-Legendre
nodes come from numpy.polynomial.legendre.leggauss in place of the vendored fastgl (LGPL), in the same
ascending order.
"""
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def gauss_legendre_vecs(N: int, a: float = -1.0, b: float = 1.0, dtype: torch.dtype = None) -> Tuple[Tensor, Tensor]:
    """
    Gauss-Legendre quadrature on [a, b].

    Returns:
        (Tensor, Tensor): weights (N,), nodes (N,)
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    x, w = np.polynomial.legendre.leggauss(N)
    wq = torch.tensor(w, dtype=dtype) * (b - a) / 2
    xq = torch.tensor(x, dtype=dtype) * (b - a) / 2 + (a + b) / 2
    return (wq, xq)


def trapezoidal_vecs(N: int, a: float = -1.0, b: float = 1.0, dtype: torch.dtype = None) -> Tuple[Tensor, Tensor]:
    """
    Trapezoidal rule on [a, b] with N equally spaced nodes.

    Returns:
        (Tensor, Tensor): weights (N,), nodes (N,)
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    x = torch.linspace(a, b, N, dtype=dtype)
    w = torch.ones(N, dtype=dtype)
    w[0] = w[0] * 0.5
    w[-1] = w[-1] * 0.5
    w = (b - a) / (N - 1) * w
    return (w, x)
