"""
Squared exponential kernels with an input warping and an analytic time derivative.

SquaredExponential follows the conventions of SVISE's Matern kernels (src/kernels/matern.py): amplitude
sigf^2, parameters rawlen and rawsigf behind a softplus. It replaces SVISE's SquaredExpKernel, which has no
warping and no time derivative.

DeepGaussianKernel is ROM-VISDE's deep kernel, reference/rom-visde/visde/kernel.py (GPL-3.0):
k(t1, t2) = sigma exp(-(phi(t1) - phi(t2))^2 / (2 lens^2)) with phi a NeuralWarping, its parameter names and
initial values, and the interpolation noise var, which AmortizedLatentVarGP adds to the kernel matrix.
Changes: phi is a NeuralWarping, and forward(..., return_grad=True) also returns dk/dt1.
"""
from typing import Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import softplus
from torch.nn.parameter import Parameter

from ..core.transforms import inverse_softplus
from .warping import IdentityWarp, InputWarping, NeuralWarping


def _squared_exponential(w1: Tensor, w2: Tensor, length_scale: Tensor, amplitude: Tensor, dw1: Tensor = None):
    """k_ij = amplitude exp(-(w1_i - w2_j)^2 / (2 l^2)) for w1 (n,), w2 (m,); with dw1, also dk_ij/dt1_i."""
    diff = w1.reshape(-1, 1) - w2.reshape(1, -1)
    k = torch.exp(-0.5 * diff.pow(2) / length_scale.pow(2)) * amplitude
    if dw1 is None:
        return k
    return k, -k * diff / length_scale.pow(2) * dw1.reshape(-1, 1)


class SquaredExponential(nn.Module):
    """Squared exponential kernel sigf^2 exp(-(w(t1) - w(t2))^2 / (2 l^2)) with an input warping w."""

    def __init__(self, input_warping: InputWarping = None, len_init: float = 1.0):
        super().__init__()
        self.rawlen = Parameter(torch.Tensor(1))
        self.rawsigf = Parameter(torch.Tensor(1))
        self.input_warping = IdentityWarp() if input_warping is None else input_warping
        nn.init.constant_(self.rawlen, inverse_softplus(torch.tensor(len_init)))
        nn.init.constant_(self.rawsigf, 0.5413)

    def get_natural_parameters(self) -> tuple:
        return (softplus(self.rawlen), softplus(self.rawsigf))

    def forward(self, x1: Tensor, x2: Tensor, return_grad=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """x1 (n,) or (n, 1), x2 (m,) or (m, 1) -> (n, m), and dk/dx1 if return_grad."""
        l, sigf = self.get_natural_parameters()
        w2 = self.input_warping(x2)
        if return_grad:
            w1, dw1 = self.input_warping(x1, return_grad=True)
            return _squared_exponential(w1, w2, l, sigf.pow(2), dw1)
        return _squared_exponential(self.input_warping(x1), w2, l, sigf.pow(2))


class DeepGaussianKernel(nn.Module):
    """
    ROM-VISDE's deep kernel. With batch_size (the window size M) and dt, the warping maps window times
    [0, (M - 1) dt] to [-1, 1], as in ROM-VISDE; else pass a NeuralWarping as warping.

    Args:
        net (nn.Module): network of the warping, R -> R; default ROM-VISDE's KernelNet (128, 128, ReLU)
        batch_size (int), dt (float): window size and time step, for the rescaling
        len_init, var_init (float): initial length scale (in rescaled time) and interpolation noise
        warping (NeuralWarping): used in place of net / batch_size / dt
    """

    def __init__(self, net: nn.Module = None, batch_size: int = None, dt: Union[float, Tensor] = None,
                 len_init: Union[float, Tensor] = 1e-1, var_init: Union[float, Tensor] = 1e-2,
                 warping: NeuralWarping = None) -> None:
        super().__init__()
        if warping is None:
            if batch_size is None or dt is None:
                raise ValueError("DeepGaussianKernel needs batch_size and dt, or a warping")
            dt = dt.item() if isinstance(dt, Tensor) else dt
            warping = NeuralWarping.for_window(batch_size, dt, net=net)
        self.warping = warping
        self.rawsigma = Parameter(inverse_softplus(torch.tensor(1.0)))
        self.rawlens = Parameter(inverse_softplus(torch.as_tensor(len_init).clone().detach()))
        self.rawvar = Parameter(inverse_softplus(torch.as_tensor(var_init).clone().detach()))

    @property
    def lens(self) -> Tensor:
        return softplus(self.rawlens)

    @property
    def var(self) -> Tensor:
        """Interpolation noise sigma^2, added to the diagonal of K by the user of the kernel."""
        return softplus(self.rawvar)

    @property
    def sigma(self) -> Tensor:
        return softplus(self.rawsigma)

    def forward(self, t1: Tensor, t2: Tensor, return_grad=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """t1 (..., n, 1), t2 (..., m, 1) -> (..., n, m), and dk/dt1 if return_grad."""
        w1, dw1 = self.warping(t1, return_grad=True)
        diff = w1 - self.warping(t2).transpose(-2, -1)
        k = self.sigma * torch.exp(-0.5 * diff.pow(2) / self.lens.pow(2))
        return (k, -k * diff / self.lens.pow(2) * dw1) if return_grad else k
