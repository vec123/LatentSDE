"""
Input warpings w(t) of the time axis, with their derivative dw/dt for kernels that return time derivatives.

InputWarping, IdentityWarp, KumaraswamyWarping: adapted from SVISE, reference/svise/svise/kernels/_utils.py
(GPL-3.0), unchanged; fixed_bounds is new.
NeuralWarping: the time map of ROM-VISDE's deep kernel, reference/rom-visde/visde/kernel.py (GPL-3.0),
phi(t) = u + net(u) with u = (t - offset) / scale, written as an InputWarping.
"""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator, Sequence, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import softplus
from torch.nn.parameter import Parameter


class InputWarping(ABC, nn.Module):
    @abstractmethod
    def forward(self, t_in: Tensor, return_grad=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """w(t), or (w(t), dw/dt) elementwise"""

    @abstractmethod
    def inverse(self, w: Tensor) -> Tensor:
        pass


class IdentityWarp(InputWarping):
    def forward(self, t_in: Tensor, return_grad=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if return_grad:
            return t_in, torch.ones_like(t_in)
        return t_in

    def inverse(self, w: Tensor) -> Tensor:
        return w


class KumaraswamyWarping(InputWarping):
    """Kumaraswamy CDF warping of [t0, tf] onto itself, with learned shape parameters alpha and beta."""

    _update_buffers: bool = True

    def __init__(self, t_span: Tuple[Tensor, float], eps: float = 1e-6, dynamic_update_bounds: bool = True):
        super().__init__()
        if not isinstance(t_span[0], Tensor):
            self.register_buffer("t0", torch.tensor(t_span[0]) - eps)
            self.register_buffer("tf", torch.tensor(t_span[1]) + eps)
        else:
            self.register_buffer("t0", (t_span[0] - eps).clone().detach_())
            self.register_buffer("tf", (t_span[1] + eps).clone().detach_())
        self.register_buffer("eps", torch.tensor(eps))
        self.rawalpha = Parameter(torch.Tensor(1))
        self.rawbeta = Parameter(torch.Tensor(1))
        self.dynamic_update_bounds = dynamic_update_bounds
        self.reset_parameters()

    @property
    def dt(self) -> Tensor:
        return self.tf - self.t0

    def update_buffers(self, t_in: Tensor) -> None:
        if self.dynamic_update_bounds:
            if t_in.min() < self.t0:
                self.t0 = (t_in.min() - self.eps).clone().detach_()
            if t_in.max() > self.tf:
                self.tf = (t_in.max() + self.eps).clone().detach_()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.rawalpha, 0.5413)  # softplus(0.5413) = 1: identity warping
        nn.init.constant_(self.rawbeta, 0.5413)

    def get_natural_parameters(self):
        return (softplus(self.rawalpha), softplus(self.rawbeta))

    def forward(self, t_in: Tensor, return_grad=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self._update_buffers:
            self.update_buffers(t_in)
        alpha, beta = self.get_natural_parameters()
        t = (t_in - self.t0) / self.dt
        diff = 1 - t.pow(alpha)
        w = (1 - diff.pow(beta)).mul(self.dt).add(self.t0)
        if return_grad:
            dw = diff.pow(beta - 1).mul(t.pow(alpha - 1)).mul(beta * alpha)
            return (w, dw)
        return w

    def inverse(self, w: Tensor) -> Tensor:
        alpha, beta = self.get_natural_parameters()
        w = w.add(-self.t0).div(self.dt)
        t = (1 - (1 - w).pow(1 / beta)).pow(1 / alpha)
        return t.mul(self.dt).add(self.t0)


@contextmanager
def fixed_bounds(module: nn.Module, times: Tensor) -> Iterator[None]:
    """
    Extends the bounds of every KumaraswamyWarping in module to cover times, then skips their bound checks inside
    the block; each check costs two reductions per call and, on a GPU, a wait for the device. All inputs in the
    block must lie in [times.min(), times.max()]. The results are then the same as with the checks if the bounds
    already cover times (the normal case: the GLMs pad their time span by 1); otherwise the bounds grow here once,
    not at the first call outside them.
    """
    warpings = [m for m in module.modules() if isinstance(m, KumaraswamyWarping)]
    previous = [w._update_buffers for w in warpings]
    for w in warpings:
        w.update_buffers(times)
        w._update_buffers = False
    try:
        yield
    finally:
        for w, value in zip(warpings, previous):
            w._update_buffers = value


def make_mlp(hidden_sizes: Sequence[int] = (128, 128), activation=nn.ReLU) -> nn.Sequential:
    """MLP R -> R with the given hidden layers; ROM-VISDE's KernelNet is make_mlp((128, 128), nn.ReLU)."""
    sizes = (1, *hidden_sizes, 1)
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(n_in, n_out), activation()]
    return nn.Sequential(*layers[:-1])


class NeuralWarping(InputWarping):
    """
    phi(t) = u + net(u), u = (t - offset) / scale, elementwise in t.

    ROM-VISDE rescales the times of a window, relative to its start, to [-1, 1]: use for_window().
    With ReLU, phi is piecewise linear and dphi/dt jumps at its kinks; tanh, softplus or SiLU give a smooth
    derivative. identity_init sets the last layer of net to zero, so that phi(t) = u at the start.

    Args:
        net (nn.Module): R -> R network on inputs (..., 1); default make_mlp(hidden_sizes, activation)
        offset, scale (float): time rescaling
    """

    def __init__(self, net: nn.Module = None, offset: float = 0.0, scale: float = 1.0,
                 hidden_sizes: Sequence[int] = (128, 128), activation=nn.ReLU, identity_init: bool = False):
        super().__init__()
        self.net = make_mlp(hidden_sizes, activation) if net is None else net
        if identity_init:
            last = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.register_buffer("offset", torch.as_tensor(float(offset)))
        self.register_buffer("scale", torch.as_tensor(float(scale)))

    @classmethod
    def for_window(cls, num_window: int, dt: float, **kwargs) -> "NeuralWarping":
        """ROM-VISDE: window times [0, (num_window - 1) dt] map to u in [-1, 1]."""
        half_width = 0.5 * (num_window - 1) * float(dt)
        return cls(offset=half_width, scale=half_width, **kwargs)

    def _phi(self, u: Tensor) -> Tensor:
        return u + self.net(u.unsqueeze(-1)).squeeze(-1)

    def forward(self, t_in: Tensor, return_grad=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        u = (t_in - self.offset) / self.scale
        if not return_grad:
            return self._phi(u)
        # phi acts elementwise, so a JVP with tangent one gives dphi/du for every element
        w, dw_du = torch.func.jvp(self._phi, (u,), (torch.ones_like(u),))
        return w, dw_du / self.scale

    def inverse(self, w: Tensor) -> Tensor:
        raise NotImplementedError("NeuralWarping is not invertible in closed form")
