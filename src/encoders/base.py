"""Shared shape of the posteriors in this package, so src/latent/model.py can take any of them."""
from typing import Protocol, Tuple, runtime_checkable

from torch import Tensor


@runtime_checkable
class Posterior(Protocol):
    """
    condition(batch) once per batch, then forward(t) at any times. A posterior can also have gauge(t), the
    skew-symmetric gauge generator (B, Q, dim_z, dim_z) of its drift (see SequencePosterior); without it, the
    gauge is zero.
    """

    def condition(self, batch: dict) -> Tensor:
        """(B, N) bool: True where batch["t"] covers a time the posterior can be evaluated at."""
        ...

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """m, log v, dm/dt, dlog v/dt at t (B, Q), each (B, Q, dim_z); or, for full covariance N(m, L L^T), m, L,
        dm/dt, dL/dt with L (B, Q, dim_z, dim_z) lower triangular (SequencePosterior(full_cov=True))."""
        ...


@runtime_checkable
class FrameEncoder(Protocol):
    """A per-frame feature map in front of a Posterior, e.g. field.py's ResidualUNetEncoder or PCAFrameEncoder."""

    def __call__(self, x: Tensor) -> Tensor:
        """x (..., prod(shape)) -> (..., dim_feat), the same leading shape."""
        ...
