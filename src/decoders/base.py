"""Shared shape of the decoders in this package, so src/latent/model.py can take any of them."""
from typing import Protocol, runtime_checkable

from torch import Tensor
from torch.distributions import Distribution


@runtime_checkable
class Decoder(Protocol):
    def forward(self, z: Tensor, mu: Tensor) -> Distribution:
        """p(x | z, mu), for z (..., dim_z) and mu (..., dim_mu)."""
        ...
