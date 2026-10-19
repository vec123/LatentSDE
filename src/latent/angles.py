"""
Angle dimensions of the latent state: the drift and the decoder see cos and sin of them, not the value, so they
are periodic in those dimensions, while the state itself (and so the posterior's path) keeps the unwrapped angle.

For a latent state that turns (a phase, a cycle), z = (cos phi, sin phi) oscillates as fast as the cycle, and a
posterior that is smooth between sparse observations cannot follow it; the unwrapped angle phi grows smoothly at
the rotation rate instead, and src/encoders/integrated.py's posterior gives it as the integral of that rate.
"""
from typing import Sequence

import torch
from torch import Tensor, nn


class AngleEmbedding(nn.Module):
    """z (..., dim_z) -> (..., dim_z + len(angles)): cos and sin of the angle dimensions, then the others."""

    def __init__(self, dim_z: int, angles: Sequence[int]):
        super().__init__()
        self.angles = sorted(set(angles))
        if not self.angles or self.angles[0] < 0 or self.angles[-1] >= dim_z:
            raise ValueError(f"angles {list(angles)} must be indices of the {dim_z} latent dimensions")
        self.others = [i for i in range(dim_z) if i not in self.angles]
        self.dim_out = dim_z + len(self.angles)

    def forward(self, z: Tensor) -> Tensor:
        a = z[..., self.angles]
        return torch.cat([torch.cos(a), torch.sin(a), z[..., self.others]], -1)


class Embedded(nn.Module):
    """module(embed(z), *rest): the drift f(z, mu, u) or the decoder p(x | z, mu), reading the embedded state."""

    def __init__(self, embed: AngleEmbedding, module: nn.Module):
        super().__init__()
        self.embed, self.module = embed, module

    def forward(self, z: Tensor, *rest):
        return self.module(self.embed(z), *rest)
