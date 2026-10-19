"""
Decoders for gridded data (e.g. test_data/data/reaction_diffusion), matching encoders/field.py's frame encoders.

ResidualUNetDecoder is the expanding path for ResidualUNetEncoder: a linear head from [z, mu] to a small
bottleneck feature map, computed from shape and channels the same way the encoder's stride-2 convs shrink it,
then the same widths read in reverse, each followed by a nearest-neighbour upsample, a final conv to one
channel, and a resize to the exact target shape (a safety net against any rounding in the stride-2/upsample
chain). Trainable, like ResidualUNetEncoder.

PCADecoder is the fixed alternative, matching PCAFrameEncoder. z is the SDE's latent state, not the PCA code
(PCAFrameEncoder's output goes through the posterior's GRU before it ever becomes z), so PCADecoder first learns
a linear map from [z, mu] to the PCA code, then applies the fixed PCA basis: p(x | z, mu) =
N(pca.decode(Linear([z, mu])), diag s^2). Only that linear map and the per-pixel std are trainable.

PCADecoder takes the PCAFrameEncoder itself, not a bare PCA, and decodes through its exact pca (same object, not
just an equal one): decoding through any other PCA, even one fitted on the same data, silently decodes in the
wrong basis.
"""
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn.functional import interpolate, softplus

from ..core.transforms import inverse_softplus_scalar
from ..encoders.field import PCAFrameEncoder, ResidualBlock


def downsampled_shape(shape: Tuple[int, int], n_stages: int) -> Tuple[int, int]:
    """Spatial shape after n_stages of ResidualUNetEncoder's stride-2, kernel-4, padding-1 downsamples."""
    h, w = shape
    for _ in range(n_stages):
        h, w = (h - 2) // 2 + 1, (w - 2) // 2 + 1
    return h, w


class UpBlock(nn.Module):
    """Nearest-neighbour upsample by 2, then a 3x3 conv changing the channel count."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.up(x))


class ResidualUNetDecoder(nn.Module):
    """p(x | z, mu) = N(mean, diag s^2), mean a small residual U-Net decoder, one learned s per pixel."""

    def __init__(self, dim_z: int, shape: Tuple[int, int], dim_mu: int = 0, channels: Sequence[int] = (16, 32, 64),
                 std_init: float = 0.1):
        super().__init__()
        self.shape = shape
        channels = tuple(reversed(channels))
        self.bottleneck = downsampled_shape(shape, len(channels) - 1)
        self.channels0 = channels[0]
        self.proj = nn.Linear(dim_z + dim_mu, channels[0] * self.bottleneck[0] * self.bottleneck[1])
        stages = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            stages += [ResidualBlock(c_in), UpBlock(c_in, c_out)]
        stages.append(ResidualBlock(channels[-1]))
        self.stages = nn.Sequential(*stages)
        self.out = nn.Conv2d(channels[-1], 1, 3, padding=1)
        self.raw_std = nn.Parameter(torch.full((shape[0] * shape[1],), inverse_softplus_scalar(std_init)))

    def forward(self, z: Tensor, mu: Tensor) -> Normal:
        lead = z.shape[:-1]
        h = self.proj(torch.cat([z, mu], -1)).reshape(-1, self.channels0, *self.bottleneck)
        h = self.stages(h)
        if tuple(h.shape[-2:]) != tuple(self.shape):
            h = interpolate(h, size=self.shape, mode="nearest")
        mean = self.out(h).reshape(*lead, -1)
        return Normal(mean, softplus(self.raw_std))


class PCADecoder(nn.Module):
    """
    p(x | z, mu) = N(pca.decode(Linear([z, mu])), diag s^2), one learned s per pixel. The linear map from the
    SDE's latent state to the PCA code is trainable; the PCA basis itself is fixed.

    Args:
        encoder (PCAFrameEncoder): the encoder this decoder matches; decodes through encoder.pca itself, so the
            two are always the same PCA model.
    """

    def __init__(self, encoder: PCAFrameEncoder, dim_z: int, dim_mu: int = 0, std_init: float = 0.1):
        super().__init__()
        self.pca = encoder.pca  # the encoder's own PCA, registered again here: shared, not copied
        self.to_code = nn.Linear(dim_z + dim_mu, self.pca.evecs.shape[1])
        self.raw_std = nn.Parameter(torch.full((self.pca.evecs.shape[0],), inverse_softplus_scalar(std_init)))

    def forward(self, z: Tensor, mu: Tensor) -> Normal:
        return Normal(self.pca.decode(self.to_code(torch.cat([z, mu], -1))), softplus(self.raw_std))
