"""
Frame encoders for gridded data (e.g. test_data/data/reaction_diffusion): LatentSDE's optional encoder argument,
for observations too large to hand a posterior directly (a 100x100 frame flattens to 10000 numbers; feeding
that straight into SequencePosterior's GRU or KernelPosterior's per-node MLP is impractical and ignores the
grid's spatial structure). LatentSDE runs this on the raw frame once per batch and hands the result, not the
raw frame, to whichever posterior (src/encoders) is doing the actual temporal modelling; the decoder still
reconstructs the raw frame, never the encoder's output, so the encoder only has to carry enough information for
the drift residual and the reconstruction's gradient to flow back through it, not reconstruct frames itself.

ResidualUNetEncoder is a trainable CNN: the contracting path of a residual U-Net (a stem conv, then stages of a
residual block followed by a strided downsample, global average pooling, a linear head), so it is trained
jointly with the rest of LatentSDE through the ELBO.

PCAFrameEncoder is the fixed alternative: a PCA (src/data/pca.py), created once from the training frames before
training starts, wrapped so it has the same call signature as ResidualUNetEncoder. It has no Parameters, only
the PCA's buffers, so it never changes during training.
"""
from typing import Sequence, Tuple

from torch import Tensor, nn

from ..data.pca import PCA


class ResidualBlock(nn.Module):
    """Two 3x3 convs with GroupNorm and SiLU, and a residual connection, channels unchanged."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        groups = min(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class ResidualUNetEncoder(nn.Module):
    """
    The contracting path of a residual U-Net, used as a frame encoder: stem conv to channels[0], then for each
    further width in channels a residual block followed by a stride-2 downsample, global average pooling, and a
    linear head to dim_feat. shape is the frame's spatial shape (H, W); input is flattened like make_batch's x,
    (..., H * W), and reshaped here.
    """

    def __init__(self, shape: Tuple[int, int], dim_feat: int, channels: Sequence[int] = (16, 32, 64)):
        super().__init__()
        self.shape = shape
        self.stem = nn.Conv2d(1, channels[0], 3, padding=1)
        stages = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            stages += [ResidualBlock(c_in), nn.Conv2d(c_in, c_out, 4, stride=2, padding=1)]
        stages.append(ResidualBlock(channels[-1]))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(channels[-1], dim_feat)

    def forward(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        h = self.stem(x.reshape(-1, 1, *self.shape))
        h = self.stages(h).mean((-2, -1))
        return self.head(h).reshape(*lead, -1)


class PCAFrameEncoder(nn.Module):
    """A fitted PCA (src/data/pca.py), as a frame encoder: forward(x) = pca.encode(x). Not trainable."""

    def __init__(self, pca: PCA):
        super().__init__()
        self.pca = pca

    def forward(self, x: Tensor) -> Tensor:
        return self.pca.encode(x)
