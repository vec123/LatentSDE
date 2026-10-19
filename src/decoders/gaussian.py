"""Gaussian decoder: p(x | z, mu) = N(MLP([z, mu]), diag s^2), one learned s per output, or s fixed (std_fixed)."""
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn.functional import softplus

from ..core.nn import mlp
from ..core.transforms import inverse_softplus_scalar


class GaussianDecoder(nn.Module):
    """
    p(x | z, mu) = N(MLP([z, mu]), diag s^2), one learned s per output; hidden=() gives a linear mean. std_fixed
    (dim_x values) fixes s, as a buffer that training does not change: a learned s can grow to explain a poor
    posterior fit as noise (runs/simple_multimodal_fullcov_gaps_20261017-152911: s of A 0.97 for the true 0.1).
    """

    def __init__(self, dim_z: int, dim_x: int, dim_mu: int = 0, hidden: Sequence[int] = (), std_init: float = 0.1,
                 std_fixed: Sequence[float] = None):
        super().__init__()
        self.mean = mlp(dim_z + dim_mu, dim_x, hidden)
        if std_fixed is None:
            self.raw_std = nn.Parameter(torch.full((dim_x,), inverse_softplus_scalar(std_init)))
        else:
            if len(std_fixed) != dim_x:
                raise ValueError(f"std_fixed has {len(std_fixed)} values for {dim_x} outputs")
            self.register_buffer("raw_std", torch.tensor([inverse_softplus_scalar(s) for s in std_fixed]))

    def forward(self, z: Tensor, mu: Tensor) -> Normal:
        return Normal(self.mean(torch.cat([z, mu], -1)), softplus(self.raw_std))

    def expected_log_prob(self, x: Tensor, m: Tensor, S: Tensor, mu: Tensor):
        """
        E log p(x | z, mu) for z ~ N(m, S), S (..., dim_z, dim_z), per value (..., dim_x), in closed form for a
        linear mean C z + D mu + d: the log density at m minus (C S C^T)_cc / (2 s_c^2). None with hidden layers.
        """
        if len(self.mean) != 1:
            return None
        C = self.mean[0].weight[:, :m.shape[-1]]
        p = self(m, mu)
        return p.log_prob(x) - 0.5 * ((C @ S) * C).sum(-1) / p.scale.pow(2)


class LatentOnlyDecoder(nn.Module):
    """A decoder that reads the latent state z alone: the static parameters mu that LatentSDE passes (covariates, the
    static latent w) are dropped, so every effect on the observations has to go through z."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, z: Tensor, mu: Tensor):
        return self.decoder(z, mu[..., :0])


class ResidualGaussianDecoder(nn.Module):
    """
    p(x | z, mu) = N(z + MLP([z, mu]), diag s^2) (one MLP per block, see sizes) with a fixed s, for dim_x == dim_z: the identity (the latent is already
    the observed vector, e.g. the frozen MLD latents) plus a correction that depends on mu (a mode one-hot) and starts at
    zero. With a mode as mu, each mode can shift the observed latents, which is what the mode explains.
    """

    def __init__(self, dim_z: int, dim_x: int, dim_mu: int = 0, hidden: Sequence[int] = (64,), std_fixed: float = 0.3,
                 sizes: Sequence[int] = None):
        super().__init__()
        if dim_z != dim_x:
            raise ValueError("a residual decoder needs dim_x == dim_z")
        # sizes: the widths of the modality blocks of z; each block then has its own MLP, which reads only that block and mu,
        # so no modality is reconstructed from another's latent (MLD: modality-specific decoders)
        self.sizes = list(sizes) if sizes else [dim_z]
        if sum(self.sizes) != dim_z:
            raise ValueError("the block sizes must add up to dim_z")
        self.nets = nn.ModuleList([mlp(s + dim_mu, s, tuple(hidden)) for s in self.sizes])
        for net in self.nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.register_buffer("raw_std", torch.full((dim_x,), inverse_softplus_scalar(std_fixed)))

    def forward(self, z: Tensor, mu: Tensor) -> Normal:
        corrections = [net(torch.cat([zb, mu], -1)) for net, zb in zip(self.nets, z.split(self.sizes, -1))]
        return Normal(z + torch.cat(corrections, -1), softplus(self.raw_std))
