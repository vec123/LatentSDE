"""
A multimodal frame encoder: one small network per modality block of the columns, the tokens concatenated.

Each block of x (B, N, D_m) with its mask goes through its own MLP to a token of d_token values; a block with no
observed value at a step gives the zero token. The features are [token_1, ..., token_M, presence_1, ..., presence_M]
(B, N, M * (d_token + 1)): the concatenation keeps the modalities apart, so the posterior can tell which modality
spoke at a step. src/latent/model.py gives the mask to an encoder with uses_mask = True.
"""
from typing import Sequence

import torch
from torch import Tensor, nn

from ..core.nn import mlp


class MultimodalEncoder(nn.Module):
    uses_mask = True

    def __init__(self, sizes: Sequence[int], d_token: int = 8, hidden: Sequence[int] = (32,)):
        super().__init__()
        self.sizes, self.d_token = list(sizes), d_token
        self.nets = nn.ModuleList([mlp(2 * d, d_token, tuple(hidden)) for d in self.sizes])
        self.dim_feat = len(self.sizes) * (d_token + 1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        tokens, present = [], []
        for net, xb, mb in zip(self.nets, x.split(self.sizes, -1), mask.split(self.sizes, -1)):
            seen = (mb.sum(-1, keepdim=True) > 0).to(x.dtype)
            tokens.append(net(torch.cat([xb * mb, mb], -1)) * seen)
            present.append(seen)
        return torch.cat(tokens + present, -1)
