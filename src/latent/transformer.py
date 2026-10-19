"""
A transformer drift f(z, mu, u): the dimensions of the latent state attend to each other.

Each latent dimension i is a token: its value z_i through a linear map, plus a learned embedding of the dimension (which
modality block it belongs to is learned from that). Two more tokens carry the conditioning, one linear map of the static
parameters mu (the covariates) and one of the forcing u (the treatment). The tokens go through pre-norm transformer
encoder layers (self-attention among all of them, so the latent dimensions, and the conditioning, interact), and a
linear head on the tokens of the latent dimensions gives the drift of each: one scalar per dimension. The head starts
small, so the drift starts near zero.

Same call as DriftNet: z (..., dim_z), mu (..., dim_mu), u (..., dim_u) with the same leading shape -> (..., dim_z).
"""
import torch
from torch import Tensor, nn


class TransformerDrift(nn.Module):
    def __init__(self, dim_z: int, dim_mu: int = 0, dim_u: int = 0, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4):
        super().__init__()
        self.dim_z, self.d_model = dim_z, d_model
        self.value = nn.Linear(1, d_model)
        self.position = nn.Parameter(0.02 * torch.randn(dim_z, d_model))
        self.mu_token = nn.Linear(dim_mu, d_model) if dim_mu else None
        self.u_token = nn.Linear(dim_u, d_model) if dim_u else None
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, z: Tensor, mu: Tensor, u: Tensor) -> Tensor:
        lead = z.shape[:-1]
        tokens = self.value(z.reshape(-1, self.dim_z).unsqueeze(-1)) + self.position  # (M, dim_z, d_model)
        extra = []
        if self.mu_token is not None:
            extra.append(self.mu_token(mu.reshape(-1, mu.shape[-1])).unsqueeze(1))
        if self.u_token is not None:
            extra.append(self.u_token(u.reshape(-1, u.shape[-1])).unsqueeze(1))
        h = self.encoder(torch.cat(extra + [tokens], 1))
        return self.head(h[:, len(extra):]).squeeze(-1).reshape(*lead, self.dim_z)
