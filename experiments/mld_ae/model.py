"""
ModalityAE: the deterministic autoencoder of one modality, as in MLD. The encoder gives z = e(x), the decoder
x_hat = d(z), and the loss is the reconstruction loss of the modality only (eq. 1 of the paper; no KL term, no
regularization of z). Encoder and decoder are MLPs of Linear, BatchNorm1d and LeakyReLU layers, as the MLD
autoencoders of the MHD trajectory and the CelebA attributes (reference/MLD/src/unimodal).

The rows have missing values (a biomarker sample has no nodule count). The encoder reads, for each column, the
value (0 where missing) and its mask; for a censored column also the BLQ flag; for a categorical column the
one-hot of the level and the mask. The loss counts only the observed values. The decoder gives one output per
scalar column and one logit per level of a categorical column. The loss of a row is the sum over its observed
values, and the loss of a batch is the mean over its rows (the MLD "sum / batch_size"):

    normal, count  (x - y)^2
    censored       (x - y)^2, or relu(y - x)^2 below the LOD (x then holds the LOD)
    binary         binary cross entropy of the logit y
    categorical    cross entropy of the logits

z_mean and z_std are the mean and std of z on the train rows. MLD standardizes each latent with them before the
diffusion stage (reference/MLD/src/models/MLD.py, normalize).
"""
from dataclasses import asdict
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .data import Column


def mlp(dims: Sequence[int]) -> nn.Sequential:
    """Linear, BatchNorm1d, LeakyReLU for each hidden layer, then a last Linear."""
    layers = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.LeakyReLU()]
    return nn.Sequential(*layers, nn.Linear(dims[-2], dims[-1]))


class ModalityAE(nn.Module):
    def __init__(self, columns: Sequence[Column], latent_dim: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.columns, self.latent_dim, self.hidden = list(columns), latent_dim, tuple(hidden)
        kinds = [c.kind for c in columns]
        self.scalar = [j for j, k in enumerate(kinds) if k != "categorical"]
        self.categorical = [j for j, k in enumerate(kinds) if k == "categorical"]
        self.censored = [j for j, k in enumerate(kinds) if k == "censored"]
        self.levels = [len(columns[j].levels) for j in self.categorical]
        self.register_buffer("binary", torch.tensor([kinds[j] == "binary" for j in self.scalar], dtype=torch.bool))
        in_dim = 2 * len(self.scalar) + len(self.censored) + sum(k + 1 for k in self.levels)
        out_dim = len(self.scalar) + sum(self.levels)
        self.encoder = mlp([in_dim, *hidden, latent_dim])
        self.decoder = mlp([latent_dim, *hidden[::-1], out_dim])
        self.register_buffer("z_mean", torch.zeros(latent_dim))
        self.register_buffer("z_std", torch.ones(latent_dim))

    def inputs(self, x: Tensor, blq: Tensor) -> Tensor:
        """x (B, D) in the model space with NaN where missing, blq (B, D) bool -> the encoder input."""
        mask = torch.isfinite(x)
        xs, ms = x[:, self.scalar], mask[:, self.scalar]
        parts = [torch.where(ms, xs, 0.0), ms.to(x.dtype), blq[:, self.censored].to(x.dtype)]
        for j, k in zip(self.categorical, self.levels):
            m = mask[:, j]
            level = torch.where(m, x[:, j], 0.0).long()
            parts += [F.one_hot(level, k).to(x.dtype) * m[:, None], m[:, None].to(x.dtype)]
        return torch.cat(parts, -1)

    def encode(self, x: Tensor, blq: Tensor) -> Tensor:
        return self.encoder(self.inputs(x, blq))

    def decode(self, z: Tensor) -> Tensor:
        """z (B, latent_dim) -> the decoder output (B, out_dim): scalar outputs, then the logits of each
        categorical column."""
        return self.decoder(z)

    def forward(self, x: Tensor, blq: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x, blq)
        return self.decode(z), z

    def _split(self, out: Tensor) -> Tuple[Tensor, List[Tensor]]:
        s = len(self.scalar)
        return out[:, :s], list(torch.split(out[:, s:], self.levels, dim=-1)) if self.levels else []

    def loss(self, out: Tensor, x: Tensor, blq: Tensor) -> Tensor:
        """The reconstruction loss of each row (B,), summed over its observed values."""
        y, logits = self._split(out)
        mask = torch.isfinite(x)
        xs, ms, below = torch.nan_to_num(x[:, self.scalar]), mask[:, self.scalar], blq[:, self.scalar]
        square = torch.where(below, F.relu(y - xs), y - xs) ** 2
        bce = F.binary_cross_entropy_with_logits(y, xs, reduction="none")
        total = (torch.where(self.binary, bce, square) * ms).sum(-1)
        for j, lg in zip(self.categorical, logits):
            m = mask[:, j]
            ce = F.cross_entropy(lg, torch.where(m, x[:, j], 0.0).long(), reduction="none")
            total = total + ce * m
        return total

    def predict(self, out: Tensor) -> Tensor:
        """The reconstruction (B, D) in the model space: the value of a normal, count or censored column, the
        probability of a binary column, the most probable level (index) of a categorical column."""
        y, logits = self._split(out)
        pred = out.new_zeros(out.shape[0], len(self.columns))
        pred[:, self.scalar] = torch.where(self.binary, torch.sigmoid(y), y)
        for j, lg in zip(self.categorical, logits):
            pred[:, j] = lg.argmax(-1).to(out.dtype)
        return pred


def save(path: str, model: ModalityAE, **extra) -> None:
    torch.save({"columns": [asdict(c) for c in model.columns], "latent_dim": model.latent_dim,
                "hidden": list(model.hidden), "state": model.state_dict(), **extra}, path)


def load(path: str) -> Tuple[ModalityAE, dict]:
    """The model (in eval mode) and the full checkpoint dict."""
    ckpt = torch.load(path, weights_only=True)
    cols = [Column(**{**c, "levels": tuple(c["levels"])}) for c in ckpt["columns"]]
    model = ModalityAE(cols, ckpt["latent_dim"], ckpt["hidden"])
    model.load_state_dict(ckpt["state"])
    return model.eval(), ckpt
