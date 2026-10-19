"""
The MLD latent SDE trained end to end: the modality autoencoders and the latent SDE are optimized together.

experiments/latent/hs_mld.py encodes the data with frozen autoencoders. Here the autoencoders of runs/mld_ae are modules of
the model (ModalityBank), warm-started from those weights and trained by the same ELBO as everything else:

    raw rows x_m (model space of mld_ae, NaN where missing)
      -> encoder e_m (trainable)              z_obs = [(e_m(x_m) - z_mean_m) / z_std_m]_m   (30 values), the posterior's input
      -> forward SDE-matching posterior       q(z(t)) over the visits (modality dropout while training)
      -> decoder d_m (trainable), z_m + shift(c) -> the likelihood of the raw x_m (the loss of mld_ae/model.py as the
         negative log-likelihood: squared error, cross entropies; a modality row counts only where it was measured)

The mode c, the transformer drift, the balance term and the exact evidence are those of src/latent/modes.py and
src/latent/transformer.py: c acts on the observations by a learned shift of the latent before the decoders (zero at the
start), and its posterior is pi(c | mu, path) times the exact evidence of all visits. The BatchNorm layers of the
autoencoders stay in eval mode (their running statistics from stage 1): a batch of padded, partly missing rows is a bad
sample for batch statistics.

The batches need the raw rows: hs_mld.load_hs_mld(..., raw=True) adds raw_<modality> (B, N, D_m, NaN where missing) and
blq_<modality> to the batch.
"""
import os
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn.functional import softplus

from experiments.mld_ae import data as mld_data
from experiments.mld_ae import model as mld_model
from src.core.transforms import inverse_softplus_scalar
from src.latent.modes import ModeLatentSDE


class ModalityBank(nn.Module):
    """The autoencoders of the modalities, in order; their latents are concatenated into one vector of sum(sizes)."""

    def __init__(self, mld_dir: str, modalities: Sequence[str]):
        super().__init__()
        self.names = list(modalities)
        self.aes = nn.ModuleList([mld_model.load(os.path.join(mld_dir, m, "model.pt"))[0] for m in self.names])
        self.sizes = [ae.latent_dim for ae in self.aes]

    def encode(self, batch: dict) -> Tuple[Tensor, Tensor]:
        """The standardized latents (B, N, D), 0 where a modality was not measured, and their block mask (B, N, D)."""
        zs, ms = [], []
        for name, ae in zip(self.names, self.aes):
            x, blq = batch[f"raw_{name}"], batch[f"blq_{name}"] > 0
            present = torch.isfinite(x).any(-1)
            z = x.new_zeros(*present.shape, ae.latent_dim)
            if present.any():
                z[present] = (ae.encode(x[present], blq[present]) - ae.z_mean) / ae.z_std
            zs.append(z)
            ms.append(present.unsqueeze(-1).expand_as(z).to(x.dtype))
        return torch.cat(zs, -1), torch.cat(ms, -1)

    def log_prob(self, z: Tensor, batch: dict) -> Tensor:
        """z (S, B, N, D) standardized latents -> (S, B, N): the log-likelihood of the raw rows of all modalities present
        at the step (minus the loss of mld_ae/model.py: squared error, cross entropies)."""
        S = z.shape[0]
        out = torch.zeros(z.shape[:3], dtype=z.dtype, device=z.device)
        a = 0
        for name, ae in zip(self.names, self.aes):
            b = a + ae.latent_dim
            x, blq = batch[f"raw_{name}"], batch[f"blq_{name}"] > 0
            present = torch.isfinite(x).any(-1)
            if present.any():
                raw = z[:, present][..., a:b] * ae.z_std + ae.z_mean  # (S, R, latent)
                R = raw.shape[1]
                loss = ae.loss(ae.decode(raw.reshape(S * R, -1)), x[present].repeat(S, 1),
                               blq[present].repeat(S, 1)).reshape(S, R)
                out[:, present] = out[:, present] - loss
            a = b
        return out

    @torch.no_grad()
    def decode_units(self, name: str, z: np.ndarray) -> Tuple[np.ndarray, list]:
        """Standardized latents of a modality (n, latent_dim) -> its values in the data units and the column names."""
        ae = self.aes[self.names.index(name)]
        raw = torch.as_tensor(z, dtype=torch.float32, device=ae.z_mean.device) * ae.z_std + ae.z_mean
        pred = ae.predict(ae.decode(raw)).cpu().numpy()
        return mld_data.to_units(pred, ae.columns), [c.name for c in ae.columns]


class OffsetDecoder(nn.Module):
    """N(z + shift(c), diag s^2): the mode's shift of the latent, zero at the start; mu is the mode one-hot. Only a
    stand-in for the evaluation (forecasts in latent space); the training likelihood is ModalityBank.log_prob."""

    def __init__(self, dim_z: int, n_modes: int, std: float = 0.3):
        super().__init__()
        self.offset = nn.Linear(n_modes, dim_z, bias=False)
        nn.init.zeros_(self.offset.weight)
        self.register_buffer("raw_std", torch.full((dim_z,), inverse_softplus_scalar(std)))

    def forward(self, z: Tensor, mu: Tensor) -> Normal:
        return Normal(z + self.offset(mu), softplus(self.raw_std))


class EndToEndModeSDE(ModeLatentSDE):
    """ModeLatentSDE whose observations are the raw rows: the encoders and decoders are in `bank`, and trained."""

    def __init__(self, bank: ModalityBank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bank = bank

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_batch", None)  # the raw rows of the last call (on the GPU): not part of the saved model
        return state

    def _condition(self, batch: dict):
        self._batch = batch  # the raw rows, for _recon_per_mode in the same call
        z, mask = self.bank.encode(batch)
        feat_mask = self._dropped_blocks(mask)  # modality dropout of the posterior's input, while training
        covered = self.posterior.condition({**batch, "x": z * feat_mask, "mask": feat_mask})
        return z * mask, covered

    def _recon_per_mode(self, z: Tensor, x: Tensor, weight: Tensor) -> Tensor:
        """(K, B): per mode, E_q(z) of the log-likelihood of the raw rows of the covered steps, with z shifted by the mode."""
        K = self.modes.n_modes
        eye = torch.eye(K, dtype=z.dtype, device=z.device)
        covered = (weight.sum(-1) > 0).to(z.dtype)  # (B, N): a covered step with data
        rows = []
        for c in range(K):
            shift = self.decoder.decoder.offset(eye[c])
            rows.append((self.bank.log_prob(z + shift, self._batch) * covered).sum(-1).mean(0))
        return torch.stack(rows)

    @torch.no_grad()
    def with_encoded(self, batch: dict) -> dict:
        """The batch with x the latents of the current encoders (for the evaluation, which reads batch["x"])."""
        z, mask = self.bank.encode(batch)
        return {**batch, "x": z * mask}
