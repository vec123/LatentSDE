"""
A latent SDE with a discrete mode c (a subpopulation) that only the decoder reads.

The mode is predicted, never given:

    prior      pi(c | mu, path)      from the static parameters mu (covariates) and a summary of the posterior
                                     path m(t) of the latent state z (its level, spread and slope over the visits)
    posterior  q(c | mu, x, path)    exact, given the posterior path of the latent state z: pi(c | mu, path) times the
                                     evidence exp(E_q(z) sum_k log p(x_k | z_k, c)) of every visit and observed
                                     column, summed over the visits (not averaged) and weighted by the noise of
                                     each column; a missing value adds nothing, so a patient with few modalities
                                     gets a mode from the values that exist

The drift f(z, mu, u), the diffusion and the posterior of the path do not read c: they run once per batch. The decoder
p(x | z, c) reads the latent state and the mode, so each mode can have its own offsets in the observations (the body
sites, the marker levels of a subtype). The loss is the ELBO with c summed out exactly (K decoder passes only):

    sum_c q(c) E_q(z) log p(x | z, c)  -  KL(q(c) || pi(c | mu))  -  drift residual  -  KL(q(z0) || p(z0)).

With balance > 0 the loss adds balance * (log K - H(q_mean)), q_mean the mean of q(c) over the batch: an entropy
penalty on the usage of the modes that keeps one from dying (the posterior of one patient may still be sharp).

The mode of a patient is argmax_c q(c); there is no network for q(c), only the prior pi(c | mu, path).
"""
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn.functional import log_softmax

from ..core.nn import mlp
from .batch import interp
from .diffusion import whiten
from .model import LatentSDE, posterior_drift


class ModeDecoder(nn.Module):
    """decoder(z, mode one-hot): of the mu a caller passes (any width), only the last K columns, the one-hot of c."""

    def __init__(self, decoder: nn.Module, n_modes: int):
        super().__init__()
        self.decoder, self.n_modes = decoder, n_modes

    def forward(self, z: Tensor, mu: Tensor):
        return self.decoder(z, mu[..., -self.n_modes:])


class ModeDrift(nn.Module):
    """drift(z, mu, u) of the first dim_mu columns of the mu a caller passes (the covariates, not the mode)."""

    def __init__(self, drift: nn.Module, dim_mu: int):
        super().__init__()
        self.drift, self.dim_mu = drift, dim_mu

    def forward(self, z: Tensor, mu: Tensor, u: Tensor) -> Tensor:
        return self.drift(z, mu[..., :self.dim_mu], u)


class ModeHead(nn.Module):
    """
    The prior of the mode, pi(c | mu, path): from the static parameters mu (all the covariates) and a summary of the
    posterior path m(t) of the latent state: the mean, the spread and the slope of m over the observed steps (the
    whole trajectory: its level, its variation and its growth rate), the log number of steps and the span in time.
    The summary is where the severity and the speed of the disease enter the mode; the observations enter through
    the exact evidence in ModeLatentSDE.
    """

    def __init__(self, dim_mu: int, dim_z: int, n_modes: int, hidden: Sequence[int] = (64,)):
        super().__init__()
        self.n_modes = n_modes
        self.prior_net = mlp(dim_mu + 3 * dim_z + 2, n_modes, tuple(hidden))

    def prior_logits(self, mu: Tensor, m: Tensor, t: Tensor, covered: Tensor) -> Tensor:
        """m (B, N, dim_z) the posterior mean at the steps t (B, N); covered (B, N) bool, the observed steps."""
        w = covered.unsqueeze(-1).to(m.dtype)
        n = w.sum(1).clamp_min(1.0)
        mean = (m * w).sum(1) / n
        std = (((m - mean.unsqueeze(1)).pow(2) * w).sum(1) / n + 1e-6).sqrt()
        tc = (t - (t * w.squeeze(-1)).sum(1, keepdim=True) / n).unsqueeze(-1)  # centred times (years)
        slope = (w * tc * (m - mean.unsqueeze(1))).sum(1) / ((w * tc.pow(2)).sum(1) + 1e-3)
        span = (torch.where(covered, t, -torch.inf).amax(1) - torch.where(covered, t, torch.inf).amin(1)).clamp_min(0)
        return self.prior_net(torch.cat([mean, std, slope, n.log(), span.unsqueeze(-1), mu], -1))


class ModeLatentSDE(LatentSDE):
    """LatentSDE with a ModeHead (self.modes); its decoder is a ModeDecoder and its drift a ModeDrift."""

    def __init__(self, *args, modes: ModeHead, balance: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.modes = modes
        self.balance = balance  # weight of log K - H(mean q(c)), the KL of the mean usage to uniform (per batch)

    def _condition(self, batch: dict):
        mask = batch["mask"]
        x = torch.where(mask > 0, batch["x"], torch.zeros_like(batch["x"]))
        if self.encoder is None:
            feat_mask = self._dropped_blocks(mask)  # modality dropout of the input while training (LatentSDE)
            feat = x * feat_mask
        else:
            feat = self.encoder(x, mask) if getattr(self.encoder, "uses_mask", False) else self.encoder(x)
            feat_mask = mask.any(-1, keepdim=True).expand_as(feat).to(feat.dtype)
        covered = self.posterior.condition({**batch, "x": feat, "mask": feat_mask})
        return x, covered

    def _recon_per_mode(self, z: Tensor, x: Tensor, weight: Tensor) -> Tensor:
        """(K, B): E_q(z) sum over the visits and columns of log p(x | z, c), for each mode c; z (S, B, N, dim_z)."""
        S, B, N, _ = z.shape
        K = self.modes.n_modes
        eye = torch.eye(K, dtype=z.dtype, device=z.device)
        return torch.stack([(self.decoder(z, eye[c].expand(S, B, N, K)).log_prob(x) * weight).sum((-1, -2)).mean(0)
                            for c in range(K)])

    @torch.no_grad()
    def assign(self, batch: dict, n_samples: int = 16) -> Tensor:
        """
        The most likely mode of each trajectory (B,), argmax_c pi(c | mu) exp(E_q(z) sum_k log p(x_k | z_k, c)): the
        evidence of all the visits, with the mask. n_samples draws of z with a fixed seed.
        """
        x, covered = self._condition(batch)
        m, scale, _, _ = self.posterior(batch["t"])
        m, scale = torch.nan_to_num(m), torch.nan_to_num(scale)  # NaN where a patient has no readable value
        gen = torch.Generator(device=m.device).manual_seed(0)
        eps = torch.randn(n_samples, *m.shape, dtype=m.dtype, device=m.device, generator=gen)
        z = m + (0.5 * scale).exp() * eps
        weight = batch["mask"] * covered.unsqueeze(-1)
        log_pi = log_softmax(self.modes.prior_logits(batch["mu"], m, batch["t"], covered), -1)
        return (log_pi + self._recon_per_mode(z, x, weight).T).argmax(-1)

    def loss(self, batch: dict, beta: float = 1.0, diffusion_penalty: float = 0.0):
        t, mu = batch["t"], batch["mu"]
        x, covered = self._condition(batch)
        weight = batch["mask"] * covered.unsqueeze(-1)
        (B, N), S, Q, K = t.shape, self.n_samples, self.n_quad, self.modes.n_modes
        t0 = torch.where(covered, t, torch.inf).amin(1, keepdim=True)
        t1 = torch.where(covered, t, -torch.inf).amax(1, keepdim=True)
        strata = torch.arange(Q, dtype=t.dtype, device=t.device) + torch.rand(B, Q, dtype=t.dtype, device=t.device)
        t_quad = t0 + (t1 - t0) * strata / Q
        m, scale, dm, dscale = self.posterior(torch.cat([t, t_quad, t0], 1))
        if scale.dim() > m.dim():
            raise ValueError("modes need a diagonal-covariance posterior")
        z = m + (0.5 * scale).exp() * torch.randn(S, *m.shape, dtype=m.dtype, device=m.device)

        # the exact posterior of the mode given the path: q(c) = pi(c | mu) exp(E sum_k log p(x_k | z_k, c)) / Z, the
        # evidence of every visit added (not averaged), a mask 0 value adding nothing
        log_pi = log_softmax(self.modes.prior_logits(mu, m[:, :N], t, covered), -1)  # (B, K)
        recon_c = self._recon_per_mode(z[:, :, :N], x, weight)  # (K, B)
        log_q = log_softmax(log_pi + recon_c.T, -1)
        q = log_q.exp()
        kl_c = (q * (log_q - log_pi)).sum(-1)
        recon = (q.T * recon_c).sum(0)

        qs = slice(N, N + Q)
        mu_e = mu[None, :, None].expand(S, B, N + Q + 1, -1)
        z_q, mu_q = z[:, :, qs], mu_e[:, :, qs]
        G, div = self.diffusion.with_divergence(z_q, mu_q, t_quad)
        f_q = posterior_drift(z_q, m[:, qs], dm[:, qs], scale[:, qs].exp(), dscale[:, qs], G, div)
        f = self.drift(z_q, mu_q, interp(t_quad, t, batch["u"]).expand(S, -1, -1, -1))
        residual = 0.5 * whiten(G, f_q - f).pow(2).sum(-1).mean((0, 2)) * (t1 - t0).squeeze(1)
        m0, logv0 = m[:, -1], scale[:, -1]
        kl0 = 0.5 * (self.z0_logvar - logv0 - 1 + (logv0.exp() + (m0 - self.z0_mean).pow(2))
                     / self.z0_logvar.exp()).sum(-1)

        n = weight.sum()
        usage = q.mean(0)
        terms = {"recon": recon.sum() / n, "residual": residual.sum() / n, "kl0": kl0.sum() / n,
                 "kl_c": kl_c.sum() / n, "usage_max": usage.max().detach(), "usage_min": usage.min().detach()}
        loss = -(terms["recon"] - beta * (terms["residual"] + terms["kl0"] + terms["kl_c"]))
        balance = getattr(self, "balance", 0.0)
        if balance:  # an entropy penalty on the mean usage: no mode may die, each patient's q(c) stays free
            terms["balance"] = balance * (torch.log(torch.tensor(float(K))) + (usage * (usage + 1e-9).log()).sum())
            loss = loss + terms["balance"]
        if diffusion_penalty:
            G2 = G.pow(2).sum((-1, -2) if G.dim() > z_q.dim() else -1)
            terms["diffusion_penalty"] = diffusion_penalty * G2.mean()
            loss = loss + terms["diffusion_penalty"]
        return loss, terms
