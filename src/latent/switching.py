"""
A latent SDE with one dynamical mode per subpopulation.

Each patient has a discrete static latent c in {0, .., K-1}, a priori uniform (or pi), that enters as a one-hot in
the parameters mu of the base LatentSDE: the drift f(z, mu, u), and the posterior, read it, so each mode has its own
drift and its own posterior of the path. The base model is trained on the exact marginal over c,

    - log sum_c pi_c exp(ELBO_c(x)),

with ELBO_c the evidence bound of the trajectory x under mode c (LatentSDE.loss(per_trajectory=True)); a bound on
log p(x) as it is, as log sum_c pi_c p(x | c) >= log sum_c pi_c exp(ELBO_c). The mode of a trajectory is the most
likely one, argmax_c pi_c exp(ELBO_c): the SDE whose dynamics explain the path best. There is no separate
classifier and no clustering step. The cost per step is K times that of the base model.
"""
import torch
from torch import Tensor, nn
from torch.nn.functional import log_softmax, one_hot

from .model import LatentSDE


class SwitchingLatentSDE(nn.Module):
    """base: a LatentSDE built with dim_mu = (the batch's mu) + n_modes. The attributes the evaluation reads
    (encoder, posterior, decoder, drift, diffusion, static, z0_mean) are those of the base."""

    def __init__(self, base: LatentSDE, n_modes: int, learn_pi: bool = False):
        super().__init__()
        self.base, self.n_modes = base, n_modes
        logits = torch.zeros(n_modes)
        if learn_pi:
            self.pi_logits = nn.Parameter(logits)
        else:
            self.register_buffer("pi_logits", logits)

    encoder = property(lambda self: self.base.encoder)
    posterior = property(lambda self: self.base.posterior)
    decoder = property(lambda self: self.base.decoder)
    drift = property(lambda self: self.base.drift)
    diffusion = property(lambda self: self.base.diffusion)
    static = property(lambda self: self.base.static)
    z0_mean = property(lambda self: self.base.z0_mean)

    def with_mode(self, batch: dict, c: Tensor) -> dict:
        """The batch with the one-hot of the mode c (B,) appended to mu."""
        return {**batch, "mu": torch.cat([batch["mu"], one_hot(c, self.n_modes).to(batch["mu"].dtype)], -1)}

    def mode_scores(self, batch: dict, beta: float = 1.0) -> Tensor:
        """(K, B): log pi_c - (negative ELBO of the trajectory under mode c)."""
        B = batch["t"].shape[0]
        nelbo = torch.stack([
            self.base.loss(self.with_mode(batch, torch.full((B,), c, dtype=torch.long, device=batch["t"].device)),
                           beta, per_trajectory=True)[0] for c in range(self.n_modes)])
        return log_softmax(self.pi_logits, 0)[:, None] - nelbo

    def loss(self, batch: dict, beta: float = 1.0, diffusion_penalty: float = 0.0):
        scores = self.mode_scores(batch, beta)
        n = batch["mask"].sum()
        loss = -torch.logsumexp(scores, 0).sum() / n
        usage = torch.softmax(scores, 0).mean(1)  # the share of the trajectories each mode explains best
        return loss, {"nelbo": loss.detach(), "usage_max": usage.max(), "usage_min": usage.min()}

    @torch.no_grad()
    def assign(self, batch: dict) -> Tensor:
        """The most likely mode of each trajectory (B,)."""
        return self.mode_scores(batch).argmax(0)

    def conditioned(self, batch: dict) -> dict:
        """The batch with the one-hot of its most likely mode appended to mu: the input of the base model."""
        return self.with_mode(batch, self.assign(batch))
