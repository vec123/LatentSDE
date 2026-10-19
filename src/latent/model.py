"""
Latent SDE with an encoder and a decoder, trained without SDE simulation.

The latent state follows dz = f(z, mu, u(t)) dt + G dW, with parameters mu and a forcing u(t) for each trajectory
(ROM-VISDE). An observation x_k at time t_k has the decoder density p(x_k | z(t_k), mu). The posterior
q(z(t)) = N(m(t), diag v(t)), or N(m(t), L L^T) with a full covariance, comes from an encoder of the observations.
The loss is the negative ELBO (Course & Nair 2023, Bartosh et al. 2025), divided by the number of observed values:

    - sum_k E_q log p(x_k | z(t_k), mu)        reconstruction
    + int E_q |G^-1 (f_q - f)|^2 / 2 dt         drift residual; f_q is posterior_drift (_full)
    + KL(q(z(t_0)) || p(z_0))                  initial state

Stratified uniform times estimate the integral; reparametrized samples estimate the expectations.
The drift has no time input: put t into a column of u for an explicit time dependence.

The posterior, the drift, the decoder and the diffusion are swappable nn.Modules: src/encoders.Posterior
(KernelPosterior, ROM-VISDE; or SequencePosterior, SDE matching), src/decoders.Decoder (GaussianDecoder), and
src/latent/diffusion.Diffusion (ConstantDiagonalDiffusion, the default; NetDiagonalDiffusion, diagonal and
dependent on the state, as in SDE matching; or NetCholeskyDiffusion, full and dependent on the state). A posterior
with a gauge(t) method (SequencePosterior(gauge=True)) also learns the gauge of the posterior drift (Gauge
Matching, arXiv 2610.04390); the marginals, and so the reconstruction and the KL term, do not depend on it.
A structured posterior (src/posterior/kalman.py's KalmanPosterior) reads the prior itself (attach) and gives its
exact drift (exact_drift), which the residual then uses in place of posterior_drift. With a linear drift (Koopman
style) it also gives the residual and the KL term in closed form (closed_form_terms), and the loss then has no
quadrature times and, for a linear decoder, no samples (LatentSDE(closed_form=True), the default; False keeps the
Monte Carlo estimate, e.g. for checks). A nonlinear drift always uses the Monte Carlo estimate.
For any other diffusion (e.g. AffineCholeskyDiffusion), src/posterior/marginal_kalman.py's MarginalKalmanPosterior
gives the smoother's marginals and marginal_drift(z, steps, G, div), the drift of q with the prior's own G(z).
src/posterior/mixture_kalman.py's MixtureKalmanPosterior is a mixture of such smoothers (q not Gaussian); the
loss then takes each expectation over its components (_mixture_loss).
A batch is a dict (make_batch, src/latent/batch.py): t (B, N)
sorted times, x (B, N, D), mask (B, N, D) with 1 for observed values, mu (B, P), u (B, N, F) forcing at t. Pad
short trajectories with mask 0 and the last time repeated.

For gridded observations (e.g. test_data/data/reaction_diffusion), an optional encoder (src/encoders/field.py's
ResidualUNetEncoder or PCAFrameEncoder) maps each raw frame to a feature vector before the posterior reads it;
the posterior and encoder are independent, swappable components, not one combined class. The decoder still
reconstructs the raw observation x, never the encoder's feature: the encoder only has to carry enough
information for the drift residual and the reconstruction's gradient to flow back through it.
"""
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from ..core.nn import mlp
from .batch import interp
from .diffusion import ConstantDiagonalDiffusion, cov_mv, whiten


def posterior_drift(z: Tensor, m: Tensor, dm: Tensor, v: Tensor, dlogv: Tensor, G: Tensor, div: Tensor,
                    omega: Tensor = None) -> Tensor:
    """
    Drift of the SDE with diffusion G and marginals N(m, diag v), in the gauge omega (Gauge Matching, arXiv
    2610.04390, eq. driftwhitened with L = diag(sqrt v)):
    dm + 1/2 dlogv (z - m) + L omega L^-1 (z - m) - 1/2 Sigma(z) (z - m) / v + 1/2 div Sigma(z). It is correct also
    when Sigma depends on z, where the sym gauge is not. omega (..., dim_z, dim_z) is skew-symmetric; None is
    omega = 0, the Cholesky gauge, which for a diagonal Sigma is the same as the sym gauge.
    """
    dz = z - m
    f = dm + 0.5 * (dlogv * dz - cov_mv(G, dz / v) + div)
    if omega is None:
        return f
    s = v.sqrt()
    return f + s * (omega @ (dz / s).unsqueeze(-1)).squeeze(-1)


def posterior_drift_full(z: Tensor, m: Tensor, dm: Tensor, L: Tensor, dL: Tensor, G: Tensor, div: Tensor,
                         omega: Tensor = None) -> Tensor:
    """
    posterior_drift for full-covariance marginals N(m, L L^T), L (..., dim_z, dim_z) lower triangular:
    dm + dL L^-1 (z - m) + L omega L^-1 (z - m) - 1/2 Sigma(z) (L L^T)^-1 (z - m) + 1/2 div Sigma(z). The flow
    part dm + (dL L^-1 + L omega L^-1)(z - m) moves the marginals, as dL L^T + L dL^T = d(L L^T)/dt and omega is
    skew-symmetric; the rest is the score term of the diffusion. With L = diag(sqrt v) it is posterior_drift.
    """
    shape = torch.broadcast_shapes(z.shape[:-1], L.shape[:-2]) + L.shape[-2:]
    L, dL = L.expand(shape), dL.expand(shape)
    w = torch.linalg.solve_triangular(L, (z - m).unsqueeze(-1), upper=False)  # L^-1 (z - m)
    score = torch.linalg.solve_triangular(L.transpose(-1, -2), w, upper=True).squeeze(-1)  # Sigma^-1 (z - m)
    flow = dL @ w if omega is None else dL @ w + L @ (omega @ w)
    return dm + flow.squeeze(-1) + 0.5 * (div - cov_mv(G, score))


class DriftNet(nn.Module):
    """f(z, mu, u) = MLP([z, mu, u]); dim_in (dim_z by default) is the width of the z it reads, e.g. an embedded z
    (src/latent/angles.py), while it always gives dim_z outputs."""

    def __init__(self, dim_z: int, dim_mu: int = 0, dim_u: int = 0, hidden: Sequence[int] = (64, 64),
                 dim_in: int = None):
        super().__init__()
        self.net = mlp((dim_in or dim_z) + dim_mu + dim_u, dim_z, hidden)

    def forward(self, z: Tensor, mu: Tensor, u: Tensor) -> Tensor:
        return self.net(torch.cat([z, mu, u], -1))


class LatentSDE(nn.Module):
    """
    Posterior, drift f(z, mu, u), decoder p(x | z, mu), a diffusion G(z, mu, t) (diagonal, state-independent by
    default) and a prior N(z0_mean, exp(z0_logvar)) on the initial state. Drift and decoder parameters are point
    estimates.
    """

    def __init__(self, posterior: nn.Module, drift: nn.Module, decoder: nn.Module, dim_z: int,
                 encoder: nn.Module = None, diffusion: nn.Module = None, g_init: float = 0.1, n_samples: int = 8,
                 n_quad: int = 32, closed_form: bool = True, static: nn.Module = None):
        super().__init__()
        self.posterior, self.drift, self.decoder = posterior, drift, decoder
        self.static = static  # StaticEncoder: q(w) per trajectory, w appended to mu (src/encoders/static.py)
        if hasattr(posterior, "attach"):  # a structured posterior reads the prior (src/posterior/kalman.py)
            posterior.attach(self)
        self.encoder = encoder
        self.diffusion = ConstantDiagonalDiffusion(dim_z, g_init) if diffusion is None else diffusion
        self.n_samples, self.n_quad, self.closed_form = n_samples, n_quad, closed_form
        self.z0_mean = nn.Parameter(torch.zeros(dim_z))
        self.z0_logvar = nn.Parameter(torch.zeros(dim_z))

    input_dropout = None  # {"p": float, "sizes": [block widths]}: modality dropout of the posterior's input, see below

    def _dropped_blocks(self, mask: Tensor) -> Tensor:
        """
        The mask the posterior reads while training: each block of columns (a modality) of a trajectory is dropped, at all
        its steps, with probability p (never every block the trajectory has). The reconstruction still counts the dropped
        values, so the posterior must infer a modality from the others, as the conditional generation of MLD does (a
        masked process: the conditioning modalities are the ones that are kept). Off in eval mode, and without
        input_dropout.
        """
        cfg = self.input_dropout
        if cfg is None or not self.training:
            return mask
        sizes, (B, N, D) = cfg["sizes"], mask.shape
        if sum(sizes) != D:
            raise ValueError(f"input_dropout sizes {sizes} do not add up to the {D} columns")
        r = torch.rand(B, len(sizes), device=mask.device)
        keep = r > cfg["p"]
        present = torch.stack([b.sum((1, 2)) > 0 for b in mask.split(sizes, -1)], 1)  # the blocks the trajectory has
        force = torch.where(present, r, torch.full_like(r, torch.inf)).argmin(1)  # one of them always stays
        keep[torch.arange(B, device=mask.device), force] = True
        per = torch.cat([keep[:, i:i + 1].expand(B, s) for i, s in enumerate(sizes)], 1)
        return mask * per.unsqueeze(1).to(mask.dtype)

    def loss(self, batch: dict, beta: float = 1.0, diffusion_penalty: float = 0.0, per_trajectory: bool = False):
        """
        Negative ELBO per observed value, with beta on the residual and the KL term; and the three terms. With
        diffusion_penalty > 0, it also adds diffusion_penalty * E |G|_F^2 (the mean over the posterior samples at the
        quadrature times of tr(G G^T), the noise variance per unit time) as a fourth term, not divided by the number
        of observed values: a penalty for a small diffusion, not a prior (a prior is one term against the residual of
        all trajectories, and so has almost no effect). With closed_form, a posterior whose closed_form_terms gives
        the terms (KalmanPosterior with a linear drift) makes the residual and kl0 exact, see _closed_form_loss.
        """
        t, mu, mask = batch["t"], batch["mu"], batch["mask"]
        x = torch.where(mask > 0, batch["x"], torch.zeros_like(batch["x"]))
        if self.encoder is None:
            feat_mask = self._dropped_blocks(mask)
            feat = x * feat_mask
        else:
            feat = self.encoder(x, mask) if getattr(self.encoder, "uses_mask", False) else self.encoder(x)
            feat_mask = mask.any(-1, keepdim=True).expand_as(feat).to(feat.dtype)
        covered = self.posterior.condition({**batch, "x": feat, "mask": feat_mask})
        weight = mask * covered.unsqueeze(-1)
        if self.static is not None:
            return self._static_loss(batch, x, weight, feat, covered, beta, diffusion_penalty)
        closed = getattr(self.posterior, "closed_form_terms", None) if getattr(self, "closed_form", True) else None
        closed = closed() if closed is not None else None
        if closed is not None:
            return self._closed_form_loss(batch, x, weight, *closed, beta, diffusion_penalty)
        if getattr(self.posterior, "components", None) is not None:  # a mixture (posterior/mixture_kalman.py)
            return self._mixture_loss(batch, x, weight, covered, beta, diffusion_penalty)
        (B, N), S, Q = t.shape, self.n_samples, self.n_quad

        # one posterior call: observation times, stratified times on the covered span, start of the span
        t0 = torch.where(covered, t, torch.inf).amin(1, keepdim=True)
        t1 = torch.where(covered, t, -torch.inf).amax(1, keepdim=True)
        strata = torch.arange(Q, dtype=t.dtype, device=t.device) + torch.rand(B, Q, dtype=t.dtype, device=t.device)
        t_quad = t0 + (t1 - t0) * strata / Q
        # scale: log v (B, ., dim_z), or the Cholesky factor L (B, ., dim_z, dim_z) of a full covariance
        m, scale, dm, dscale = self.posterior(torch.cat([t, t_quad, t0], 1))
        full = scale.dim() > m.dim()
        eps = torch.randn(S, *m.shape, dtype=m.dtype, device=m.device)
        z = m + ((scale @ eps.unsqueeze(-1)).squeeze(-1) if full else (0.5 * scale).exp() * eps)
        mu = mu[None, :, None].expand(S, B, N + Q + 1, -1)

        recon =(self.decoder(z[:, :, :N], mu[:, :, :N]).log_prob(x) * weight).sum((-1, -2)).mean(0)

        q = slice(N, N + Q)
        z_q, mu_q = z[:, :, q], mu[:, :, q]
        G, div = self.diffusion.with_divergence(z_q, mu_q, t_quad)
        gauge = getattr(self.posterior, "gauge", None)
        omega = gauge(t_quad) if gauge is not None else None
        exact_drift = getattr(self.posterior, "exact_drift", None)
        marginal_drift = getattr(self.posterior, "marginal_drift", None)
        if marginal_drift is not None:  # Kalman marginals, drift with the prior's own G (posterior/marginal_kalman.py)
            f_q = marginal_drift(z_q, q, G, div)
        elif exact_drift is not None:  # a structured posterior knows its drift: no gauge, no time derivatives
            f_q = exact_drift(z_q, q)
        elif full:
            f_q = posterior_drift_full(z_q, m[:, q], dm[:, q], scale[:, q], dscale[:, q], G, div, omega)
        else:
            f_q = posterior_drift(z_q, m[:, q], dm[:, q], scale[:, q].exp(), dscale[:, q], G, div, omega)
        f = self.drift(z_q, mu_q, interp(t_quad, t, batch["u"]).expand(S, -1, -1, -1))
        residual = 0.5 * whiten(G, f_q - f).pow(2).sum(-1).mean((0, 2)) * (t1 - t0).squeeze(1)

        # KL(N(m0, S0) || N(z0_mean, diag exp(z0_logvar))): with a full S0 = L0 L0^T, the diagonal of S0 and
        # log det S0 = sum of 2 log diag L0 take the place of v0 and the sum of log v0
        m0 = m[:, -1]
        if full:
            L0 = scale[:, -1]
            logv0, v0 = 2 * L0.diagonal(dim1=-2, dim2=-1).log(), L0.pow(2).sum(-1)
        else:
            logv0 = scale[:, -1]
            v0 = logv0.exp()
        kl0 = 0.5 * (self.z0_logvar - logv0 - 1 + (v0 + (m0 - self.z0_mean).pow(2)) / self.z0_logvar.exp()).sum(-1)

        if per_trajectory:  # (B,) negative ELBO of each trajectory, not divided by the number of values (src/latent/switching.py)
            return -(recon - beta * (residual + kl0)), None
        n = weight.sum()
        terms = {"recon": recon.sum() / n, "residual": residual.sum() / n, "kl0": kl0.sum() / n}
        loss = -(terms["recon"] - beta * (terms["residual"] + terms["kl0"]))
        if diffusion_penalty:
            G2 = G.pow(2).sum((-1, -2) if G.dim() > z_q.dim() else -1)
            terms["diffusion_penalty"] = diffusion_penalty * G2.mean()
            loss = loss + terms["diffusion_penalty"]
        return loss, terms

    def _static_loss(self, batch: dict, x: Tensor, weight: Tensor, feat: Tensor, covered: Tensor, beta: float,
                     diffusion_penalty: float):
        """
        loss with a static latent w (self.static): the Monte Carlo ELBO of loss(), with one w sample per z sample
        appended to mu, and KL(q(w) || N(0, I)) as a fourth term (weight beta). Gaussian posterior only.
        """
        t = batch["t"]
        (B, N), S, Q = t.shape, self.n_samples, self.n_quad
        m_w, logv_w = self.static(feat, covered, batch["mu"])
        w = m_w + (0.5 * logv_w).exp() * torch.randn(S, *m_w.shape, dtype=m_w.dtype, device=m_w.device)
        if self.static.n_components > 1:  # mixture prior: Monte Carlo KL, log q(w) - log p(w), at the samples
            log_q = (-0.5 * ((w - m_w).pow(2) / logv_w.exp() + logv_w + math.log(2 * math.pi))).sum(-1)
            kl_w = (log_q - self.static.log_prior(w)).mean(0)
        else:
            kl_w = 0.5 * ((m_w - self.static.prior_mean(batch["mu"])).pow(2) + logv_w.exp() - 1 - logv_w).sum(-1)
        t0 = torch.where(covered, t, torch.inf).amin(1, keepdim=True)
        t1 = torch.where(covered, t, -torch.inf).amax(1, keepdim=True)
        strata = torch.arange(Q, dtype=t.dtype, device=t.device) + torch.rand(B, Q, dtype=t.dtype, device=t.device)
        t_quad = t0 + (t1 - t0) * strata / Q
        m, scale, dm, dscale = self.posterior(torch.cat([t, t_quad, t0], 1))
        if scale.dim() > m.dim():
            raise ValueError("a static latent needs a diagonal-covariance posterior")
        eps = torch.randn(S, *m.shape, dtype=m.dtype, device=m.device)
        z = m + (0.5 * scale).exp() * eps
        mu = torch.cat([batch["mu"].expand(S, -1, -1), w], -1)[:, :, None].expand(S, B, N + Q + 1, -1)
        recon = (self.decoder(z[:, :, :N], mu[:, :, :N]).log_prob(x) * weight).sum((-1, -2)).mean(0)
        q = slice(N, N + Q)
        z_q, mu_q = z[:, :, q], mu[:, :, q]
        G, div = self.diffusion.with_divergence(z_q, mu_q, t_quad)
        f_q = posterior_drift(z_q, m[:, q], dm[:, q], scale[:, q].exp(), dscale[:, q], G, div)
        f = self.drift(z_q, mu_q, interp(t_quad, t, batch["u"]).expand(S, -1, -1, -1))
        residual = 0.5 * whiten(G, f_q - f).pow(2).sum(-1).mean((0, 2)) * (t1 - t0).squeeze(1)
        m0, logv0 = m[:, -1], scale[:, -1]
        kl0 = 0.5 * (self.z0_logvar - logv0 - 1 + (logv0.exp() + (m0 - self.z0_mean).pow(2))
                     / self.z0_logvar.exp()).sum(-1)
        n = weight.sum()
        terms = {"recon": recon.sum() / n, "residual": residual.sum() / n, "kl0": kl0.sum() / n,
                 "kl_w": kl_w.sum() / n}
        loss = -(terms["recon"] - beta * (terms["residual"] + terms["kl0"] + terms["kl_w"]))
        if diffusion_penalty:
            G2 = G.pow(2).sum((-1, -2) if G.dim() > z_q.dim() else -1)
            terms["diffusion_penalty"] = diffusion_penalty * G2.mean()
            loss = loss + terms["diffusion_penalty"]
        return loss, terms

    def _mixture_loss(self, batch: dict, x: Tensor, weight: Tensor, covered: Tensor, beta: float,
                      diffusion_penalty: float):
        """
        loss for a mixture posterior (MixtureKalmanPosterior, constant weights pi_k): every expectation is
        sum_k pi_k E_{N_k}[.], from n_samples reparametrized samples of each component (no discrete sampling). The
        residual uses mixture_drift, the drift that keeps the mixture marginals with the prior's G(z); KL(q(z(t0)) ||
        p(z0)) is a Monte Carlo estimate (a mixture has no closed form).
        """
        t, post = batch["t"], self.posterior
        (B, N), S, Q = t.shape, self.n_samples, self.n_quad
        t0 = torch.where(covered, t, torch.inf).amin(1, keepdim=True)
        t1 = torch.where(covered, t, -torch.inf).amax(1, keepdim=True)
        strata = torch.arange(Q, dtype=t.dtype, device=t.device) + torch.rand(B, Q, dtype=t.dtype, device=t.device)
        t_quad = t0 + (t1 - t0) * strata / Q
        m, L = post.components(torch.cat([t, t_quad, t0], 1))  # (K, B, N + Q + 1, d), (..., d, d)
        K = m.shape[0]
        pi = post.log_weights(m).exp().view(K, 1)
        eps = torch.randn(S, *m.shape, dtype=m.dtype, device=m.device)
        z = m + (L @ eps.unsqueeze(-1)).squeeze(-1)  # (S, K, B, N + Q + 1, d)
        mu = batch["mu"][None, None, :, None].expand(S, K, B, N + Q + 1, -1)

        log_p = (self.decoder(z[..., :N, :], mu[..., :N, :]).log_prob(x) * weight).sum((-1, -2)).mean(0)  # (K, B)
        recon = (pi * log_p).sum(0)

        q = slice(N, N + Q)
        z_q, mu_q = z[..., q, :], mu[..., q, :]
        G, div = self.diffusion.with_divergence(z_q, mu_q, t_quad)
        f_q = post.mixture_drift(z_q, q, G, div)
        f = self.drift(z_q, mu_q, interp(t_quad, t, batch["u"]).expand(S, K, -1, -1, -1))
        residual = (pi * 0.5 * whiten(G, f_q - f).pow(2).sum(-1).mean((0, 3))).sum(0) * (t1 - t0).squeeze(1)

        z0 = z[..., -1, :]  # (S, K, B, d) at t0, the last query column
        log_p0 = Normal(self.z0_mean, (0.5 * self.z0_logvar).exp()).log_prob(z0).sum(-1)
        kl0 = (pi * (post.mixture_log_density(z0, -1) - log_p0).mean(0)).sum(0)

        n = weight.sum()
        terms = {"recon": recon.sum() / n, "residual": residual.sum() / n, "kl0": kl0.sum() / n}
        loss = -(terms["recon"] - beta * (terms["residual"] + terms["kl0"]))
        if diffusion_penalty:
            G2 = G.pow(2).sum((-1, -2) if G.dim() > z_q.dim() else -1)
            terms["diffusion_penalty"] = diffusion_penalty * G2.mean()
            loss = loss + terms["diffusion_penalty"]
        return loss, terms

    def _closed_form_loss(self, batch: dict, x: Tensor, weight: Tensor, m: Tensor, S: Tensor, kl0: Tensor,
                          residual: Tensor, beta: float, diffusion_penalty: float):
        """
        loss for a posterior that gives the KL terms in closed form (closed_form_terms: KalmanPosterior with a linear
        drift): the smoothed marginals N(m, S) at the steps (B, N, ...) and kl0, residual (B,). No quadrature times.
        The reconstruction is in closed form for a decoder with expected_log_prob (a linear GaussianDecoder), else
        from n_samples samples of the marginals at the steps.
        """
        mu = batch["mu"][:, None].expand(-1, m.shape[1], -1)
        expected = getattr(self.decoder, "expected_log_prob", None)
        log_p = expected(x, m.to(x.dtype), S.to(x.dtype), mu) if expected is not None else None
        if log_p is None:
            I = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
            L = torch.linalg.cholesky(S + getattr(self.posterior, "jitter", 1e-10) * I).to(x.dtype)
            eps = torch.randn(self.n_samples, *m.shape, dtype=x.dtype, device=x.device)
            z = m.to(x.dtype) + (L @ eps.unsqueeze(-1)).squeeze(-1)
            log_p = self.decoder(z, mu.expand(self.n_samples, -1, -1, -1)).log_prob(x).mean(0)
        n = weight.sum()
        terms = {"recon": (log_p * weight).sum() / n, "residual": residual.sum().to(x.dtype) / n,
                 "kl0": kl0.sum().to(x.dtype) / n}
        loss = -(terms["recon"] - beta * (terms["residual"] + terms["kl0"]))
        if diffusion_penalty:  # the posterior needs a constant diagonal diffusion: G is the same at every state
            G = self.diffusion(m.to(x.dtype), mu, batch["t"])
            terms["diffusion_penalty"] = diffusion_penalty * G.pow(2).sum(-1).mean()
            loss = loss + terms["diffusion_penalty"]
        return loss, terms
