"""
Latent SDE with an encoder and a decoder, trained without SDE simulation.

The latent state follows dz = f(z, mu, u(t)) dt + g dW, with parameters mu and a forcing u(t) for each trajectory
(ROM-VISDE). An observation x_k at time t_k has the decoder density p(x_k | z(t_k), mu). The posterior
q(z(t)) = N(m(t), diag v(t)) comes from an encoder of the observations. The loss is the negative ELBO
(Course & Nair 2023, Bartosh et al. 2025), divided by the number of observed values:

    - sum_k E_q log p(x_k | z(t_k), mu)        reconstruction
    + int E_q |f_q - f|^2 / (2 g^2) dt          drift residual; f_q is the posterior drift of the sym gauge
    + KL(q(z(t_0)) || p(z_0))                  initial state

Stratified uniform times estimate the integral; reparametrized samples estimate the expectations.
The drift has no time input: put t into a column of u for an explicit time dependence.

Posteriors (condition(batch) once, then m, log v, dm/dt, dlog v/dt at any times):
    KernelPosterior     ROM-VISDE (visde/sdevar.py, AmortizedLatentVarGP; GPL-3.0): an encoder maps each window
                        of num_window observations to N(m_k, v_k) at its first time; a kernel interpolates m and
                        log v in time. Needs evenly spaced, fully observed trajectories.
    SequencePosterior   SDE matching (Bartosh et al. 2025): a GRU reads observations at any times, with a mask;
                        an MLP of the attention context at t gives m(t) and log v(t). Written from the paper,
                        because the sde_matching repository has no license.

A batch is a dict (make_batch): t (B, N) sorted times, x (B, N, D), mask (B, N, D) with 1 for observed values,
mu (B, P), u (B, N, F) forcing at t. Pad short trajectories with mask 0 and the last time repeated.
"""
import itertools
import math
from typing import Iterable, Sequence, Union

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn.functional import softplus

from .posterior.gauges import sym_drift_diagonal


def _raw(value: float) -> float:
    """Inverse of softplus."""
    return math.log(math.expm1(value))


def mlp(n_in: int, n_out: int, hidden: Sequence[int] = (64, 64), activation=nn.SiLU) -> nn.Sequential:
    sizes = (n_in, *hidden, n_out)
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), activation()]
    return nn.Sequential(*layers[:-1])


def make_batch(t: Tensor, x: Tensor, mask: Tensor = None, mu: Tensor = None, u: Tensor = None) -> dict:
    """Batch with defaults: all values observed, no parameters (P = 0), no forcing (F = 0)."""
    B, N, _ = x.shape
    return {"t": t, "x": x, "mask": torch.ones_like(x) if mask is None else mask,
            "mu": x.new_zeros(B, 0) if mu is None else mu, "u": x.new_zeros(B, N, 0) if u is None else u}


def interp(t_query: Tensor, t: Tensor, y: Tensor) -> Tensor:
    """
    Linear interpolation of y (B, N, F) at sorted times t (B, N), for t_query (B, Q); holds the end values.
    In a padded row, the first copy of the last time is the end.
    """
    t = t.contiguous()
    i = torch.minimum(torch.searchsorted(t, t_query.contiguous()), torch.searchsorted(t, t[:, -1:])).clamp(min=1)
    t0, t1 = t.gather(1, i - 1), t.gather(1, i)
    w = ((t_query - t0) / (t1 - t0).clamp_min(1e-12)).clamp(0, 1).unsqueeze(-1)
    i = i.unsqueeze(-1).expand(-1, -1, y.shape[-1])
    y0, y1 = y.gather(1, i - 1), y.gather(1, i)
    return y0 + w * (y1 - y0)


class DriftNet(nn.Module):
    """f(z, mu, u) = MLP([z, mu, u])."""

    def __init__(self, dim_z: int, dim_mu: int = 0, dim_u: int = 0, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        self.net = mlp(dim_z + dim_mu + dim_u, dim_z, hidden)

    def forward(self, z: Tensor, mu: Tensor, u: Tensor) -> Tensor:
        return self.net(torch.cat([z, mu, u], -1))


class GaussianDecoder(nn.Module):
    """p(x | z, mu) = N(MLP([z, mu]), diag s^2), one learned s per output; hidden=() gives a linear mean."""

    def __init__(self, dim_z: int, dim_x: int, dim_mu: int = 0, hidden: Sequence[int] = (), std_init: float = 0.1):
        super().__init__()
        self.mean = mlp(dim_z + dim_mu, dim_x, hidden)
        self.raw_std = nn.Parameter(torch.full((dim_x,), _raw(std_init)))

    def forward(self, z: Tensor, mu: Tensor) -> Normal:
        return Normal(self.mean(torch.cat([z, mu], -1)), softplus(self.raw_std))


class KernelPosterior(nn.Module):
    """
    ROM-VISDE's amortized posterior. kernel(t1 (..., n, 1), t2 (..., m, 1), return_grad) has an interpolation
    noise kernel.var, e.g. DeepGaussianKernel(batch_size=N - num_window + 1, dt=dt) for N observations.
    """

    def __init__(self, dim_x: int, dim_z: int, num_window: int, kernel: nn.Module, dim_mu: int = 0,
                 hidden: Sequence[int] = (128, 128)):
        super().__init__()
        self.num_window = num_window
        self.kernel = kernel
        self.encoder = mlp(num_window * dim_x + dim_mu, 2 * dim_z, hidden)

    def condition(self, batch: dict) -> Tensor:
        """Encodes the windows; returns (B, N), True at the window starts (the times the posterior covers)."""
        t, mu = batch["t"], batch["mu"]
        windows = batch["x"].unfold(1, self.num_window, 1).flatten(-2)  # (B, K, D * num_window)
        K = windows.shape[1]
        nodes = self.encoder(torch.cat([windows, mu.unsqueeze(1).expand(-1, K, -1)], -1))  # [m_k, log v_k]
        self.t0 = t[:, :1]
        self.t_nodes = (t[:, :K] - self.t0).unsqueeze(-1)
        gram = self.kernel(self.t_nodes, self.t_nodes)
        gram = gram + self.kernel.var * torch.eye(K, dtype=t.dtype, device=t.device)
        self.weights = torch.cholesky_solve(nodes, torch.linalg.cholesky(gram))
        return (torch.arange(t.shape[1], device=t.device) < K).expand_as(t)

    def forward(self, t: Tensor):
        """m, log v, dm/dt, dlog v/dt at t (B, Q), each (B, Q, dim_z)."""
        k, dk = self.kernel((t - self.t0).unsqueeze(-1), self.t_nodes, return_grad=True)
        m, logv = (k @ self.weights).chunk(2, -1)
        dm, dlogv = (dk @ self.weights).chunk(2, -1)
        return m, logv, dm, dlogv


class SequencePosterior(nn.Module):
    """
    A bidirectional GRU reads [x * mask, mask, t - t_0] at the observation times. At a time t, the context is the
    mean of the GRU outputs with weights softmax(-((t - t_k) / width)^2) over the observed steps; an MLP of
    [context, mu, t - t_0] gives m(t) and log v(t), and forward-mode AD their time derivatives.
    """

    def __init__(self, dim_x: int, dim_z: int, dim_mu: int = 0, hidden: int = 64, width: float = 1.0):
        super().__init__()
        self.gru = nn.GRU(2 * dim_x + 1, hidden, batch_first=True, bidirectional=True)
        self.head = mlp(2 * hidden + dim_mu + 1, 2 * dim_z, (hidden, hidden))
        self.raw_width = nn.Parameter(torch.tensor(_raw(width)))

    def condition(self, batch: dict) -> Tensor:
        """Runs the GRU; returns (B, N), True at the steps with an observed value."""
        t, mask = batch["t"], batch["mask"]
        self.t_obs, self.mu, self.observed = t, batch["mu"], mask.any(-1)
        self.h, _ = self.gru(torch.cat([batch["x"] * mask, mask, (t - t[:, :1]).unsqueeze(-1)], -1))
        return self.observed

    def _mean_logvar(self, t: Tensor):
        logits = -((t.unsqueeze(-1) - self.t_obs.unsqueeze(1)) / softplus(self.raw_width)).pow(2)  # (B, Q, N)
        context = logits.masked_fill(~self.observed.unsqueeze(1), -torch.inf).softmax(-1) @ self.h
        mu = self.mu.unsqueeze(1).expand(-1, t.shape[1], -1)
        return self.head(torch.cat([context, mu, (t - self.t_obs[:, :1]).unsqueeze(-1)], -1)).chunk(2, -1)

    def forward(self, t: Tensor):
        """m, log v, dm/dt, dlog v/dt at t (B, Q), each (B, Q, dim_z)."""
        (m, logv), (dm, dlogv) = torch.func.jvp(self._mean_logvar, (t,), (torch.ones_like(t),))
        return m, logv, dm, dlogv


class LatentSDE(nn.Module):
    """
    Posterior, drift f(z, mu, u), decoder p(x | z, mu), a constant diffusion g = softplus(raw_g) and a prior
    N(z0_mean, exp(z0_logvar)) on the initial state. Drift and decoder parameters are point estimates.
    """

    def __init__(self, posterior: nn.Module, drift: nn.Module, decoder: nn.Module, dim_z: int,
                 g_init: float = 0.1, n_samples: int = 8, n_quad: int = 32):
        super().__init__()
        self.posterior, self.drift, self.decoder = posterior, drift, decoder
        self.n_samples, self.n_quad = n_samples, n_quad
        self.raw_g = nn.Parameter(torch.full((dim_z,), _raw(g_init)))
        self.z0_mean = nn.Parameter(torch.zeros(dim_z))
        self.z0_logvar = nn.Parameter(torch.zeros(dim_z))

    @property
    def g(self) -> Tensor:
        return softplus(self.raw_g)

    def loss(self, batch: dict, beta: float = 1.0):
        """Negative ELBO per observed value, with beta on the residual and the KL term; and the three terms."""
        t, mu, mask = batch["t"], batch["mu"], batch["mask"]
        x = torch.where(mask > 0, batch["x"], torch.zeros_like(batch["x"]))
        covered = self.posterior.condition({**batch, "x": x})
        (B, N), S, Q = t.shape, self.n_samples, self.n_quad

        # one posterior call: observation times, stratified times on the covered span, start of the span
        t0 = torch.where(covered, t, torch.inf).amin(1, keepdim=True)
        t1 = torch.where(covered, t, -torch.inf).amax(1, keepdim=True)
        strata = torch.arange(Q, dtype=t.dtype, device=t.device) + torch.rand(B, Q, dtype=t.dtype, device=t.device)
        t_quad = t0 + (t1 - t0) * strata / Q
        m, logv, dm, dlogv = self.posterior(torch.cat([t, t_quad, t0], 1))
        z = m + (0.5 * logv).exp() * torch.randn(S, *m.shape, dtype=m.dtype, device=m.device)
        mu = mu[None, :, None].expand(S, B, N + Q + 1, -1)

        weight = mask * covered.unsqueeze(-1)
        recon = (self.decoder(z[:, :, :N], mu[:, :, :N]).log_prob(x) * weight).sum((-1, -2)).mean(0)

        q = slice(N, N + Q)
        v = logv[:, q].exp()
        f_q = sym_drift_diagonal(z[:, :, q], m[:, q], dm[:, q], v, v * dlogv[:, q], self.g.pow(2))
        f = self.drift(z[:, :, q], mu[:, :, q], interp(t_quad, t, batch["u"]).expand(S, -1, -1, -1))
        residual = 0.5 * ((f_q - f) / self.g).pow(2).sum(-1).mean((0, 2)) * (t1 - t0).squeeze(1)

        m0, logv0 = m[:, -1], logv[:, -1]
        kl0 = 0.5 * (self.z0_logvar - logv0 - 1
                     + (logv0.exp() + (m0 - self.z0_mean).pow(2)) / self.z0_logvar.exp()).sum(-1)

        n = weight.sum()
        terms = {"recon": recon.sum() / n, "residual": residual.sum() / n, "kl0": kl0.sum() / n}
        return -(terms["recon"] - beta * (terms["residual"] + terms["kl0"])), terms


def fit(model: LatentSDE, data: Union[dict, Iterable[dict]], n_iter: int = 5000, lr: float = 1e-3,
        n_warmup: int = 1000, lr_decay_every: int = 1000, log_every: int = 500) -> list:
    """
    Adam with the learning rate lr * 0.9^(i / lr_decay_every), and beta from 0 to 1 over the first n_warmup
    iterations (ROM-VISDE). data is one batch, or an iterable of batches. Returns the loss of each iteration.
    """
    batches = itertools.repeat(data) if isinstance(data, dict) else itertools.cycle(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9 ** (1 / lr_decay_every))
    history = []
    for i, batch in zip(range(n_iter), batches):
        loss, terms = model.loss(batch, beta=min(1.0, i / n_warmup) if n_warmup else 1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        history.append(loss.item())
        if log_every and i % log_every == 0:
            print(i, f"loss {loss.item():.4g}", *(f"{k} {v.item():.4g}" for k, v in terms.items()))
    return history
