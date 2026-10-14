import math

import torch
import torch.nn as nn

from ..core.interpolation import linterp
from ..kernels import DeepGaussianKernel
from ..latent import (DriftNet, GaussianDecoder, KernelPosterior, LatentSDE, SequencePosterior, fit, interp,
                      make_batch)
from ._reference import run_tests, use_float64, visde_module

use_float64()


def test_interp_matches_linterp_and_holds_padding():
    torch.manual_seed(0)
    t = torch.rand(3, 9).sort(-1).values
    t[2, 6:] = t[2, 5]  # padded row
    y = torch.randn(3, 9, 2)
    t_query = torch.rand(3, 20) * 1.2 - 0.1
    out = interp(t_query, t, y)
    for b, n in ((0, 9), (1, 9), (2, 6)):
        torch.testing.assert_close(out[b], linterp(y[b, :n], t[b, :n], t_query[b]))
    assert interp(t_query, t, y[..., :0]).shape == (3, 20, 0)


class ExactOU(nn.Module):
    """Marginals of dz = -a z dt + g dW, z(0) ~ N(m0, v0): the residual of the true drift is zero."""

    def __init__(self, a, g, m0, v0):
        super().__init__()
        self.a, self.g, self.m0, self.v0 = a, g, m0, v0

    def condition(self, batch):
        return torch.ones_like(batch["t"], dtype=torch.bool)

    def _mean_logvar(self, t):
        e = torch.exp(-self.a * t).unsqueeze(-1)
        v = self.v0 * e.pow(2) + self.g ** 2 / (2 * self.a) * (1 - e.pow(2))
        return self.m0 * e, v.log()

    def forward(self, t):
        (m, logv), (dm, dlogv) = torch.func.jvp(self._mean_logvar, (t,), (torch.ones_like(t),))
        return m, logv, dm, dlogv


def test_residual_and_kl_vanish_for_exact_ou_posterior():
    a, g, m0, v0 = 0.7, 0.3, torch.tensor([1.0, -2.0]), torch.tensor([0.2, 0.5])
    model = LatentSDE(ExactOU(a, g, m0, v0), lambda z, mu, u: -a * z, GaussianDecoder(2, 3), dim_z=2, g_init=g)
    with torch.no_grad():
        model.z0_mean.copy_(m0)
        model.z0_logvar.copy_(v0.log())
    t = torch.linspace(0, 2, 10).repeat(4, 1)
    _, terms = model.loss(make_batch(t, torch.randn(4, 10, 3)))
    assert abs(terms["residual"].item()) < 1e-12 and abs(terms["kl0"].item()) < 1e-12


def test_kernel_posterior_matches_rom_visde():
    torch.manual_seed(0)
    N, M, D, dz, dt = 12, 3, 4, 2, 0.1
    K = N - M + 1
    sdevar, kernel = visde_module("sdevar"), visde_module("kernel")
    ref_kernel = kernel.DeepGaussianKernel(
        nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)), K, dt)
    ours = KernelPosterior(D, dz, M, DeepGaussianKernel(batch_size=K, dt=dt), dim_mu=1, hidden=(16,))
    ours.kernel.warping.net.load_state_dict(ref_kernel.net.state_dict())
    ours.kernel.load_state_dict({**ours.kernel.state_dict(), "rawlens": ref_kernel.rawlens.detach(),
                                 "rawsigma": ref_kernel.rawsigma.detach(), "rawvar": ref_kernel.rawvar.detach()})

    class Encoder(nn.Module):
        def forward(self, mu, x_win):  # (K, P), (K, M, D) -> mean, var
            mean, logvar = ours.encoder(torch.cat([x_win.transpose(-2, -1).flatten(-2), mu], -1)).chunk(2, -1)
            return mean, logvar.exp()

    reference = sdevar.AmortizedLatentVarGP(sdevar.LatentVarConfig(1, dz), ref_kernel, Encoder())
    t, x, mu = 0.3 + dt * torch.arange(N), torch.randn(N, D), torch.randn(1)
    ours.condition(make_batch(t[None], x[None], mu=mu[None]))
    reference.form_window(mu.expand(K, 1), t[:K, None], x.unfold(0, M, 1).transpose(-2, -1))
    t_query = 0.3 + torch.rand(7) * (K - 1) * dt
    for a, b in zip(ours(t_query[None]), reference(mu.expand(7, 1), t_query[:, None])):
        torch.testing.assert_close(a[0], b)


def _oscillators(B=6, N=30, D=5, seed=0):
    """z'' = -z - mu z' + u(t) with u = 0.5 sin(2t), seen through a random linear map with noise 0.05."""
    torch.manual_seed(seed)
    mu = 0.1 + 0.3 * torch.rand(B, 1)
    z = torch.randn(B, 2)
    t_fine = torch.linspace(0, 6, 60 * (N - 1) + 1)
    path = [z]
    for h, s in zip(t_fine.diff(), t_fine[:-1]):
        z = z + h * torch.stack([z[:, 1], -z[:, 0] - mu[:, 0] * z[:, 1] + 0.5 * torch.sin(2 * s)], -1)
        path.append(z)
    z = torch.stack(path, 1)[:, ::60]
    t = t_fine[::60].repeat(B, 1)
    x = z @ torch.randn(2, D) + 0.05 * torch.randn(B, N, D)
    return make_batch(t, x, mu=mu, u=0.5 * torch.sin(2 * t).unsqueeze(-1))


def _model(posterior):
    return LatentSDE(posterior, DriftNet(2, 1, 1, (32, 32)), GaussianDecoder(2, 5, 1), dim_z=2, n_samples=4)


def test_masked_values_do_not_change_loss():
    batch = _oscillators()
    batch["mask"] = (torch.rand_like(batch["x"]) > 0.3).double()
    model = _model(SequencePosterior(5, 2, 1, hidden=16))
    garbage = torch.where(batch["mask"] > 0, batch["x"], torch.nan)
    torch.manual_seed(1)
    loss, _ = model.loss(batch)
    torch.manual_seed(1)
    loss_garbage, _ = model.loss({**batch, "x": garbage})
    torch.testing.assert_close(loss, loss_garbage)


def test_training_lowers_the_loss():
    batch = _oscillators()
    K = 30 - 3 + 1
    for posterior in (SequencePosterior(5, 2, 1, hidden=32),
                      KernelPosterior(5, 2, 3, DeepGaussianKernel(batch_size=K, dt=6 / 29), 1, hidden=(32,))):
        torch.manual_seed(0)
        history = fit(_model(posterior), batch, n_iter=300, lr=3e-3, n_warmup=100, log_every=0)
        assert all(math.isfinite(h) for h in history)
        assert sum(history[-20:]) / 20 < history[0] - 1.0, (history[0], history[-1])


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
