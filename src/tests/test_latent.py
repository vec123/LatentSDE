import math

import torch
import torch.nn as nn

from ..core.interpolation import linterp
from ..core.transforms import inverse_softplus_scalar
from ..kernels import DeepGaussianKernel
from ..latent import (CollapseGuard, ConstantDiagonalDiffusion, DriftNet, GaussianDecoder, KernelPosterior, LatentSDE,
                      NetCholeskyDiffusion, NetDiagonalDiffusion, ReleaseDiffusion, SequencePosterior, fit, interp,
                      make_batch,
                      posterior_drift, posterior_drift_full)
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


def test_net_diagonal_diffusion_reduces_to_the_constant_case():
    """A NetDiagonalDiffusion forced to output a constant g matches the exact-OU closed form, same as g_init."""
    a, g, m0, v0 = 0.7, 0.3, torch.tensor([1.0, -2.0]), torch.tensor([0.2, 0.5])
    diffusion = NetDiagonalDiffusion(2, 0, hidden=())
    with torch.no_grad():
        diffusion.net[0].weight.zero_()
        diffusion.net[0].bias.fill_(inverse_softplus_scalar(g))
    model = LatentSDE(ExactOU(a, g, m0, v0), lambda z, mu, u: -a * z, GaussianDecoder(2, 3), dim_z=2,
                      diffusion=diffusion)
    with torch.no_grad():
        model.z0_mean.copy_(m0)
        model.z0_logvar.copy_(v0.log())
    t = torch.linspace(0, 2, 10).repeat(4, 1)
    _, terms = model.loss(make_batch(t, torch.randn(4, 10, 3)))
    assert abs(terms["residual"].item()) < 1e-10 and abs(terms["kl0"].item()) < 1e-10


class StandardNormal(nn.Module):
    """Stationary posterior N(0, 1) in each dimension."""

    def condition(self, batch):
        return torch.ones_like(batch["t"], dtype=torch.bool)

    def forward(self, t):
        zero = torch.zeros(*t.shape, 2, dtype=t.dtype)
        return zero, zero, zero, zero


class SqrtOnePlusSquare(nn.Module):
    """g_k = sqrt(1 + z_k^2), so d g_k^2 / d z_k = 2 z_k."""

    def forward(self, z, mu, t):
        return (1 + z.pow(2)).sqrt()

    def with_divergence(self, z, mu, t):
        return self(z, mu, t), 2 * z


def test_residual_vanishes_for_state_dependent_diffusion():
    """
    dz = (z/2 - z^3/2) dt + sqrt(1 + z^2) dW has the stationary density (1/g^2) exp(int 2f/g^2) = N(0, 1), so its
    residual against the posterior N(0, 1) is zero. Without the divergence term it is not.
    """
    model = LatentSDE(StandardNormal(), lambda z, mu, u: 0.5 * z - 0.5 * z.pow(3), GaussianDecoder(2, 3),
                      dim_z=2, diffusion=SqrtOnePlusSquare())
    t = torch.linspace(0, 2, 10).repeat(4, 1)
    _, terms = model.loss(make_batch(t, torch.randn(4, 10, 3)))
    assert abs(terms["residual"].item()) < 1e-12 and abs(terms["kl0"].item()) < 1e-12


def _sigma(diffusion):
    """z, mu, t -> Sigma = G G^T as a full matrix."""
    def sigma(z, mu, t):
        G = diffusion(z, mu, t)
        return G @ G.transpose(-1, -2) if G.dim() > z.dim() else torch.diag_embed(G.pow(2))
    return sigma


def test_net_diffusion_divergence_matches_the_jacobian():
    """with_divergence gives div(Sigma)_i = sum_j d Sigma_ij / d z_j, from the full Jacobian of Sigma."""
    torch.manual_seed(0)
    z, mu, t = torch.randn(4, 5, 3), torch.randn(4, 5, 2), torch.rand(4, 5)
    for diffusion in (NetDiagonalDiffusion(3, 2, hidden=(16, 16)), NetCholeskyDiffusion(3, 2, hidden=(16, 16))):
        _, div = diffusion.with_divergence(z, mu, t)
        jac = torch.func.vmap(torch.func.jacrev(_sigma(diffusion)))(z.flatten(0, 1), mu.flatten(0, 1), t.flatten())
        torch.testing.assert_close(div.flatten(0, 1), torch.einsum("nijj->ni", jac))


def test_net_cholesky_diffusion_is_a_cholesky_factor():
    torch.manual_seed(0)
    G = NetCholeskyDiffusion(4, 1, hidden=(16,))(torch.randn(6, 4), torch.randn(6, 1), torch.rand(6))
    assert (G.triu(1) == 0).all() and (G.diagonal(dim1=-2, dim2=-1) > 0).all()
    assert (G[:, torch.ones(4, 4, dtype=torch.bool).tril(-1)] != 0).all()


def test_posterior_drift_solves_fokker_planck():
    """
    q = N(m(t), diag v(t)) solves dq/dt = -div(f q) + 1/2 sum_ij d_i d_j (Sigma_ij q) for f = posterior_drift and
    each diffusion, also the full Sigma(z) of NetCholeskyDiffusion, with a zero and with a nonzero gauge omega(t).
    """
    torch.manual_seed(0)
    mu = torch.zeros(0)
    J = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

    def m_logv(t):
        return torch.stack([torch.sin(t), torch.cos(2 * t)]), torch.stack([0.5 * torch.sin(t), -t])

    def q(z, t):
        m, logv = m_logv(t)
        return torch.exp(-0.5 * ((z - m).pow(2) / logv.exp() + logv + math.log(2 * math.pi)).sum())

    for diffusion in (ConstantDiagonalDiffusion(2, 0.3), NetDiagonalDiffusion(2, 0, hidden=(16,)),
                      NetCholeskyDiffusion(2, 0, hidden=(16,))):
        sigma = _sigma(diffusion)
        for omega in (lambda t: None, lambda t: (1 + torch.sin(3 * t)) * J):

            def drift(z, t):
                (m, logv), (dm, dlogv) = torch.func.jvp(m_logv, (t,), (torch.ones_like(t),))
                G, div = diffusion.with_divergence(z, mu, t)
                return posterior_drift(z, m, dm, logv.exp(), dlogv, G, div, omega(t))

            def fokker_planck(z, t):
                dq_dt = torch.func.jacfwd(q, 1)(z, t)
                div_fq = torch.func.jacfwd(lambda z: drift(z, t) * q(z, t))(z).trace()
                hessian = torch.func.jacfwd(torch.func.jacfwd(lambda z: sigma(z, mu, t) * q(z, t)))(z)
                return dq_dt + div_fq - 0.5 * torch.einsum("ijij->", hessian)

            residual = torch.func.vmap(fokker_planck)(torch.randn(8, 2), torch.rand(8))
            assert residual.abs().max() < 1e-10, (type(diffusion).__name__, omega(torch.zeros(())), residual)


def _m_chol(t):
    """A 2-D mean and a full, lower-triangular Cholesky factor, both moving in t."""
    m = torch.stack([torch.sin(t), torch.cos(2 * t)])
    L = torch.stack([torch.stack([torch.exp(0.25 * torch.sin(t)), torch.zeros_like(t)]),
                     torch.stack([0.5 * torch.cos(t), torch.exp(-0.5 * t)])])
    return m, L


def test_full_posterior_drift_solves_fokker_planck():
    """
    q = N(m(t), L(t) L(t)^T), with a full covariance, solves the Fokker-Planck equation for f = posterior_drift_full
    and each diffusion, with a zero and with a nonzero gauge omega(t).
    """
    torch.manual_seed(0)
    mu = torch.zeros(0)
    J = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

    def q(z, t):
        m, L = _m_chol(t)
        w = torch.linalg.solve_triangular(L, (z - m).unsqueeze(-1), upper=False).squeeze(-1)
        return torch.exp(-0.5 * w.pow(2).sum() - L.diagonal().log().sum() - math.log(2 * math.pi))

    for diffusion in (ConstantDiagonalDiffusion(2, 0.3), NetDiagonalDiffusion(2, 0, hidden=(16,)),
                      NetCholeskyDiffusion(2, 0, hidden=(16,))):
        sigma = _sigma(diffusion)
        for omega in (lambda t: None, lambda t: (1 + torch.sin(3 * t)) * J):

            def drift(z, t):
                (m, L), (dm, dL) = torch.func.jvp(_m_chol, (t,), (torch.ones_like(t),))
                G, div = diffusion.with_divergence(z, mu, t)
                return posterior_drift_full(z, m, dm, L, dL, G, div, omega(t))

            def fokker_planck(z, t):
                dq_dt = torch.func.jacfwd(q, 1)(z, t)
                div_fq = torch.func.jacfwd(lambda z: drift(z, t) * q(z, t))(z).trace()
                hessian = torch.func.jacfwd(torch.func.jacfwd(lambda z: sigma(z, mu, t) * q(z, t)))(z)
                return dq_dt + div_fq - 0.5 * torch.einsum("ijij->", hessian)

            residual = torch.func.vmap(fokker_planck)(torch.randn(8, 2), torch.rand(8))
            assert residual.abs().max() < 1e-10, (type(diffusion).__name__, omega(torch.zeros(())), residual)


class AsFull(nn.Module):
    """A diagonal posterior, given as the full form: L = diag(sqrt v), dL = diag(sqrt v dlogv / 2)."""

    def __init__(self, posterior):
        super().__init__()
        self.inner = posterior

    def condition(self, batch):
        return self.inner.condition(batch)

    def forward(self, t):
        m, logv, dm, dlogv = self.inner(t)
        s = (0.5 * logv).exp()
        return m, torch.diag_embed(s), dm, torch.diag_embed(0.5 * s * dlogv)

    def gauge(self, t):
        return self.inner.gauge(t)


def test_full_covariance_loss_matches_the_diagonal_loss_for_a_diagonal_l():
    """
    The full-covariance path of LatentSDE.loss (z = m + L eps, posterior_drift_full, the KL with log det) gives the
    loss of the diagonal path when L is diagonal, with a nonzero gauge and a diffusion that depends on the state.
    """
    batch = _oscillators()
    posterior = SequencePosterior(5, 2, 1, hidden=16, gauge=True)
    with torch.no_grad():
        posterior.gauge_head.bias.normal_()
    drift, decoder = DriftNet(2, 1, 1, (32, 32)), GaussianDecoder(2, 5, 1)
    diffusion = NetDiagonalDiffusion(2, 1, hidden=(16,))
    losses = []
    for post in (posterior, AsFull(posterior)):
        torch.manual_seed(1)
        loss, terms = LatentSDE(post, drift, decoder, dim_z=2, n_samples=4, diffusion=diffusion).loss(batch)
        losses.append(torch.stack([loss, *terms.values()]))
    torch.testing.assert_close(*losses)


def test_full_covariance_posterior_gives_a_cholesky_factor_and_trains():
    """SequencePosterior(full_cov=True): L lower triangular with a positive diagonal, dL its time derivative, and a
    few Adam steps with the gauge keep the loss finite."""
    batch = _oscillators()
    posterior = SequencePosterior(5, 3, 1, hidden=16, gauge=True, full_cov=True)
    posterior.condition(batch)
    t = batch["t"][:, :7]
    m, L, dm, dL = posterior(t)
    assert m.shape == (6, 7, 3) and L.shape == (6, 7, 3, 3)
    assert (L.triu(1) == 0).all() and (L.diagonal(dim1=-2, dim2=-1) > 0).all() and (L.tril(-1) != 0).any()
    h = 1e-5
    torch.testing.assert_close(dL, (posterior(t + h)[1] - posterior(t - h)[1]) / (2 * h), atol=1e-6, rtol=0)
    model = LatentSDE(posterior, DriftNet(3, 1, 1, (32, 32)), GaussianDecoder(3, 5, 1), dim_z=3, n_samples=4)
    torch.manual_seed(0)
    history = fit(model, batch, n_iter=5, lr=3e-3, n_warmup=0, log_every=0)
    assert all(math.isfinite(h) for h in history)


def test_bridge_times_vanish_at_observations_and_are_continuous():
    """
    Column 0 seen at t = 0 and 2.5, column 1 at 1 and 4, column 2 never; the last step is padding. The span is 4:
    the cap of d- and d+, and span / 2 for column 2. The bridge time is 0 at an observation, continuous across it,
    and its jvp derivative is the finite difference.
    """
    t = torch.tensor([[0.0, 1.0, 2.5, 4.0, 4.0]])
    mask = torch.zeros(1, 5, 3)
    mask[0, [0, 2], 0] = 1
    mask[0, [1, 3], 1] = 1
    posterior = SequencePosterior(3, 2, hidden=8, gap_features=True)
    posterior.condition(make_batch(t, torch.randn(1, 5, 3), mask))
    bridge = lambda dm, dp: dm * dp / (dm + dp)
    query = torch.tensor([[1.75, 2.5, 4.5]])
    expected = torch.tensor([[[bridge(1.75, 0.75), bridge(0.75, 2.25), 2.0],
                              [0.0, bridge(1.5, 1.5), 2.0],
                              [bridge(2.0, 4.0), bridge(0.5, 4.0), 2.0]]])
    torch.testing.assert_close(posterior.bridge_times(query), expected)
    near = posterior.bridge_times(torch.tensor([[2.5 - 1e-7, 2.5 + 1e-7]]))[0, :, 0]
    assert near.abs().max() < 1e-6
    q = torch.tensor([[0.3, 1.75, 3.1, 4.5]])
    _, slope = torch.func.jvp(posterior.bridge_times, (q,), (torch.ones_like(q),))
    h = 1e-6
    torch.testing.assert_close(slope, (posterior.bridge_times(q + h) - posterior.bridge_times(q - h)) / (2 * h))


def test_full_covariance_posterior_with_gap_features_trains():
    """SequencePosterior with full_cov, gauge and gap_features, on columns with their own times: finite losses."""
    batch = _oscillators()
    batch["mask"] = (torch.rand_like(batch["x"]) > 0.5).double()
    posterior = SequencePosterior(5, 3, 1, hidden=16, gauge=True, full_cov=True, gap_features=True)
    model = LatentSDE(posterior, DriftNet(3, 1, 1, (32, 32)), GaussianDecoder(3, 5, 1), dim_z=3, n_samples=4)
    torch.manual_seed(0)
    history = fit(model, batch, n_iter=5, lr=3e-3, n_warmup=0, log_every=0)
    assert all(math.isfinite(h) for h in history)


def test_gauge_posterior_starts_at_the_fixed_gauge_and_learns():
    """
    The gauge head of SequencePosterior starts at zero, so the loss is that of the fixed gauge. A few Adam steps
    make Omega nonzero, and it stays skew-symmetric.
    """
    batch = _oscillators()
    fixed = SequencePosterior(5, 3, 1, hidden=16)
    learned = SequencePosterior(5, 3, 1, hidden=16, gauge=True)
    learned.load_state_dict(fixed.state_dict(), strict=False)
    drift, decoder = DriftNet(3, 1, 1, (32, 32)), GaussianDecoder(3, 5, 1)
    losses = []
    for posterior in (fixed, learned):
        torch.manual_seed(1)
        losses.append(LatentSDE(posterior, drift, decoder, dim_z=3, n_samples=4).loss(batch)[0])
    torch.testing.assert_close(*losses)

    model = LatentSDE(learned, drift, decoder, dim_z=3, n_samples=4,
                      diffusion=NetDiagonalDiffusion(3, 1, hidden=(16,)))
    torch.manual_seed(0)
    history = fit(model, batch, n_iter=5, lr=3e-3, n_warmup=0, log_every=0)
    assert all(math.isfinite(h) for h in history)
    omega = learned.gauge(batch["t"])
    assert omega.shape == (*batch["t"].shape, 3, 3) and omega.abs().max() > 0
    torch.testing.assert_close(omega, -omega.transpose(-1, -2))


def test_explicit_constant_diffusion_matches_the_default():
    """ConstantDiagonalDiffusion(dim_z, g_init), passed explicitly, is the same module the default builds."""
    batch = _oscillators()
    posterior = SequencePosterior(5, 2, 1, hidden=16)
    drift, decoder = DriftNet(2, 1, 1, (32, 32)), GaussianDecoder(2, 5, 1)
    default_model = LatentSDE(posterior, drift, decoder, dim_z=2, n_samples=4)
    explicit_model = LatentSDE(posterior, drift, decoder, dim_z=2, n_samples=4,
                                diffusion=ConstantDiagonalDiffusion(2, 0.1))
    torch.manual_seed(1)
    loss_default, _ = default_model.loss(batch)
    torch.manual_seed(1)
    loss_explicit, _ = explicit_model.loss(batch)
    torch.testing.assert_close(loss_default, loss_explicit)


def test_net_diffusions_train_without_shape_errors():
    """The (samples, batch, time, dim) broadcast of the network diffusions does not corrupt the loss."""
    for make in (NetDiagonalDiffusion, NetCholeskyDiffusion):
        for n_samples, B in ((4, 6), (1, 3), (5, 1)):
            batch = _oscillators(B=B)
            model = LatentSDE(SequencePosterior(5, 2, 1, hidden=16), DriftNet(2, 1, 1, (32, 32)),
                              GaussianDecoder(2, 5, 1), dim_z=2, diffusion=make(2, 1, hidden=(16,)),
                              n_samples=n_samples)
            torch.manual_seed(0)
            history = fit(model, batch, n_iter=20, lr=3e-3, n_warmup=10, log_every=0)
            assert all(math.isfinite(h) for h in history), (make.__name__, n_samples, B, history)


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


def test_collapse_guard_stops_training_when_a_column_is_noise():
    """A decoder row of zeros makes column 2 independent of z (explained as noise): after `patience` checks in a row,
    fit stops early and the guard names the column; the other columns, with weight, do not count."""
    torch.manual_seed(0)
    t = torch.linspace(0, 1, 10).expand(4, -1).contiguous()
    batch = make_batch(t, torch.randn(4, 10, 3))
    model = LatentSDE(SequencePosterior(3, 3, hidden=8), DriftNet(3, hidden=(8,)), GaussianDecoder(3, 3), 3,
                      n_samples=2, n_quad=4)
    with torch.no_grad():
        model.decoder.mean[0].weight.copy_(torch.diag(torch.tensor([10.0, 10.0, 0.0])))
    guard = CollapseGuard(batch, every=1, start=0, ratio=0.1, patience=2)
    losses = fit(model, batch, n_iter=50, lr=0.0, n_warmup=0, log_every=0, callbacks=[guard])
    assert guard.stopped_at == 2 and guard.collapsed == [2] and len(losses) == 2


def test_fixed_attention_width_does_not_change_in_training():
    """learn_width=False: the width is a buffer, not a parameter, so Adam leaves it at its start value."""
    batch = _oscillators()
    model = _model(SequencePosterior(5, 2, 1, hidden=16, width=0.75, learn_width=False))
    assert "posterior.raw_width" not in dict(model.named_parameters())
    before = model.posterior.raw_width.clone()
    fit(model, batch, n_iter=5, lr=1e-2, n_warmup=0, log_every=0)
    torch.testing.assert_close(model.posterior.raw_width, before)
    torch.testing.assert_close(torch.nn.functional.softplus(before), torch.tensor(0.75, dtype=before.dtype))


def test_fixed_decoder_std_does_not_change_in_training():
    """std_fixed: the decoder's std is a buffer at the given values, not a parameter, so Adam leaves it."""
    batch = _oscillators()
    decoder = GaussianDecoder(2, 5, 1, std_fixed=[0.1, 0.05, 0.05, 0.2, 0.3])
    model = LatentSDE(SequencePosterior(5, 2, 1, hidden=16), DriftNet(2, 1, 1, (32, 32)), decoder, dim_z=2,
                      n_samples=4)
    assert "decoder.raw_std" not in dict(model.named_parameters())
    fit(model, batch, n_iter=5, lr=1e-2, n_warmup=0, log_every=0)
    std = decoder(torch.zeros(1, 2), torch.zeros(1, 1)).stddev[0]
    torch.testing.assert_close(std, torch.tensor([0.1, 0.05, 0.05, 0.2, 0.3], dtype=std.dtype))


def test_fixed_diffusion_dimension_does_not_change_in_training():
    """ConstantDiagonalDiffusion(fixed=[0.05, None]): g of dimension 0 stays 0.05, g of dimension 1 learns."""
    batch = _oscillators()
    diffusion = ConstantDiagonalDiffusion(2, 0.1, fixed=[0.05, None])
    model = LatentSDE(SequencePosterior(5, 2, 1, hidden=16), DriftNet(2, 1, 1, (32, 32)), GaussianDecoder(2, 5, 1),
                      dim_z=2, n_samples=4, diffusion=diffusion)
    fit(model, batch, n_iter=5, lr=1e-2, n_warmup=0, log_every=0)
    g = diffusion(torch.zeros(1, 2), torch.zeros(1, 1), torch.zeros(()))[0]
    torch.testing.assert_close(g[0], torch.tensor(0.05, dtype=g.dtype))
    assert abs(g[1].item() - 0.1) > 1e-4


def test_input_scaling_standardizes_what_the_gru_reads():
    """set_input_scaling(shift, scale, 1): the GRU states equal those of the same posterior without scaling, given
    (x - shift) / scale; the masked values stay 0."""
    batch = _oscillators()
    batch["mask"] = (torch.rand_like(batch["x"]) > 0.3).double()
    batch["x"] = batch["x"] * batch["mask"]
    shift, scale = torch.randn(5), 0.5 + torch.rand(5)
    scaled, plain = SequencePosterior(5, 2, 1, hidden=16), SequencePosterior(5, 2, 1, hidden=16)
    plain.load_state_dict(scaled.state_dict())
    scaled.set_input_scaling(shift, scale, 1.0)
    scaled.condition(batch)
    plain.condition({**batch, "x": (batch["x"] - shift) / scale * batch["mask"]})
    torch.testing.assert_close(scaled.h, plain.h)


def test_neighbor_context_is_continuous_and_reads_both_ends():
    """
    Column 0 seen at t = 0 and 2.5, column 1 at 1 and 4, column 2 never (as in the bridge-time test). The neighbor
    context is continuous across an observation of a column, its jvp derivative is the finite difference, it is
    h of the observation (and 0 change) at that observation, and 0 for the column with no observation.
    """
    t = torch.tensor([[0.0, 1.0, 2.5, 4.0, 4.0]])
    mask = torch.zeros(1, 5, 3)
    mask[0, [0, 2], 0] = 1
    mask[0, [1, 3], 1] = 1
    posterior = SequencePosterior(3, 2, hidden=4, neighbor_context=True)
    posterior.condition(make_batch(t, torch.randn(1, 5, 3), mask))
    H = posterior.h.shape[-1]
    ctx = lambda q: posterior.neighbor_context(torch.tensor([q])).view(len(q), 3, 2, H)
    left, at, right = ctx([2.5 - 1e-7, 2.5, 2.5 + 1e-7])
    torch.testing.assert_close(left[0], at[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(right[0], at[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(at[0, 0], posterior.h[0, 2])
    assert at[0, 1].abs().max() < 1e-12 and at[2].abs().max() == 0
    q = torch.tensor([[0.3, 1.75, 3.1, 4.5]])
    _, slope = torch.func.jvp(posterior.neighbor_context, (q,), (torch.ones_like(q),))
    h = 1e-6
    fd = (posterior.neighbor_context(q + h) - posterior.neighbor_context(q - h)) / (2 * h)
    torch.testing.assert_close(slope, fd)


def test_posterior_with_neighbor_context_trains():
    """SequencePosterior with full_cov, gauge, gap_features and neighbor_context: finite losses on sparse masks."""
    batch = _oscillators()
    batch["mask"] = (torch.rand_like(batch["x"]) > 0.5).double()
    posterior = SequencePosterior(5, 3, 1, hidden=16, gauge=True, full_cov=True, gap_features=True,
                                  neighbor_context=True)
    model = LatentSDE(posterior, DriftNet(3, 1, 1, (32, 32)), GaussianDecoder(3, 5, 1), dim_z=3, n_samples=4)
    torch.manual_seed(0)
    history = fit(model, batch, n_iter=5, lr=3e-3, n_warmup=0, log_every=0)
    assert all(math.isfinite(h) for h in history)


def test_released_diffusion_dimension_starts_at_its_fixed_value_and_learns():
    """ReleaseDiffusion(3): g of the fixed dimension is 0.05 for the first 3 steps, still 0.05 right after the
    release (continuous), and learns after it."""
    batch = _oscillators()
    diffusion = ConstantDiagonalDiffusion(2, 0.1, fixed=[0.05, None])
    model = LatentSDE(SequencePosterior(5, 2, 1, hidden=16), DriftNet(2, 1, 1, (32, 32)), GaussianDecoder(2, 5, 1),
                      dim_z=2, n_samples=4, diffusion=diffusion)
    g0 = lambda: diffusion(torch.zeros(1, 2), torch.zeros(1, 1), torch.zeros(()))[0, 0].item()
    seen = []
    record = lambda step, model, loss: seen.append((step, g0(), bool(diffusion.is_fixed[0])))
    fit(model, batch, n_iter=3, lr=1e-2, n_warmup=0, log_every=0, callbacks=[ReleaseDiffusion(3), record])
    assert all(abs(g - 0.05) < 1e-12 for _, g, _ in seen) and not seen[-1][2] and seen[1][2]
    fit(model, batch, n_iter=5, lr=1e-2, n_warmup=0, log_every=0)
    assert abs(g0() - 0.05) > 1e-4


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
