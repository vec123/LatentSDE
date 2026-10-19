import io
import math

import torch

from ..latent import ConstantDiagonalDiffusion, DriftNet, GaussianDecoder, LatentSDE, fit, make_batch
from ..posterior.kalman import GaussianPotentials, KalmanPosterior, transition, transition_shared
from ._reference import run_tests, use_float64

use_float64()

D, Z = 3, 3


def _batch():
    """Two trajectories at irregular times, some values unobserved; the second row padded (last time repeated)."""
    torch.manual_seed(1)
    t = torch.cumsum(torch.rand(2, 8) * 0.8 + 0.05, 1)
    t[1, 6:] = t[1, 5]
    x = torch.randn(2, 8, D)
    mask = (torch.rand(2, 8, D) < 0.6).double()
    mask[:, 0] = 1.0
    mask[1, 6:] = 0.0
    return make_batch(t, x * mask, mask)


def _model(potentials=None, hidden=(), extended=None, dim_mu=0, decoder_hidden=(), closed_form=True):
    torch.manual_seed(2)
    drift = DriftNet(Z, dim_mu, hidden=hidden)
    decoder = GaussianDecoder(Z, D, dim_mu, hidden=decoder_hidden, std_fixed=[0.3, 0.2, 0.4])
    with torch.no_grad():
        if not decoder_hidden:
            decoder.mean[0].weight[:, :Z].copy_(torch.eye(D) + 0.3 * torch.randn(D, Z))
            decoder.mean[0].bias.copy_(0.1 * torch.randn(D))
        drift.net[0].weight.mul_(0.5)
    model = LatentSDE(KalmanPosterior(Z, potentials, extended), drift, decoder, Z,
                      diffusion=ConstantDiagonalDiffusion(Z, 0.4), n_samples=64, n_quad=2048, closed_form=closed_form)
    with torch.no_grad():
        model.z0_mean.copy_(torch.tensor([0.2, -0.1, 0.3]))
        model.z0_logvar.copy_(torch.tensor([0.1, -0.5, 0.2]))
    return model


def _exact_log_likelihood(model, batch):
    """log p(x) of the linear-Gaussian model from the dense joint Gaussian of the observed values."""
    W, bias = model.drift.net[0].weight, model.drift.net[0].bias
    g = model.diffusion(torch.zeros(1, Z), torch.zeros(1, 0), torch.zeros(()))[0]
    C, dvec = model.decoder.mean[0].weight, model.decoder.mean[0].bias
    s = torch.nn.functional.softplus(model.decoder.raw_std)
    total = 0.0
    for t, x, mask in zip(batch["t"], batch["x"], batch["mask"]):
        A, c, Q = transition(W, bias, torch.diag(g ** 2), t[1:] - t[:-1])
        mu, K = [model.z0_mean], [[torch.diag(model.z0_logvar.exp())]]
        for k in range(len(t) - 1):
            mu.append(A[k] @ mu[-1] + c[k])
            row = [A[k] @ K[k][j] for j in range(k + 1)]
            row.append(A[k] @ K[k][k] @ A[k].T + Q[k])
            K.append(row)
        n = len(t)
        full = torch.zeros(n * Z, n * Z)
        for i in range(n):
            for j in range(i + 1):
                full[i * Z:(i + 1) * Z, j * Z:(j + 1) * Z] = K[i][j]
                full[j * Z:(j + 1) * Z, i * Z:(i + 1) * Z] = K[i][j].T
        H = torch.block_diag(*[C] * n)
        keep = mask.flatten() > 0
        mean = (H @ torch.cat(mu) + dvec.repeat(n))[keep]
        cov = (H @ full @ H.T)[keep][:, keep] + torch.diag(s.repeat(n)[keep] ** 2)
        total += torch.distributions.MultivariateNormal(mean, cov).log_prob(x.flatten()[keep])
    return total


def test_transition_matches_its_integrals():
    torch.manual_seed(0)
    F, b, GG, tau = 0.4 * torch.randn(Z, Z), torch.randn(Z), torch.diag(torch.rand(Z)), 1.3
    A, c, Q = transition(F, b, GG, torch.tensor(tau))
    s = torch.linspace(0, tau, 4001)
    E = torch.linalg.matrix_exp(F * s[:, None, None])
    w = torch.full_like(s, tau / 4000)
    w[0] = w[-1] = tau / 8000  # trapezoid
    torch.testing.assert_close(A, E[-1])
    torch.testing.assert_close(c, (w[:, None] * (E @ b)).sum(0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(Q, (w[:, None, None] * (E @ GG @ E.transpose(-1, -2))).sum(0), atol=1e-6, rtol=1e-6)


def test_shared_transitions_match_the_general_ones():
    torch.manual_seed(0)
    F, GG = 0.5 * torch.randn(Z, Z), torch.diag(torch.rand(Z))
    for tau in (torch.rand(4, 6) * 3, torch.tensor([0.0, 40.0])):  # the second one goes to the fallback
        b = torch.randn(*tau.shape, Z)
        for a, c in zip(transition_shared(F, b, GG, tau), transition(F, b, GG, tau)):
            torch.testing.assert_close(a, c)


def _terms_over_seeds(model, batch, n_seeds=8):
    """Mean and standard error over seeds of each loss term."""
    runs = []
    with torch.no_grad():
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            runs.append(model.loss(batch)[1])
    values = {k: torch.stack([r[k] for r in runs]) for k in runs[0]}
    return {k: (v.mean().item(), v.std().item() / math.sqrt(n_seeds)) for k, v in values.items()}


def _random_potentials(dim_mu=0):
    """A recognition network whose potentials depend on x and are not exact."""
    torch.manual_seed(3)
    pot = GaussianPotentials(D, Z, dim_mu, hidden=(8,))
    with torch.no_grad():
        pot.net[-1].weight.normal_(0, 0.3)
    return pot


def test_closed_form_elbo_is_the_log_likelihood():
    batch, model = _batch(), _model()
    with torch.no_grad():
        loss, terms = model.loss(batch)
    assert terms["residual"] > 0 and terms["kl0"] > 0
    torch.testing.assert_close(-loss * batch["mask"].sum(), _exact_log_likelihood(model, batch))


def test_closed_form_terms_match_the_monte_carlo_estimate():
    batch = _batch()
    torch.manual_seed(4)
    batch["mu"] = torch.randn(2, 2)
    model = _model(_random_potentials(2), dim_mu=2)
    with torch.no_grad():
        model.posterior.condition(batch)
        assert model.posterior.closed_form_terms() is not None
        _, closed = model.loss(batch)
    model.closed_form = False
    for key, (mean, err) in _terms_over_seeds(model, batch).items():
        assert abs(closed[key].item() - mean) < 4 * err + 1e-8, (key, closed[key].item(), mean, err)


def test_nonlinear_decoder_samples_the_marginals_at_the_steps():
    batch = _batch()
    model = _model(_random_potentials(), decoder_hidden=(8,))
    closed = _terms_over_seeds(model, batch)
    model.closed_form = False
    for key, (mean, err) in _terms_over_seeds(model, batch).items():
        c_mean, c_err = closed[key]
        assert abs(c_mean - mean) < 4 * math.hypot(err, c_err) + 1e-8, (key, c_mean, mean, err, c_err)


def test_nonlinear_drift_keeps_the_monte_carlo_loss():
    batch = _batch()
    model = _model(_random_potentials(), hidden=(8,))
    model.n_samples, model.n_quad = 4, 16
    model.posterior.condition(batch)
    assert model.posterior.closed_form_terms() is None
    losses = []
    for closed_form in (True, False):
        model.closed_form = closed_form
        torch.manual_seed(0)
        losses.append(model.loss(batch)[0])
    torch.testing.assert_close(*losses, rtol=0, atol=0)


def test_monte_carlo_elbo_with_exact_potentials_is_the_log_likelihood():
    batch, model = _batch(), _model(closed_form=False)
    exact = _exact_log_likelihood(model, batch).item()
    n = batch["mask"].sum().item()
    with torch.no_grad():
        values = []
        for seed in range(8):
            torch.manual_seed(seed)
            values.append(-model.loss(batch)[0].item() * n)
    values = torch.tensor(values)
    err = values.std() / math.sqrt(len(values))
    assert abs(values.mean() - exact) < 4 * err + 1e-3, (values.mean().item(), exact, err.item())


def test_extended_filter_matches_the_exact_one_for_a_linear_drift():
    batch = _batch()
    t_query = torch.sort(batch["t"][:, :1] + torch.rand(2, 30) * (batch["t"][:, -1:] - batch["t"][:, :1]), 1).values
    out = []
    for extended in (None, True):
        model = _model(extended=extended)
        model.posterior.condition(batch)
        m, L, _, _ = model.posterior(t_query)
        z = m + 0.1
        out.append((m, L, model.posterior.exact_drift(z, slice(None))))
    for a, b in zip(*out):
        torch.testing.assert_close(a, b)


def test_potentials_at_the_decoder_values_are_exact():
    batch, model = _batch(), _model()
    pot = GaussianPotentials(D, Z, init="identity")
    with torch.no_grad():
        pot.W.copy_(model.decoder.mean[0].weight)
        pot.log_s.copy_(torch.nn.functional.softplus(model.decoder.raw_std).log())
        pot.net[-1].bias.view(D, Z + 2)[:, Z].copy_(-model.decoder.mean[0].bias)  # y = x - d, x_scale 1
    lam, eta = pot(batch["x"], batch["mask"], batch["mu"])
    lam0, eta0 = model.posterior._decoder_potentials(batch["x"], batch["mask"])
    torch.testing.assert_close(lam, lam0)
    torch.testing.assert_close(eta, eta0)


def test_gradients_reach_the_prior_and_the_potentials():
    batch = _batch()
    for closed_form in (True, False):
        model = _model(GaussianPotentials(D, Z, hidden=(8,)), closed_form=closed_form)
        model.n_samples, model.n_quad = 4, 16
        model.loss(batch)[0].backward()
        pot = model.posterior.potentials
        for p in (model.drift.net[0].weight, model.diffusion.raw_g, model.z0_mean, model.z0_logvar, pot.W, pot.log_s,
                  pot.net[-1].weight, model.decoder.mean[0].weight):
            assert p.grad is not None and p.grad.abs().sum() > 0, closed_form


def test_extended_filter_trains_a_nonlinear_drift_and_survives_a_reload():
    batch = _batch()
    model = _model(GaussianPotentials(D, Z, hidden=(8,)), hidden=(8,))
    model.n_samples, model.n_quad = 4, 16
    losses = fit(model, batch, n_iter=3, lr=1e-2, n_warmup=0, log_every=0)
    assert all(math.isfinite(v) for v in losses)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    loaded.posterior.condition(batch)
    m, L, _, _ = loaded.posterior(batch["t"])
    model.posterior.condition(batch)
    torch.testing.assert_close(m, model.posterior(batch["t"])[0])


def test_filtered_posterior_uses_no_later_step():
    """filtered(t) sees only the steps at or before t: a change of later observations, or cutting them, changes nothing;
    after the last step it is the smoother's marginal there (no backward message), and between steps the SDE's forecast."""
    model, batch = _model(), _batch()
    t_query = batch["t"][:1, :4]  # the first row's first 4 steps
    model.posterior.condition(batch)
    base = model.posterior.filtered(t_query)[0]
    changed = {**batch, "x": batch["x"].clone(), "mask": batch["mask"].clone()}
    changed["x"][:, 4:], changed["mask"][:, 4:] = 5.0, 1.0
    model.posterior.condition(changed)
    torch.testing.assert_close(base, model.posterior.filtered(t_query)[0])
    cut = {k: (v[:, :4] if v.dim() > 1 and v.shape[1] == 8 else v) for k, v in batch.items()}
    model.posterior.condition(cut)
    torch.testing.assert_close(base, model.posterior.filtered(t_query)[0])
    # at the last step of a cut batch the filter equals the smoother
    torch.testing.assert_close(model.posterior.filtered(cut["t"][:, -1:])[0], model.posterior(cut["t"][:, -1:])[0])
    # a forecast: later than the last step, the filter moves by the prior drift only (the covariance grows)
    later = cut["t"][:, -1:] + 0.5
    m_late, L_late = model.posterior.filtered(later)
    m_last, L_last = model.posterior.filtered(cut["t"][:, -1:])
    assert (L_late @ L_late.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum() > (
        L_last @ L_last.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum()


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
