import numpy as np
import torch

from ..discovery import learner
from ..discovery.learner import NeuralSDE, StateEstimator, sample_paths, solve_sde
from ..discovery.likelihood import IndepGaussLikelihood
from ._reference import run_tests, svise_module, use_float64

use_float64()
d = 2
T_OBS = torch.linspace(0.0, 10.0, 60)
X_TRUE = torch.stack([torch.cos(T_OBS), -torch.sin(T_OBS)], dim=-1)
NOISE = torch.full((d,), 1e-2)
Y_OBS = X_TRUE + NOISE.sqrt() * torch.randn(60, d, generator=torch.Generator().manual_seed(0))


def _seeded(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build(module):
    return module.SparsePolynomialSDE(d, (0.0, 10.0), degree=3, n_reparam_samples=8, G=torch.eye(d), num_meas=d,
                                      measurement_noise=NOISE, train_t=T_OBS, train_x=Y_OBS, n_quad=32, n_tau=60)


def test_elbo_matches_svise():
    _seeded()
    reference = _build(svise_module("sde_learning"))
    _seeded()
    sde = _build(learner)
    for beta in (0.0, 0.5, 1.0):
        _seeded(1)
        expected = reference.elbo(T_OBS, Y_OBS, beta, len(T_OBS))
        _seeded(1)
        torch.testing.assert_close(sde.elbo(T_OBS, Y_OBS, beta, len(T_OBS)), expected)


def test_closed_form_likelihood_matches_monte_carlo():
    _seeded()
    sde = _build(learner)
    G = torch.tensor([[1.0, 0.5]])
    likelihood = IndepGaussLikelihood(G, 1, torch.tensor([0.05]))
    t, y = T_OBS[:10], Y_OBS[:10] @ G.T
    exact = likelihood.mean_log_likelihood(t, y, sde.marginal_sde, 1)
    z = sde.marginal_sde.generate_samples(t, 200_000)
    mc = torch.distributions.Normal(z @ G.T, 0.05**0.5).log_prob(y).sum(-1).mean()
    assert abs(exact - mc) < 1e-2


def test_short_training_run_and_simulation():
    _seeded()
    sde = _build(learner)
    optimizer = torch.optim.Adam([{"params": sde.state_params()}, {"params": sde.sde_params(), "lr": 1e-2}], lr=1e-3)
    losses = []
    for step in range(60):
        optimizer.zero_grad()
        loss = -sde.elbo(T_OBS, Y_OBS, 1.0, len(T_OBS))  # fixed beta: a warm-up would raise the loss on purpose
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(np.isfinite(losses)) and np.mean(losses[-10:]) < np.mean(losses[:10])
    sde.resample_sde_params(4)
    paths = solve_sde(sde, X_TRUE[:1].repeat(4, 1), torch.linspace(0, 1, 11))
    assert paths.shape == (11, 4, d) and torch.isfinite(paths).all()


def test_state_estimator_and_neural_sde_elbos_are_finite():
    _seeded()
    common = dict(d=d, t_span=(0.0, 10.0), n_reparam_samples=4, G=torch.eye(d), measurement_noise=NOISE,
                  train_t=T_OBS, train_x=Y_OBS, n_quad=16, n_tau=40)
    estimator = StateEstimator(drift=lambda t, z: torch.stack([z[..., 1], -z[..., 0]], -1), num_meas=d,
                               Q_diag=torch.full((d,), 1e-3), **common)
    neural = NeuralSDE(drift_layer_description=[16, 16], nonlinearity=torch.nn.Tanh(), Q_diag=torch.full((d,), 1e-2),
                       **common)
    for sde in (estimator, neural):
        assert torch.isfinite(sde.elbo(T_OBS, Y_OBS, 1.0, len(T_OBS)))


def test_sample_paths_start_from_the_variational_posterior():
    _seeded()
    common = dict(d=d, t_span=(0.0, 10.0), n_reparam_samples=4, G=torch.eye(d), measurement_noise=NOISE,
                  train_t=T_OBS, train_x=Y_OBS, n_quad=16, n_tau=40)
    neural = NeuralSDE(drift_layer_description=[16, 16], nonlinearity=torch.nn.Tanh(), Q_diag=torch.full((d,), 1e-16), **common)
    t, n = torch.linspace(0.0, 1.0, 11), 4000
    for sde in (_build(learner), neural):
        sde.eval()
        paths = sample_paths(sde, t, n)
        assert paths.shape == (11, n, d) and torch.isfinite(paths).all()
        sd = sde.marginal_sde.K(t[:1])[0].diagonal().sqrt()
        torch.testing.assert_close(paths[0].mean(0), sde.marginal_sde.mean(t[:1])[0], atol=5 * sd.max().item() / n**0.5, rtol=0)
        torch.testing.assert_close(paths[0].std(0), sd, rtol=0.1, atol=0)
    # fixed weights and no diffusion: paths from the same start are the same
    x0 = paths[0, :1].repeat(3, 1)
    neural.resample_sde_params(3)
    same = solve_sde(neural, x0, t)
    torch.testing.assert_close(same[-1], same[-1, :1].expand(3, -1))


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
