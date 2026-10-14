import torch

from ..discovery.drift import DriftFCNN, ExactMotionModel, SparseIntegratorGLM, SparseMultioutputGLM
from ..discovery.features import SparsePolynomialFeatures, SparsePolynomialNeighbour1D
from ._reference import run_tests, svise_module, use_float64

use_float64()


def _duffing(x):
    x1, x2 = x[..., 0], x[..., 1]
    return torch.stack([x2, x1 - x1**3 - 0.35 * x2], dim=-1)


def test_polynomial_features_order_and_names():
    features = SparsePolynomialFeatures(2, degree=2, input_labels=["x", "y"])
    assert list(features.feature_names) == ["1", "x", "y", "x^2", "x y", "y^2"]
    x = torch.tensor([[2.0, 3.0]])
    torch.testing.assert_close(features(x), torch.tensor([[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]]))
    assert SparsePolynomialFeatures(2, degree=5).num_features == 21


def test_neighbour_features_shape():
    features = SparsePolynomialNeighbour1D(6, degree=2)
    assert features(torch.randn(4, 6)).shape == (4, 6, features.num_features)


def test_least_squares_initialisation_recovers_coefficients():
    """With exact derivatives, the drift prior starts at the true coefficients (raw feature basis)."""
    torch.manual_seed(0)
    x = 2 * torch.rand(200, 2) - 1
    drift = SparseMultioutputGLM(2, SparsePolynomialFeatures(2, degree=3), n_reparam_samples=4, tau=1e-5,
                                 train_x=x, train_y=_duffing(x))
    mean, _ = drift.coefficients()
    expected = torch.zeros(2, 10)
    expected[0, 2] = 1.0                                       # x2
    expected[1, 1], expected[1, 2], expected[1, 6] = 1.0, -0.35, -1.0  # x1, x2, x1^3
    torch.testing.assert_close(mean, expected, atol=2e-2, rtol=0)


def test_paired_samples_and_centered_parameterization():
    for parameterization in ("noncentered", "centered"):
        drift = SparseMultioutputGLM(2, SparsePolynomialFeatures(2, degree=2), n_reparam_samples=5, tau=1e-5,
                                     parameterization=parameterization)
        z = torch.randn(5, 7, 2)
        assert drift(torch.zeros(7), z).shape == (5, 7, 2)
        assert torch.isfinite(drift.kl_divergence())


def test_sparse_glm_matches_svise():
    torch.manual_seed(0)
    x = torch.randn(50, 2)
    torch.manual_seed(1)
    reference = svise_module("sde_learning").SparseMultioutputGLM(
        2, svise_module("variationalsparsebayes").SparsePolynomialFeatures(2, degree=3), 8, tau=1e-5,
        train_x=x, train_y=_duffing(x))
    torch.manual_seed(1)
    drift = SparseMultioutputGLM(2, SparsePolynomialFeatures(2, degree=3), 8, tau=1e-5, train_x=x, train_y=_duffing(x))
    z = torch.randn(8, 6, 2)
    torch.manual_seed(2)
    expected = reference(torch.zeros(6), z)
    torch.manual_seed(2)
    torch.testing.assert_close(drift(torch.zeros(6), z), expected)
    torch.testing.assert_close(drift.kl_divergence(), reference.kl_divergence())
    assert drift.feature_names == reference.feature_names


def test_integrator_glm_keeps_dx_equal_v():
    drift = SparseIntegratorGLM(4, SparsePolynomialFeatures(4, degree=2), n_reparam_samples=3, tau=1e-5)
    z = torch.randn(3, 5, 4)
    torch.testing.assert_close(drift(torch.zeros(5), z)[..., :2], z[..., 2:])


def test_exact_and_neural_drifts():
    exact = ExactMotionModel(lambda t, z: -z)
    z = torch.randn(3, 5, 2)
    torch.testing.assert_close(exact(torch.zeros(5), z), -z)
    neural = DriftFCNN(2, [16, 16], torch.nn.Tanh())
    assert neural(torch.linspace(0, 1, 5), z).shape == (3, 5, 2)


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
