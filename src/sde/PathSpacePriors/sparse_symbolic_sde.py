from src.distributions.variationalHorsehoe import variationalHorsehoe
from src.sde.covariance_parameterizations.diagonal import DiagonalCovariance
from src.sde.PathSpacePriors.core.prior_sde import PriorSDE
from src.sde.PathSpacePriors.sparse_symbolic.sde import SparsePolynomialDrift


class SparseSymbolicSDE:
    """
    Parameters -> sparse polynomial drift and diffusion -> PriorSDE, in the form the Trainer expects.

    Drift: SparsePolynomialDrift, with its coefficients Theta under a variational horseshoe prior.
    Diffusion: DiagonalCovariance, a fitted diagonal G without a prior.
    """

    def __init__(self, D, degree=3, include_bias=True, horseshoe_kwargs=None, noise_init_scale=0.1):
        """horseshoe_kwargs: passed to variationalHorsehoe (tau_0, local_shape, ...)."""
        self.drift = SparsePolynomialDrift(D, degree, include_bias)
        self.prior = variationalHorsehoe(self.drift.num_params, **(horseshoe_kwargs or {}))
        self.noise = DiagonalCovariance(D, init_scale=noise_init_scale)

    def init_params(self, key):
        return {"drift": self.prior.init_params(key), "noise": self.noise.init_params()}

    def sample(self, params, key):
        """One reparameterized sample of the drift coefficients -> PriorSDE."""
        theta = self.prior.sample_posterior(key, params["drift"])
        return PriorSDE(self.drift.drift_fn(theta), self.noise.G_fn(params["noise"]))

    def kl_divergence(self, params):
        return self.prior.kl_divergence(params["drift"])

    def coefficients(self, params):
        """Posterior mean of Theta, shape (D, num_features), e.g. for the EquationPrinter callback."""
        return params["drift"]["theta"]["mu"].reshape(self.drift.D, self.drift.num_features)
