import jax
import jax.numpy as jnp

from src.sde.GaussianPaths.gaussian_path import GaussianPathSDE
from src.sde.GaussianPaths.means.matern_52_mean import MaternMeanModel
from src.sde.GaussianPaths.Variances.matern_52_var import MaternCovariance


class MaternGaussianPath:
    """
    Parameters -> Matern mean and covariance functions -> GaussianPathSDE, in the form the Trainer expects.

    Mean: MaternMeanModel interpolating the learnable anchor values Hm (num_support, D) at fixed support times.
    Covariance: MaternCovariance with learnable weights W_R (rotations) and W_L (eigenvalues) on fixed basis centers.
    Length scales and warping parameters are fixed.
    """

    def __init__(self, D, T, num_support=50, num_basis=40, mean_length_scale=1.0, cov_length_scale=1.0,
                 cov_type="full", alpha=1.0, beta=1.0, sigma=1.0, noise_var=1e-3, Hm_init=None):
        """Hm_init: initial anchor values (num_support, D), e.g. observations interpolated to t_support; zeros if None."""
        self.D = D
        self.T = T
        self.t_support = jnp.linspace(0.0, T, num_support)
        self.basis_centers = jnp.linspace(0.0, T, num_basis)
        self.mean_length_scale = mean_length_scale
        self.cov_length_scale = cov_length_scale
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.Hm_init = Hm_init
        self.cov = MaternCovariance(D, (0.0, T), type=cov_type, noise_var=noise_var)

    def init_params(self, key):
        k_R, k_L = jax.random.split(key)
        Hm = jnp.zeros((self.t_support.shape[0], self.D)) if self.Hm_init is None else jnp.asarray(self.Hm_init)
        num_weights = self.basis_centers.shape[0] + 1   # get_basis_vector prepends a constant
        return {
            "Hm": Hm,
            "W_R": 1e-2 * jax.random.normal(k_R, (self.cov.dim_R, num_weights)),
            "W_L": 1e-1 * jax.random.normal(k_L, (self.D, num_weights)),
        }

    def build(self, params, G_fn):
        mean = MaternMeanModel(self.t_support, params["Hm"], self.mean_length_scale,
                               T=self.T, alpha=self.alpha, beta=self.beta)

        def Sigma_fn(t):
            return self.cov.get_cov(t, params["W_R"], params["W_L"], self.basis_centers,
                                    self.cov_length_scale, self.alpha, self.beta, self.sigma)

        return GaussianPathSDE(mean, mean.get_dot_mu, Sigma_fn, jax.jacobian(Sigma_fn), G_fn)
