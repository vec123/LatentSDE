import jax
import jax.numpy as jnp

from src.distributions.learnable import init_gaussian_params
from src.distributions.divergences import kl_ig, kl_gamma

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

"""
        A Variational Horseshoe Posterior model for sparse feature selection.

        The Horseshoe prior models sparse coefficients by expressing variances as the product
        of a global scale parameter (governing overall sparsity) and local scale parameters
        (governing individual variable shrinkage). This implementation approximates the half-Cauchy
        distributions hierarchically using Gamma and Inverse-Gamma scale mixtures.

        ========================================================================================
        HYPERPARAMETER TUNING GUIDE
        ========================================================================================

        1. tau_0 (Global Scale Base Guess)
        ----------------------------------------------------------------------------------------
        Acts as a baseline scaling factor for the global shrinkage parameter.
        * HIGHER values: Relaxes overall shrinkage. It acts as an a priori belief that the
          true signal vector contains many non-zero entries or very large signals.
        * LOWER values: Forces severe, aggressive global shrinkage across all parameters,
          demanding much stronger evidence in the data to pull any parameter away from zero.

        2. local_shape & local_rate (Local Scale Hyperparameters)
        ----------------------------------------------------------------------------------------
        Governs the local variances ($z_a, z_b$), dictating the behavior of individual features.
        * Standard Horseshoe equivalence: Achieved when shape = 0.5 and rate = 1.0.
        * LOWER shape (< 1.0, e.g., 0.5): Creates an infinitely tall "spike" at zero and
          extremely heavy tails. This provides the classic "horseshoe" behavior: it ruthlessly
          crushes noise to zero while leaving true, large signals completely un-shrunk.
        * HIGHER shape (> 2.0): Smooths out the spike at zero and thins the tails. The prior
          mass behaves more like a normal/Ridge prior. It allows small, weak signals to be
          retained/recovered, but at the cost of losing clean, hard-zero sparsity on noise.
        * Adjusting rate: Lowering the rate shifts the local distribution mass outward,
          allowing individual features to grow more easily if the data supports them.

        3. global_shape & global_rate (Global Shrinkage Hyperparameters)
        ----------------------------------------------------------------------------------------
        Governs the global variance ($s_a, s_b$), dictating the expected overall sparsity.
        * Standard Horseshoe equivalence: Achieved when shape = 0.5 and rate = 1.0.
        * LOWER shape (< 1.0): Pulls the global scale close to zero. This enforces an
          assumption of extreme sparsity (e.g., only a few non-zero weights out of hundreds).
        * HIGHER shape (> 1.0): Allows the global scale to safely grow larger. Use this if
          you expect a denser model where a larger percentage of features are active.
        * HIGHER global_rate: Increases the rate parameter of the global Inverse-Gamma,
          which squashes the global scale down closer to zero, amplifying overall shrinkage.
"""


# --- Core Model: Horseshoe Posterior ---
class VariationalHorseshoe:
    """Variational Horseshoe Posterior model for sparse feature selection."""

    def __init__(
        self,
        D: int,
        tau_0: float = 1.0,
        local_shape: float = 0.5,
        local_rate: float = 1.0,
        global_shape: float = 0.5,
        global_rate: float = 1.0,
        noise_std: float = 0.7071,
    ):
        self.D = D
        self.tau_0 = tau_0
        self.local_shape = local_shape
        self.local_rate = local_rate
        self.global_shape = global_shape
        self.global_rate = global_rate
        self.noise_std = noise_std

    def init_params(self, key):
        """Initializes variational parameters (means and log-variances)."""
        keys = jax.random.split(key, 5)

        def make_param(k, shape):
            k1, k2 = jax.random.split(k)
            return {
                "mu": jax.random.normal(k1, shape) * 0.01,
                "log_var": jax.random.normal(k2, shape) * 0.1 - 2.0,
            }

        return {
            "theta": make_param(keys[0], (self.D,)),
            "z_a": make_param(keys[1], (self.D,)),
            "z_b": make_param(keys[2], (self.D,)),
            "s_a": make_param(keys[3], ()),
            "s_b": make_param(keys[4], ()),
        }

    def _get_log_sigma(self, p):
        if "log_sigma" in p:
            return p["log_sigma"]
        if "log_var" in p:
            return 0.5 * p["log_var"]
        raise KeyError(
            "Parameters must contain either 'log_sigma' or 'log_var'"
        )

    def _kl_lognormal_gamma(self, mu, log_sigma, alpha, beta):
        """KL( LogNormal(mu, sigma^2) || Gamma(alpha, beta) )"""
        sigma = jnp.exp(log_sigma)
        return (
            -0.5
            - log_sigma
            - gammaln(alpha)
            + alpha * jnp.log(beta)
            + (alpha - 1.0) * mu
            + beta * jnp.exp(mu + 0.5 * sigma**2)
        )

    def _kl_lognormal_ig(self, mu, log_sigma, alpha, beta):
        """KL( LogNormal(mu, sigma^2) || InverseGamma(alpha, beta) )"""
        sigma = jnp.exp(log_sigma)
        return (
            -0.5
            - log_sigma
            - gammaln(alpha)
            + alpha * jnp.log(beta)
            - (alpha + 1.0) * mu
            + beta * jnp.exp(-mu + 0.5 * sigma**2)
        )

    def kl_divergence(self, params):
        s_a, s_b = params["s_a"], params["s_b"]
        z_a, z_b = params["z_a"], params["z_b"]
        theta = params["theta"]

        ls_theta = self._get_log_sigma(theta)
        ls_s_a = self._get_log_sigma(s_a)
        ls_s_b = self._get_log_sigma(s_b)
        ls_z_a = self._get_log_sigma(z_a)
        ls_z_b = self._get_log_sigma(z_b)

        # Expected log variance under Log-Normal approximations
        sum_mu = z_a["mu"] + z_b["mu"] + s_a["mu"] + s_b["mu"]
        sum_var = (
            jnp.exp(2 * ls_z_a)
            + jnp.exp(2 * ls_z_b)
            + jnp.exp(2 * ls_s_a)
            + jnp.exp(2 * ls_s_b)
        )
        exponent = jnp.clip(-sum_mu + 0.5 * sum_var, a_min=-50.0, a_max=50.0)
        E_inv_var = jnp.exp(exponent)

        # KL for Gaussian theta parameters
        var_theta = jnp.exp(2 * ls_theta)
        kl_theta = 0.5 * jnp.sum(
            sum_mu - 2 * ls_theta - 1.0 + (var_theta + theta["mu"] ** 2) * E_inv_var
        )

        global_rate_a = 1.0 / (self.tau_0**2)

        # KL divergence terms for hyper-priors
        kl_scales = (
            self._kl_lognormal_gamma(
                s_a["mu"], ls_s_a, self.global_shape, global_rate_a
            )
            + self._kl_lognormal_ig(
                s_b["mu"], ls_s_b, self.global_shape, self.global_rate
            )
            + jnp.sum(
                self._kl_lognormal_gamma(
                    z_a["mu"], ls_z_a, self.local_shape, self.local_rate
                )
            )
            + jnp.sum(
                self._kl_lognormal_ig(
                    z_b["mu"], ls_z_b, self.local_shape, self.local_rate
                )
            )
        )

        return kl_theta + kl_scales

    def loss(self, params, y):
        noise_var = self.noise_std**2
        ls_theta = self._get_log_sigma(params["theta"])
        var_theta = jnp.exp(2 * ls_theta)

        nll = (0.5 / noise_var) * jnp.sum((params["theta"]["mu"] - y) ** 2 + var_theta)
        return nll + self.kl_divergence(params)

    def sample_prior(self, key):
        """Draws a sample from the Horseshoe prior distribution."""
        k_sa, k_sb, k_za, k_zb, k_theta = jax.random.split(key, 5)

        global_rate_a = 1.0 / (self.tau_0**2)

        s_a = jax.random.gamma(k_sa, self.global_shape) / global_rate_a
        s_b = self.global_rate / jax.random.gamma(k_sb, self.global_shape)

        z_a = (
            jax.random.gamma(k_za, self.local_shape, shape=(self.D,))
            / self.local_rate
        )
        z_b = self.local_rate / jax.random.gamma(
            k_zb, self.local_shape, shape=(self.D,)
        )

        scale = jnp.sqrt((z_a * z_b) * (s_a * s_b))
        return jax.random.normal(k_theta, (self.D,)) * scale

    def sample_posterior(self, key, params):
        """Draws a sample from the variational posterior q(theta)."""
        mu = params["theta"]["mu"]
        std = jnp.exp(self._get_log_sigma(params["theta"]))
        return mu + std * jax.random.normal(key, shape=(self.D,))

    def get_uncertainty(self, params):
        """Returns the mean and standard deviation for the posterior."""
        mu = params["theta"]["mu"]
        std = jnp.exp(self._get_log_sigma(params["theta"]))
        return mu, std