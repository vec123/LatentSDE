import jax
import jax.numpy as jnp

from distributions.fixed.learnable import init_gaussian_params
from distributions.fixed.divergences import kl_ig, kl_gamma

# --- Core Model: Horseshoe Posterior ---
class variationalHorsehoe:
    def __init__(self, D, tau_0 = 1,
                 local_shape=0.5, local_rate=1.0,
                 global_shape=0.5, global_rate=1.0):
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

        self.D = D
        self.tau_0 = tau_0
        self.local_shape = local_shape
        self.local_rate = local_rate
        self.global_shape = global_shape
        self.global_rate = global_rate

    def init_params(self, key):
            k1, k2, k3, k4, k5 = jax.random.split(key, 5)
            # Pass type='diagonal' to ensure 'log_var' is created
            return {
                'theta': init_gaussian_params(k1, self.D, type='diagonal'),
                's_a': init_gaussian_params(k2, 1, type='diagonal'),
                's_b': init_gaussian_params(k3, 1, type='diagonal'),
                'z_a': init_gaussian_params(k4, self.D, type='diagonal'),
                'z_b': init_gaussian_params(k5, self.D, type='diagonal')
            }

    def kl_divergence(self, params):
        # Helper to get the correct log_sigma from whatever init_params returned
        def get_log_sigma(p):
            if 'log_sigma' in p: return p['log_sigma']
            if 'log_var' in p: return 0.5 * p['log_var']
            raise KeyError("Parameters must contain either 'log_sigma' or 'log_var'")

        rate = 1.0 / (self.tau_0**2)
        # Extract parameters
        s_a, s_b, z_a, z_b, theta = params['s_a'], params['s_b'], params['z_a'], params['z_b'], params['theta']

        # Calculate log_sigma for all components
        ls_theta = get_log_sigma(theta)
        ls_s_a = get_log_sigma(s_a)
        ls_s_b = get_log_sigma(s_b)
        ls_z_a = get_log_sigma(z_a)
        ls_z_b = get_log_sigma(z_b)

        # Use these in your KL formulas
        # Half-Cauchy scales z = sqrt(z_a*z_b), s = sqrt(s_a*s_b), so the prior variance is z^2 s^2 = z_a*z_b*s_a*s_b
        log_z_s = z_a['mu'] + z_b['mu'] + s_a['mu'] + s_b['mu']
        z_sq_s_sq = jnp.exp(log_z_s)

        kl_theta = 0.5 * jnp.sum(
            log_z_s - 2 * ls_theta - 1 +
            (jnp.exp(2 * ls_theta) + theta['mu']**2) / (z_sq_s_sq + 1e-8)
        )

        kl_scales = (
            kl_gamma(s_a['mu'], ls_s_a, self.global_shape, rate) +
            kl_ig(s_b['mu'], ls_s_b, self.global_shape, self.global_rate) +
            jnp.sum(kl_gamma(z_a['mu'], ls_z_a, self.local_shape, self.local_rate)) +
            jnp.sum(kl_ig(z_b['mu'], ls_z_b, self.local_shape, self.local_rate))
        )
        return kl_theta + kl_scales

    def loss(self, params, y):
            # ELBO = NLL + KL
            nll = jnp.sum((params['theta']['mu'] - y)**2)
            total_loss = nll + self.kl_divergence(params)

            # Ensure the output is a 0-D scalar
            return jnp.sum(total_loss)

    def sample_prior(self, key):
        """Draws a sample from the Horseshoe prior distribution."""
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Draw scales from Gamma and Inverse-Gamma priors
        s_a = jax.random.gamma(k1, 0.5) / 1.0  # Gamma(0.5, 1.0)
        s_b = 1.0 / jax.random.gamma(k2, 0.5)  # IG(0.5, 1.0) -> 1/Gamma
        z_a = jax.random.gamma(k3, 0.5, shape=(self.D,))
        z_b = 1.0 / jax.random.gamma(k4, 0.5, shape=(self.D,))

        # Combine scales: theta ~ N(0, (z_a*z_b) * (s_a*s_b)), since sqrt(Gamma * IG) is half-Cauchy
        scale = jnp.sqrt((z_a * z_b) * (s_a * s_b))
        theta = jax.random.normal(k1, (self.D,)) * scale
        return theta

    def sample_posterior(self, key, params):
        """Draws a sample from the variational posterior q(theta)."""
        # theta ~ N(mu, sigma^2)
        mu = params['theta']['mu']
        sigma = jnp.exp(0.5 * params['theta']['log_var'])
        noise = jax.random.normal(key, shape=(self.D,))
        return mu + sigma * noise

    def get_uncertainty(self, params):
        """Returns the mean and standard deviation for the posterior."""
        mu = params['theta']['mu']
        # Convert log_var to standard deviation: std = exp(0.5 * log_var)
        std = jnp.exp(0.5 * params['theta']['log_var'])
        return mu, std
