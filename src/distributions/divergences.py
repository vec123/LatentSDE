import jax
import jax.numpy as jnp

# --- Modules: Priors (KL Formulas) ---
def kl_ig(mu, log_sigma, a, b):
    var = jnp.exp(2 * log_sigma)
    return -0.5 * (jnp.log(2 * jnp.pi * var) + 1) - (a * jnp.log(b) - jax.scipy.special.gammaln(a) - a * mu - b * jnp.exp(-mu + var / 2))

def kl_gamma(mu, log_sigma, a, b):
    var = jnp.exp(2 * log_sigma)
    return -0.5 * (jnp.log(2 * jnp.pi * var) + 1) - (a * jnp.log(b) - jax.scipy.special.gammaln(a) + a * mu - b * jnp.exp(mu + var / 2))

