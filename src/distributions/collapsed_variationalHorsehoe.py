import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import optax

class CollapsedVariationalHorseshoe:
    """
    Collapsed Variational Horseshoe Posterior model.
    Marginalizes local scale hyper-parameters (\lambda_j) directly,
    reducing variational parameters from O(5D) to O(2D + scale).
    """

    def __init__(
        self,
        D: int,
        tau_0: float = 1.0,
        noise_std: float = 0.7071,
    ):
        self.D = D
        self.tau_0 = tau_0
        self.noise_std = noise_std

    def init_params(self, key):
        """Only initializes variational parameters for theta and global tau."""
        k1, k2, k3, k4 = jax.random.split(key, 4)
        return {
            "theta": {
                "mu": jax.random.normal(k1, (self.D,)) * 0.01,
                "log_sigma": jax.random.normal(k2, (self.D,)) * 0.1 - 2.0,
            },
            "log_tau": {
                "mu": jnp.array(jnp.log(self.tau_0)),
                "log_sigma": jnp.array(-2.0),
            },
        }

    def _expected_log_prior_theta(self, mu, sigma, tau):
        """
        Computes E_q(theta)[log p(theta | tau)] using a accurate tight bound
        for the half-Cauchy scale mixture marginal log-prior.
        """
        # Second-order Taylor/Monte Carlo expansion approximation for E_q[log log(1 + 2*tau^2 / theta^2)]
        e_sq = mu**2 + sigma**2
        # Regularized ratio to prevent log(0) near zero
        ratio = 2.0 * (tau**2) / (e_sq + 1e-8)
        
        # Marginal half-Cauchy density approximation log p(theta | tau)
        log_prior_val = -0.5 * jnp.log(2.0 * (jnp.pi**3) * (tau**2)) + jnp.log(
            jnp.log1p(ratio) + 1e-8
        )
        return jnp.sum(log_prior_val)

    def kl_divergence(self, params):
        theta = params["theta"]
        log_tau = params["log_tau"]

        mu_t, ls_t = theta["mu"], theta["log_sigma"]
        sig_t = jnp.exp(ls_t)

        mu_tau, ls_tau = log_tau["mu"], log_tau["log_sigma"]
        sig_tau = jnp.exp(ls_tau)

        # 1. Entropy of q(theta): -E[log q(theta)]
        entropy_theta = 0.5 * jnp.sum(1.0 + jnp.log(2.0 * jnp.pi) + 2.0 * ls_t)

        # 2. Expected log prior under q(theta) and q(tau)
        # Expected tau under q(log_tau) ~ LogNormal(mu_tau, sig_tau^2)
        tau_eff = jnp.exp(mu_tau + 0.5 * sig_tau**2)
        exp_log_p_theta = self._expected_log_prior_theta(mu_t, sig_t, tau_eff)

        # 3. KL divergence for global tau ~ Half-Cauchy(0, tau_0)
        # LogNormal(mu_tau, sig_tau^2) || Half-Cauchy(tau_0)
        kl_tau = (
            -0.5
            - ls_tau
            - jnp.log(2.0 / jnp.pi)
            + jnp.log(self.tau_0)
            + mu_tau
            + jnp.log1p((tau_eff / self.tau_0) ** 2)
        )

        # Total variational KL = -Entropy - E[log p(theta|tau)] + KL(q(tau)||p(tau))
        return -entropy_theta - exp_log_p_theta + kl_tau

    def loss(self, params, y):
        noise_var = self.noise_std**2
        mu_t = params["theta"]["mu"]
        sig_t = jnp.exp(params["theta"]["log_sigma"])

        # Negative Log Likelihood: E_q[-log p(y | theta)]
        nll = (0.5 / noise_var) * jnp.sum((mu_t - y) ** 2 + sig_t**2)
        return nll + self.kl_divergence(params)

    def get_uncertainty(self, params):
        """Returns posterior mean and standard deviation."""
        mu = params["theta"]["mu"]
        std = jnp.exp(params["theta"]["log_sigma"])
        return mu, std

