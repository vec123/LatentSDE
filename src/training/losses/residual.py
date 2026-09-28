import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve


def residual_term(key, path, prior_drift, quad, num_samples=32):
    """
    1/2 int E_q ||f_theta(t, x) - f_q(t, x)||^2_{(G G^T)^-1} dt: the time integral by `quad`
    (e.g. UnbiasedGaussLegendreQuad), the expectation over x by reparameterized Monte Carlo.
    """
    def at_time(t, k):
        m, S = path.mean_fn(t), path.Sigma_fn(t)
        A, b = path.A_fn(t), path.b_fn(t)
        G = path.diffusion(t)
        x = m + jax.random.normal(k, (num_samples, m.shape[0])) @ jnp.linalg.cholesky(S).T
        r = jax.vmap(lambda xi: prior_drift(t, xi))(x) - (x @ A.T + b)
        Q_chol = cho_factor(G @ G.T, lower=True)
        return jnp.mean(jnp.sum(r * cho_solve(Q_chol, r.T).T, axis=-1))

    def integrand(times, k):
        return jax.vmap(at_time)(times, jax.random.split(k, times.shape[0]))

    return 0.5 * quad(integrand, key)
