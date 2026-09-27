import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve


def expected_log_likelihood(y, m, S, H, R):
    """E_{x ~ N(m, S)}[log N(y | H x, R)] = log N(y | H m, R) - 1/2 tr(R^-1 H S H^T)."""
    y = jnp.atleast_1d(y)
    R_chol = cho_factor(R, lower=True)
    diff = y - H @ m
    log_det_R = 2.0 * jnp.sum(jnp.log(jnp.diag(R_chol[0])))
    mahalanobis = diff @ cho_solve(R_chol, diff)
    trace = jnp.trace(cho_solve(R_chol, H @ S @ H.T))
    return -0.5 * (mahalanobis + log_det_R + y.shape[0] * jnp.log(2 * jnp.pi) + trace)


def log_likelihood_term(path, t_obs, y_obs, H, R):
    """Sum of expected log-likelihoods of observations y_obs (N, M) at times t_obs (N,)."""
    def at_obs(t, y):
        return expected_log_likelihood(y, path.mean_fn(t), path.Sigma_fn(t), H, R)
    return jnp.sum(jax.vmap(at_obs)(t_obs, y_obs))
