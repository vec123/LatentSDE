"""
Evidence lower bound for state estimation with a Gaussian-path posterior (Course & Nair, Nature 2023):

    ELBO = sum_i E_q[log p(y_i | x(t_i))]
           - 1/2 int_0^T E_q || f_theta(t, x) - f_q(t, x) ||^2_{(G G^T)^-1} dt
           - KL(q(theta) || p(theta))

q is the linear SDE of a GaussianPathSDE with drift f_q(t, x) = A(t) x + b(t) and marginals N(m(t), S(t)).
f_theta is the drift of the prior SDE. Both SDEs share the diffusion G(t), which must be invertible.
If f_theta is random (e.g. horseshoe coefficients), sample theta once per step and close over it.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor, cho_solve


def gauss_legendre(t0, t1, num_nodes, num_intervals=1):
    """Composite Gauss-Legendre rule on [t0, t1]. Returns nodes and weights, each of shape (num_intervals * num_nodes,)."""
    xi, wi = np.polynomial.legendre.leggauss(num_nodes)
    edges = np.linspace(t0, t1, num_intervals + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    t = (mid[:, None] + half[:, None] * xi[None, :]).ravel()
    w = (half[:, None] * wi[None, :]).ravel()
    return jnp.asarray(t), jnp.asarray(w)


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


def residual_term(key, path, prior_drift, t_nodes, w_nodes, num_samples=32):
    """1/2 int E_q ||f_theta(t, x) - f_q(t, x)||^2_{(G G^T)^-1} dt: quadrature in t, reparameterized Monte Carlo in x."""
    def at_node(t, k):
        m, S = path.mean_fn(t), path.Sigma_fn(t)
        A, b = path.A_fn(t), path.b_fn(t)
        G = path.diffusion(t)
        x = m + jax.random.normal(k, (num_samples, m.shape[0])) @ jnp.linalg.cholesky(S).T
        r = jax.vmap(lambda xi: prior_drift(t, xi))(x) - (x @ A.T + b)
        Q_chol = cho_factor(G @ G.T, lower=True)
        return jnp.mean(jnp.sum(r * cho_solve(Q_chol, r.T).T, axis=-1))

    keys = jax.random.split(key, t_nodes.shape[0])
    return 0.5 * jnp.sum(w_nodes * jax.vmap(at_node)(t_nodes, keys))


def elbo_terms(key, path, prior_drift, t_obs, y_obs, H, R, t_nodes, w_nodes,
               kl_theta=0.0, num_samples=32, num_data=None):
    """
    The three ELBO terms, each as it enters with a minus sign except the log-likelihood.
    For a minibatch of observations, pass num_data (total number of observations) to rescale the log-likelihood.
    """
    log_likelihood = log_likelihood_term(path, t_obs, y_obs, H, R)
    if num_data is not None:
        log_likelihood = log_likelihood * num_data / t_obs.shape[0]
    return {
        "log_likelihood": log_likelihood,
        "residual": residual_term(key, path, prior_drift, t_nodes, w_nodes, num_samples),
        "kl": kl_theta,
    }


def elbo(key, path, prior_drift, t_obs, y_obs, H, R, t_nodes, w_nodes,
         kl_theta=0.0, num_samples=32, num_data=None, beta=1.0):
    """ELBO = log-likelihood - residual - beta * KL. Ramp beta from 0 to 1 for KL warm-up."""
    terms = elbo_terms(key, path, prior_drift, t_obs, y_obs, H, R, t_nodes, w_nodes,
                       kl_theta, num_samples, num_data)
    return terms["log_likelihood"] - terms["residual"] - beta * terms["kl"]
