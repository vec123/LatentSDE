import jax
import jax.numpy as jnp
import numpy as np


def gauss_legendre(t0, t1, num_nodes, num_intervals=1):
    """Composite Gauss-Legendre rule on [t0, t1]. Returns nodes and weights, each of shape (num_intervals * num_nodes,)."""
    xi, wi = np.polynomial.legendre.leggauss(num_nodes)
    edges = np.linspace(t0, t1, num_intervals + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    t = (mid[:, None] + half[:, None] * xi[None, :]).ravel()
    w = (half[:, None] * wi[None, :]).ravel()
    return jnp.asarray(t), jnp.asarray(w)


def stratified_times(key, t0, t1, num_times):
    """One uniform random time in each of num_times equal slices of [t0, t1] (stratified sampling), shape (num_times,)."""
    width = (t1 - t0) / num_times
    return t0 + (jnp.arange(num_times) + jax.random.uniform(key, (num_times,))) * width
