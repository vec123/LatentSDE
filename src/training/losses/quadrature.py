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
