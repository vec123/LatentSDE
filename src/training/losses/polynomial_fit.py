import jax
import jax.numpy as jnp


def lagrange_basis(nodes, s):
    """Evaluate the Lagrange basis at ``s`` using the stable barycentric formula.

    The barycentric weights are computed in log space, avoiding the very large or
    small products in the direct Lagrange formula. ``nodes`` must be distinct.
    Returns an array with the same shape as ``nodes``.
    """
    differences = nodes[:, None] - nodes[None, :]
    off_diagonal = ~jnp.eye(nodes.shape[0], dtype=bool)
    safe_differences = jnp.where(off_diagonal, differences, 1.0)

    log_abs_weights = -jnp.sum(
        jnp.where(off_diagonal, jnp.log(jnp.abs(safe_differences)), 0.0), axis=1
    )
    signs = jnp.prod(jnp.where(off_diagonal, jnp.sign(safe_differences), 1.0), axis=1)
    weights = signs * jnp.exp(log_abs_weights - jnp.max(log_abs_weights))

    delta = s - nodes
    tolerance = 8.0 * jnp.finfo(nodes.dtype).eps * jnp.maximum(1.0, jnp.abs(s))
    at_node = jnp.abs(delta) <= tolerance
    safe_delta = jnp.where(at_node, 1.0, delta)
    terms = weights / safe_delta
    basis = terms / jnp.sum(terms)

    # At a node the interpolant is exactly that node's value; use the limiting
    # basis vector directly instead of relying on a near-singular division.
    node_basis = jax.nn.one_hot(jnp.argmax(at_node), nodes.shape[0], dtype=nodes.dtype)
    return jnp.where(jnp.any(at_node), node_basis, basis)


def piecewise_interpolate(values, ref_nodes, t0, t1, num_intervals, t):
    """
    Evaluates at times t (M,) the piecewise polynomial that interpolates `values` at the composite Gauss-Legendre
    nodes of quadrature.gauss_legendre(t0, t1, len(ref_nodes), num_intervals), one polynomial per interval.
    ``ref_nodes`` are the Gauss-Legendre nodes on [-1, 1]. Every interval uses the same
    nodes in its local coordinate, so the barycentric weights are shared.
    """
    y = values.reshape(num_intervals, -1)
    width = (t1 - t0) / num_intervals
    k = jnp.clip(jnp.floor((t - t0) / width).astype(jnp.int32), 0, num_intervals - 1)
    s = 2.0 * (t - (t0 + k * width)) / width - 1.0
    basis = jax.vmap(lambda si: lagrange_basis(ref_nodes, si))(s)
    return jnp.sum(y[k] * basis, axis=-1)
