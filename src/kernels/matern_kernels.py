
import jax.numpy as jnp
import jax.numpy as jnp


def kumaraswamy_warping(t, T, alpha, beta):
    # Normalize t to [0, 1] for the warping function
    t_norm = t / T
    return 1 - (1 - t_norm**alpha)**beta

def matern_52_kernel(t1, t2, length_scale, sigma):
    r = jnp.abs(t1 - t2)
    sqrt5_r_l = jnp.sqrt(5) * r / length_scale
    return sigma * (1 + sqrt5_r_l + (5 * r**2) / (3 * length_scale**2)) * jnp.exp(-sqrt5_r_l)

def matern_52_kernel_(t1, t2, length_scale=1.0, variance=1.0):
    """
    Computes the Matern 5/2 kernel between two time points (or arrays).
    
    :param t1: Time point or array (n,)
    :param t2: Time point or array (m,)
    :param length_scale: (ell) Controls the range of correlation
    :param variance: (sigma^2) Controls the magnitude of the signal
    :return: Kernel matrix (n, m)
    """
    # Ensure inputs are shaped for broadcasting (n, 1) and (1, m)
    t1 = jnp.atleast_1d(t1)[:, None]
    t2 = jnp.atleast_1d(t2)[None, :]
    
    r = jnp.abs(t1 - t2) / length_scale
    sqrt5_r = jnp.sqrt(5) * r
    
    kernel = variance * (1 + sqrt5_r + (5 * r**2) / 3) * jnp.exp(-sqrt5_r)
    return kernel

def matern_kernel(t1, t2, length_scale=1.0, nu=1.5):
    # Standard Matern 3/2 or 5/2 implementation
    dist = jnp.abs(t1 - t2)
    return (1 + jnp.sqrt(3)*dist/length_scale) * jnp.exp(-jnp.sqrt(3)*dist/length_scale)

