import jax
import jax.numpy as jnp


class PriorSDE:
    """
    Prior SDE dx = f(t, x) dt + G(t) dW for a given drift and diffusion.
    Shared by all drift parameterizations (sparse combinations of basis functions, neural networks, ...);
    the drift is typically one sample of its random parameters.
    """
    def __init__(self, drift_fn, G_fn):
        self.drift_fn = drift_fn
        self.G_fn = G_fn

    def drift(self, t, x):
        """f(t, x) of shape (D,)"""
        return self.drift_fn(t, x)

    def diffusion(self, t):
        """G(t) of shape (D, D)"""
        return self.G_fn(t)

    def sample_trajectory(self, x0, t_eval, key):
        """Euler-Maruyama path from x0 (D,) on the grid t_eval, shape (len(t_eval), D). vmap over x0 / key for many paths."""
        def step_fn(x, inputs):
            t, dt, k = inputs
            dW = jax.random.normal(k, x.shape) * jnp.sqrt(dt)
            x_new = x + self.drift(t, x) * dt + self.diffusion(t) @ dW
            return x_new, x_new

        dts = jnp.diff(t_eval)
        keys = jax.random.split(key, dts.shape[0])
        _, path = jax.lax.scan(step_fn, x0, (t_eval[:-1], dts, keys))
        return jnp.vstack([x0, path])
