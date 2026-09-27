import jax.numpy as jnp
from jax.nn import softplus


class DiagonalCovariance:
    """
    Learnable diagonal diffusion G = diag(g), g = softplus(raw_scale) + min_scale > 0, constant in time.
    The noise covariance is G G^T = diag(g^2).
    Pass the same G_fn to the GaussianPathSDE and use it for the prior SDE:
    the ELBO's path KL is only finite when both SDEs share the diffusion.
    """

    def __init__(self, D, init_scale=0.1, min_scale=1e-4):
        self.D = D
        self.init_scale = init_scale
        self.min_scale = min_scale

    def init_params(self):
        # Inverse softplus, so that g starts at init_scale
        raw = jnp.log(jnp.expm1(self.init_scale - self.min_scale))
        return {"raw_scale": jnp.full(self.D, raw)}

    def scales(self, params):
        return softplus(params["raw_scale"]) + self.min_scale

    def G(self, params):
        return jnp.diag(self.scales(params))

    def G_fn(self, params):
        """Time-independent G with signature (t,), as GaussianPathSDE expects."""
        G = self.G(params)
        return lambda t: G
