import jax.numpy as jnp
from numpyro import distributions as dist
import numpyro

def basic_horseshoe(n, y=None):
    """
    Classic horseshoe prior.
    """
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(n)))
    tau = 1
    sigma = 1
    betas = numpyro.sample("betas", dist.Normal(0, lambdas*tau))
    y = numpyro.sample("y", dist.Normal(betas, sigma), obs=y)

