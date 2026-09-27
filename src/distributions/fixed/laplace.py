import jax.numpy as jnp
from numpyro import distributions as dist
import numpyro

def basic_laplace(n, y=None):
    """
    Like horseshoe but using Exponential priors on the lambda_i, thus yielding a Laplace prior on beta.
    """
    lambdas2 = numpyro.sample("lambdas", dist.Exponential(rate=0.5*jnp.ones(n)))
    lambdas = jnp.sqrt(lambdas2)

    tau = 1
    sigma = 1

    betas = numpyro.sample("betas", dist.Normal(0, lambdas*tau))
    y = numpyro.sample("y", dist.Normal(betas, sigma), obs=y)