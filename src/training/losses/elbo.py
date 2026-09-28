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

from src.training.losses.likelihood import log_likelihood_term
from src.training.losses.residual import residual_term


def elbo_terms(key, path, prior_drift, t_obs, y_obs, H, R, quad,
               kl_theta=0.0, num_samples=32, num_data=None):
    """
    The three ELBO terms, each as it enters with a minus sign except the log-likelihood.
    quad: time quadrature for the residual, e.g. UnbiasedGaussLegendreQuad.
    For a minibatch of observations, pass num_data (total number of observations) to rescale the log-likelihood.
    """
    log_likelihood = log_likelihood_term(path, t_obs, y_obs, H, R)
    if num_data is not None:
        log_likelihood = log_likelihood * num_data / t_obs.shape[0]
    return {
        "log_likelihood": log_likelihood,
        "residual": residual_term(key, path, prior_drift, quad, num_samples),
        "kl": kl_theta,
    }


def elbo(key, path, prior_drift, t_obs, y_obs, H, R, quad,
         kl_theta=0.0, num_samples=32, num_data=None, beta=1.0):
    """ELBO = log-likelihood - residual - beta * KL. Ramp beta from 0 to 1 for KL warm-up."""
    terms = elbo_terms(key, path, prior_drift, t_obs, y_obs, H, R, quad,
                       kl_theta, num_samples, num_data)
    return terms["log_likelihood"] - terms["residual"] - beta * terms["kl"]


def make_elbo_loss(gp, sde, H, R, quad, num_samples=32, num_data=None):
    """
    Negative ELBO as loss_fn(params, key, batch, beta) -> (loss, terms), with params = {"gp": ..., "sde": ...}
    and batch = (t_obs, y_obs). quad: time quadrature for the residual, e.g. UnbiasedGaussLegendreQuad.

    gp:  build(gp_params, G_fn) -> GaussianPathSDE
    sde: sample(sde_params, key) -> PriorSDE, one reparameterized sample of the drift parameters and G,
         kl_divergence(sde_params) -> KL of the drift and noise priors
    The same sample of G is used in the residual and in the GaussianPathSDE's Lyapunov solve.
    """
    feature_normalizer = getattr(sde, "feature_normalizer", None)

    def loss_fn(params, key, batch, beta, feature_scales=None):
        k_sde, k_residual, k_features = jax.random.split(key, 3)
        if feature_normalizer is None:
            prior = sde.sample(params["sde"], k_sde)
        else:
            prior = sde.sample(params["sde"], k_sde, feature_scales)
        path = gp.build(params["gp"], prior.diffusion)
        t_obs, y_obs = batch
        terms = elbo_terms(k_residual, path, prior.drift, t_obs, y_obs, H, R, quad,
                           sde.kl_divergence(params["sde"]), num_samples, num_data)
        value = terms["log_likelihood"] - terms["residual"] - beta * terms["kl"]
        if feature_normalizer is not None:
            batch_variance = feature_normalizer.estimate_from_path(
                path, k_features, t_obs, num_samples)
            return -value, ({**terms, "beta": beta}, batch_variance)
        return -value, {**terms, "beta": beta}

    return loss_fn
