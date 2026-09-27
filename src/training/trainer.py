import jax
import jax.numpy as jnp
import optax

from src.training.step import make_step


def linear_warmup(num_steps):
    """beta(step) rising linearly from 0 to 1 over num_steps, then staying at 1."""
    return lambda step: min(1.0, step / num_steps) if num_steps > 0 else 1.0


class Trainer:
    """
    Fits the Gaussian path (gp) and the SDE parameterization (sde) jointly by maximizing the ELBO.
    The Python-side loop (keys, beta schedule, minibatches, callbacks) lives here; the step itself is jitted (step.py).

    gp:  init_params(key), build(gp_params, G_fn) -> GaussianPathSDE
    sde: init_params(key), sample(sde_params, key) -> PriorSDE, kl_divergence(sde_params)
    loss_fn(params, key, batch, beta) -> (loss, terms), e.g. from losses.elbo.make_elbo_loss(gp, sde, ...)
    optimizers: {"gp": optax optimizer, "sde": optax optimizer}, one per parameter group
    callbacks: callables callback(step, params, loss, terms), run after every step
    """

    def __init__(self, gp, sde, loss_fn, optimizers, beta_schedule=None, callbacks=()):
        self.gp = gp
        self.sde = sde
        self.beta_schedule = beta_schedule or (lambda step: 1.0)
        self.callbacks = list(callbacks)

        # Two parameter groups: every leaf under params["gp"] / params["sde"] gets that group's optimizer
        def labels(params):
            return {group: jax.tree_util.tree_map(lambda _: group, sub) for group, sub in params.items()}
        self.optimizer = optax.multi_transform(optimizers, labels)
        self.step = make_step(loss_fn, self.optimizer)

    def init(self, key):
        k_gp, k_sde = jax.random.split(key)
        params = {"gp": self.gp.init_params(k_gp), "sde": self.sde.init_params(k_sde)}
        return params, self.optimizer.init(params)

    def fit(self, key, data, num_steps, batch_size=None, params=None, opt_state=None, start_step=0):
        """
        data: (t_obs, y_obs) with shapes (N,) and (N, M).
        batch_size: if given, a random minibatch of that size is drawn each step (pass num_data to the loss).
        params / opt_state: continue from a previous fit or checkpoint instead of initializing.
        """
        t_obs, y_obs = data
        key, k_init = jax.random.split(key)
        if params is None:
            params, opt_state = self.init(k_init)
        elif opt_state is None:
            opt_state = self.optimizer.init(params)

        for i in range(start_step, start_step + num_steps):
            key, k_step, k_batch = jax.random.split(key, 3)
            if batch_size is None:
                batch = (t_obs, y_obs)
            else:
                idx = jax.random.choice(k_batch, t_obs.shape[0], (batch_size,), replace=False)
                batch = (t_obs[idx], y_obs[idx])
            beta = jnp.asarray(self.beta_schedule(i))
            params, opt_state, loss, terms = self.step(params, opt_state, k_step, batch, beta)
            for callback in self.callbacks:
                callback(i, params, loss, terms)

        return params, opt_state
