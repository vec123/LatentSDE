import jax
import optax


def make_step(loss_fn, optimizer):
    """
    Pure, jitted optimization step:
    (params, opt_state, key, batch, beta) -> (params, opt_state, loss, terms).
    beta is passed as an argument so that the warm-up schedule does not trigger recompilation.
    """
    @jax.jit
    def step(params, opt_state, key, batch, beta):
        (loss, terms), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, batch, beta)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, terms

    return step
