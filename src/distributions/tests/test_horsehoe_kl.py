import jax
import jax.numpy as jnp
import optax
from jax.nn import softplus

from src.distributions.variationalHorsehoe import variationalHorsehoe


def fit_horseshoe():
    D = 10
    # Instantiating model with updated parameter naming
    model = variationalHorsehoe(
        D,
        tau_0=1.0,
        local_shape=0.5,
        local_rate=1.0,
        global_shape=0.5,
        global_rate=1.0,
    )
    key, data_key = jax.random.split(jax.random.PRNGKey(42))

    # Sparse signal, observed with noise
    signal = jnp.array([6.0, 0.0, 0.5, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0])
    noise_std = jnp.sqrt(0.5)
    y = signal + noise_std * jax.random.normal(data_key, signal.shape)
    
    params = model.init_params(key)
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss_val, grads = jax.value_and_grad(model.loss)(params, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss_val

    for i in range(4001):
        params, opt_state, loss = step(params, opt_state)
        if i % 500 == 0:
            # Fixed 0-D scalar indexing: params['s_a']['mu'] is a scalar array
            global_mu = float(params["s_a"]["mu"])
            print(f"Step {i:4d} | Loss: {loss:8.4f} | Global Scale Mean: {global_mu:.4f}")

    mu, std = model.get_uncertainty(params)
    print("\n--- Final Uncertainty Check ---")
    for i in range(D):
        print(
            f"Index {i}: True = {signal[i]:5.2f} | Observed = {y[i]:5.2f} | "
            f"Mean = {mu[i]:5.2f} | Std = {std[i]:5.2f}"
        )


if __name__ == "__main__":
    fit_horseshoe()