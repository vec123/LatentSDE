import jax
import jax.numpy as jnp
import optax
from jax.nn import softplus

from src.distributions.learnable import init_gaussian_params
from src.distributions.divergences import kl_ig, kl_gamma
from src.distributions.variationalHorsehoe import variationalHorsehoe

# --- Execution ---
def fit_horseshoe():
    D = 10
    model = variationalHorsehoe(D,  tau_0 = 1000,
                 local_shape=2.0, local_rate=1.0,
                 global_shape=5, global_rate=0.5)
    key = jax.random.PRNGKey(42)

    # Dummy data: Sparse signal
    y = jnp.array([6.0, 0.0, 0.5, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0])
    y = y / jnp.std(y)
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
        if i % 200 == 0:
            print(f"Step {i}, Loss: {loss:.4f}, Global Scale Mean: {params['s_a']['mu'][0]:.4f}")
            print(f"Step {i}, Loss: {loss:.4f}, Theta Means: {params['theta']['mu']}")

    mu, std = model.get_uncertainty(params)
    print("\n--- Final Uncertainty Check ---")
    for i in range(D):
        print(f"Index {i}: Mean = {mu[i]:.4f}, Std Dev = {std[i]:.4f}")

if __name__ == "__main__":
    fit_horseshoe()
