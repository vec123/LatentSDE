import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import optax
from src.distributions.collapsed_variationalHorsehoe import CollapsedVariationalHorseshoe

# --- Execution Test Loop ---
def fit_collapsed_horseshoe():
    D = 10
    model = CollapsedVariationalHorseshoe(D=D, tau_0=1.0)
    key, data_key = jax.random.split(jax.random.PRNGKey(42))

    # Sparse signal test
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

    print("--- Training Collapsed Variational Horseshoe ---")
    for i in range(4001):
        params, opt_state, loss = step(params, opt_state)
        if i % 1000 == 0:
            tau_val = float(jnp.exp(params["log_tau"]["mu"]))
            print(f"Step {i:4d} | Loss: {loss:8.4f} | Estimated Tau: {tau_val:.4f}")

    mu, std = model.get_uncertainty(params)
    print("\n--- Final Recovery Check ---")
    for i in range(D):
        print(
            f"Index {i:2d}: True = {signal[i]:5.2f} | Observed = {y[i]:5.2f} | "
            f"Mean = {mu[i]:5.2f} | Std = {std[i]:5.2f}"
        )


if __name__ == "__main__":
    fit_collapsed_horseshoe()