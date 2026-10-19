import jax
import jax.numpy as jnp

def euler_maruyama_step(system_fn, t, x, dt, key):
    drift, diff = system_fn(t, x)
    dw = jax.random.normal(key, shape=x.shape) * jnp.sqrt(dt)
    return x + drift * dt + diff @ dw

def simulate_batch(system_fn, x0_batch, t_span, dt, key):
    num_steps = int((t_span[1] - t_span[0]) / dt)
    
    def step_fn(x, k):
        return euler_maruyama_step(system_fn, 0.0, x, dt, k), None

    # Vmap over initial conditions
    def run_single(x0, rng):
        keys = jax.random.split(rng, num_steps)
        # Scan over time
        _, traj = jax.lax.scan(lambda x, k: (euler_maruyama_step(system_fn, 0.0, x, dt, k), x), x0, keys)
        return traj

    rngs = jax.random.split(key, x0_batch.shape[0])
    return jax.vmap(run_single)(x0_batch, rngs)

