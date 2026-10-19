import jax
import jax.numpy as jnp

# --- 1. Linear 2D System: Ornstein-Uhlenbeck ---
def linear_2d(t, x):
    # Drift: A*x where A = [[-0.5, 0.2], [-0.2, -0.5]]
    A = jnp.array([[1, 0.0], [0.0, 1.0]])
    drift = A @ x
    # Diffusion: Constant diagonal noise
    diff = jnp.array([[0.03, 0.0], [0.0, 0.03]])
    return drift, diff

# --- 2. Nonlinear 2D System: Stochastic Double-Well ---
def nonlinear_2d(t, x):
    # Drift derived from potential V(x) = (x^2 - 1)^2
    # f(x) = -grad(V) = -4*x*(x^2 - 1)
    x1, x2 = x[0], x[1]
    drift = jnp.array([-4 * x1 * (x1**2 - 1) - 0.5 * x1, 
                       -4 * x2 * (x2**2 - 1) - 0.5 * x2])
    # State-dependent diffusion
    diff = jnp.array([[0.005, 0.0], [0.0, 0.005]])
    return drift, diff

def vanderpol_2d(t, x, mu=1.0):
    """
    Van der Pol oscillator in state-space form.
    x1_dot = x2
    x2_dot = mu * (1 - x1^2) * x2 - x1
    """
    x1, x2 = x[0], x[1]
    
    drift = jnp.array([
        x2, 
        mu * (1 - x1**2) * x2 - x1
    ])
    
    # Diffusion: Low-intensity noise
    diff = jnp.array([[0.05, 0.0], 
                      [0.0, 0.05]])
    
    return drift, diff