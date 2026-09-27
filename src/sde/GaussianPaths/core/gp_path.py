import jax
import jax.numpy as jnp

class GaussianPathSDE:
    """
    Computes the SDE drift parameters (A(t), b(t)) and diffusion G(t) 
    for a target Gaussian Process path [mu(t), Sigma(t)].
    """
    def __init__(self, mean_fn, dmean_dt_fn, Sigma_fn, dSigma_dt_fn, G_fn):
        self.mean_fn = mean_fn
        self.dmean_dt_fn = dmean_dt_fn
        self.Sigma_fn = Sigma_fn
        self.dSigma_dt_fn = dSigma_dt_fn
        self.G_fn = G_fn
        
        # A(t) and b(t) are computed via the Lyapunov SDE relationship:
        # dSigma/dt = A*Sigma + Sigma*A.T + G*G.T
        self.A_fn = self._compute_A_spectral
        self.b_fn = self._compute_b
        
    def get_initial_state(self, key):
        """Samples X0 from the distribution at t=0."""
        mu0 = self.mean_fn(0.0)
        sigma0 = self.Sigma_fn(0.0)
        return jax.random.multivariate_normal(key, mu0, sigma0)
    
    def _compute_A_spectral(self, t):
        """Solves the Lyapunov equation for A(t) using spectral decomposition."""
        Sigma = self.Sigma_fn(t)
        dSigma_dt = self.dSigma_dt_fn(t)
        Q = self.G_fn(t) @ self.G_fn(t).T
        
        # R @ Lambda @ R.T = Sigma
        eigvals, R = jnp.linalg.eigh(Sigma)
        
        # Transform the Lyapunov equation into the eigenbasis:
        # B = R.T @ A @ R
        # RHS = R.T @ (dSigma/dt - G@G.T) @ R
        # B_ij * (lambda_i + lambda_j) = RHS_ij
        rhs = R.T @ (dSigma_dt - Q) @ R
        denom = eigvals[:, None] + eigvals[None, :]
        
        # B_ij = RHS_ij / (lambda_i + lambda_j)
        # Note: B_ii = RHS_ii / (2 * lambda_i)
        B = rhs / (denom + 1e-10) 
        
        return R @ B @ R.T

    def _compute_b(self, t):
        """b(t) = dmu/dt - A(t)mu(t)"""
        return self.dmean_dt_fn(t) - (self.A_fn(t) @ self.mean_fn(t))

    def drift(self, t, x):
        """Returns drift f(t, x) = A(t)x + b(t)"""
        return self.A_fn(t) @ x + self.b_fn(t)

    def diffusion(self, t):
        """Returns diffusion G(t)"""
        return self.G_fn(t)

    def sde_sigma(self, t, sigma):
        """Computes the evolution of the covariance matrix: dSigma/dt."""
        A = self.A_fn(t)
        G = self.G_fn(t)
        return A @ sigma + sigma @ A.T + G @ G.T
    
    def sample_trajectories(self, num_trajectories, t_eval, key):
        num_points = len(t_eval)
        dt = t_eval[1] - t_eval[0]
        
        def simulate_one(k):
            # 1. Sample initial state
            k, subkey = jax.random.split(k)
            x = self.get_initial_state(subkey)
            
            # 2. Prepare keys for the loop (one for each time step)
            # We fold_in the index to ensure every step gets a unique, uncorrelated key
            def step_fn(carry, i):
                x, k = carry
                t = t_eval[i-1]
                k, noise_key = jax.random.split(k)
                
                f = self.drift(t, x)
                g = self.diffusion(t)
                
                dW = jax.random.normal(noise_key, x.shape) * jnp.sqrt(dt)
                x_new = x + f * dt + g @ dW
                return (x_new, k), x_new

            # Use scan for performance and JIT-compatibility
            _, path = jax.lax.scan(step_fn, (x, k), jnp.arange(1, num_points))
            
            # Prepend initial state
            return jnp.vstack([x, path])

        keys = jax.random.split(key, num_trajectories)
        paths = jax.vmap(simulate_one)(keys)

        num_samples, num_steps, D = paths.shape
        # Convert to a list of (t, y) pairs internally so the rest of your code works
        paths = [(t_eval, paths[i]) for i in range(num_samples)]

        return paths