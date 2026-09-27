import jax
import jax.numpy as jnp
import haiku as hk
from scipy.interpolate import make_lsq_spline, BSpline

from src.kernels.matern_kernels import matern_52_kernel
from src.models.NNs import nn_forward

class MaternMeanModel(BaseMeanModel):
    def __init__(self, time_points, Hm, length_scale, sigma_f, noise_var=1e-6):
        self.time_points = time_points
        self.Hm = Hm
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        
        # Precompute the weights for the GP interpolation
        # K_MM: (N, N)
        K_MM = matern_52_kernel(time_points, time_points, length_scale, sigma_f**2)
        # Solve (K + sigma*I) * W = Hm
        self.weights = jax.scipy.linalg.solve(
            K_MM + noise_var * jnp.eye(time_points.shape[0]), 
            Hm, 
            assume_a='pos'
        )
        
    def __call__(self, t):
        # K(t, time_points): (1, N)
        kt = matern_52_kernel(t, self.time_points, self.length_scale, self.sigma_f**2)
        return jnp.dot(kt, self.weights)
    


class NNMeanModel(BaseMeanModel):
    def __init__(self, time_points, Hm, in_dim, out_dim, hidden_sizes, key=jax.random.PRNGKey(42)):
        self.Hm = Hm
        self.in_dim = in_dim
        
        # INCREASED frequency range to cover 4*pi
        def fourier_features(t):
            # Using 10 frequencies that span the period of the data
            freqs = jnp.linspace(0.1, 2.0, 10) 
            return jnp.concatenate([jnp.sin(freqs * t), jnp.cos(freqs * t)], axis=-1)

        def model_fn(t):
            x = fourier_features(t)
            initializer = hk.initializers.VarianceScaling(2.0, mode='fan_in', distribution='truncated_normal')
            model = hk.Sequential([
                hk.Linear(hidden_sizes[0], w_init=initializer),
                jax.nn.silu,
                hk.Linear(hidden_sizes[1], w_init=initializer),
                jax.nn.silu,
                hk.Linear(out_dim, w_init=initializer)
            ])
            return model(x)
        
        self.transformed = hk.transform(model_fn)
        self.apply_fn = self.transformed.apply 
        
        # Initialize with dummy input
        dummy_input = jnp.zeros((1,)) 
        self.params = self.transformed.init(key, dummy_input)
        
    def __call__(self, t):
        return self.apply_fn(self.params, None, jnp.atleast_1d(t))
    

import jax
import jax.numpy as jnp

class JAXBSpline:
    def __init__(self, knots, degree):
        self.knots = jnp.array(knots)
        self.k = degree
        self.num_basis = len(knots) - degree - 1

    def basis(self, t, i, k):
        """Cox-de Boor recursion."""
        # Base case: degree 0
        if k == 0:
            return jnp.where((t >= self.knots[i]) & (t < self.knots[i+1]), 1.0, 0.0)
        
        # Recursive step
        denom1 = self.knots[i+k] - self.knots[i]
        term1 = jnp.where(denom1 != 0, ((t - self.knots[i]) / denom1) * self.basis(t, i, k-1), 0.0)
        
        denom2 = self.knots[i+k+1] - self.knots[i+1]
        term2 = jnp.where(denom2 != 0, ((self.knots[i+k+1] - t) / denom2) * self.basis(t, i+1, k-1), 0.0)
        
        return term1 + term2

    def get_design_matrix(self, t):
        # Evaluate all basis functions at all time points t
        return jnp.stack([self.basis(t, i, self.k) for i in range(self.num_basis)], axis=-1)

class BSplineMeanModel:
    def __init__(self, t, y, num_knots=5, degree=3):
        t = jnp.asarray(t).flatten()
        y = jnp.asarray(y).flatten()
        
        # Setup knots
        knots = jnp.linspace(t.min(), t.max(), num_knots)
        self.knots_padded = jnp.concatenate([
            jnp.repeat(t.min(), degree), knots, jnp.repeat(t.max(), degree)
        ])
        
        self.spline_basis = JAXBSpline(self.knots_padded, degree)
        
        # Least Squares fit
        phi = self.spline_basis.get_design_matrix(t)
        self.coeffs, _, _, _ = jnp.linalg.lstsq(phi, y, rcond=None)

    def __call__(self, t):
        # Now this works perfectly with vmap!
        phi_t = self.spline_basis.get_design_matrix(t)
        return jnp.dot(phi_t, self.coeffs)
    
    def get_dot_mu(self, t):
        # You can now take the gradient directly!
        return jax.grad(self.__call__)(t)