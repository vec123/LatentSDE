import jax
import jax.numpy as jnp
import haiku as hk
from scipy.interpolate import make_lsq_spline, BSpline

from src.kernels.matern_kernels import matern_52_kernel
from src.sde.GaussianPaths.means.base import BaseMeanModel
from src.kernels.matern_kernels import kumaraswamy_warping, matern_52_kernel

    
class MaternMeanModel(BaseMeanModel):
    def __init__(self, time_points, Hm, length_scale, sigma_f = 1.0, 
                    T=1.0, alpha=1.0, beta=1.0, noise_var=1e-6): 

        """
        T: The maximum time interval [0, T]
        alpha, beta: Tunable parameters for the Kumaraswamy warping function
        """
        self.time_points = time_points
        self.Hm = Hm
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.T = T
        self.alpha = alpha
        self.beta = beta
        
        self.warped_time_points = kumaraswamy_warping(time_points, self.T, self.alpha, self.beta)
        
        # Precompute the kernel using warped coordinates
        # K_MM: (N, N)
        K_MM = self._compute_warped_kernel(time_points, time_points) 
        
        # Solve (K + sigma*I) * W = Hm
        self.weights = jax.scipy.linalg.solve(
            K_MM + noise_var * jnp.eye(time_points.shape[0]), 
            Hm, 
            assume_a='pos'
        )
        
    def _compute_warped_kernel(self, t1, t2):
        # Apply warping to inputs before kernel evaluation
        # Note: Depending on your specific implementation of matern_52_kernel, 
        # you may need to pass the warped times directly if the kernel expects scalars.
        w_t1 = kumaraswamy_warping(t1, self.T, self.alpha, self.beta)
        w_t2 = kumaraswamy_warping(t2, self.T, self.alpha, self.beta)
        
        # Reshape to ensure broadcasting compatibility
        return matern_52_kernel(w_t1[:, None], w_t2[None, :], self.length_scale, self.sigma_f**2)
        
    def __call__(self, t):
        # K(t, time_points): (1, N)
        kt = self._compute_warped_kernel(jnp.atleast_1d(t), self.time_points)
        return jnp.dot(kt, self.weights).squeeze()

