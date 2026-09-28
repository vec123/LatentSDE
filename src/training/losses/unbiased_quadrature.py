"""
Unbiased Gauss-Legendre quadrature (L.-F. Lee, Econometric Theory 17, 2001), as used by Course & Nair (2023):

    int_{t0}^{t1} f dt  ~  I_q + (t1 - t0) * mean_i [ f(tau_i) - p(tau_i) ]

I_q:   composite Gauss-Legendre quadrature of f (quadrature.py)
p:     the piecewise polynomial through f at the quadrature nodes (polynomial_fit.py), which I_q integrates exactly
tau_i: stratified random times (quadrature.py)

Because I_q is the exact integral of p, the correction has mean zero: the estimate is unbiased. Its variance is
small because f - p is small. This holds even when f at the nodes is itself a Monte Carlo estimate.
"""
import jax
import jax.numpy as jnp
import numpy as np

from src.training.losses.polynomial_fit import piecewise_interpolate
from src.training.losses.quadrature import gauss_legendre, stratified_times


class UnbiasedGaussLegendreQuad:
    def __init__(self, t0, t1, num_nodes, num_intervals=1, num_mc_times=0):
        """num_mc_times = 0 gives plain composite Gauss-Legendre quadrature (deterministic in t, slightly biased)."""
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.num_intervals = num_intervals
        self.num_mc_times = num_mc_times
        self.t_nodes, self.w_nodes = gauss_legendre(t0, t1, num_nodes, num_intervals)
        self.ref_nodes = jnp.asarray(np.polynomial.legendre.leggauss(num_nodes)[0])

    def __call__(self, f, key):
        """f(times, key) -> integrand values of shape (len(times),). Returns the estimate of int_{t0}^{t1} f dt."""
        k_nodes, k_times, k_mc = jax.random.split(key, 3)
        f_nodes = f(self.t_nodes, k_nodes)
        I_q = jnp.sum(self.w_nodes * f_nodes)
        if self.num_mc_times == 0:
            return I_q

        tau = stratified_times(k_times, self.t0, self.t1, self.num_mc_times)
        p_tau = piecewise_interpolate(f_nodes, self.ref_nodes, self.t0, self.t1, self.num_intervals, tau)
        return I_q + (self.t1 - self.t0) * jnp.mean(f(tau, k_mc) - p_tau)
