"""Quadrature for the time integral of the drift residual."""
from .gauss_legendre import gauss_legendre_vecs, trapezoidal_vecs
from .rules import (GaussLegendreQuad, MonteCarloQuad, QuadRule1D, StratifiedSampling, UnbiasedGaussLegendreQuad,
                    stratified_sample1d)
