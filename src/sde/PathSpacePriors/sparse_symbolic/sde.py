from src.libraries.polynomials import MultivariatePolynomialLibrary


class SparsePolynomialDrift:
    """
    Drift f_theta(t, x) = Theta @ psi(x), with psi(x) all monomials of the D state variables up to total degree `degree`.
    Theta has shape (D, num_features); sparsity comes from the prior placed on it (e.g. the horseshoe).
    """

    def __init__(self, D, degree=3, include_bias=True):
        self.D = D
        self.library = MultivariatePolynomialLibrary(D, degree, include_bias)
        self.num_features = self.library.num_features
        self.num_params = D * self.num_features

    def drift(self, t, x, theta):
        """
        x: (..., D). theta: (D, num_features), or flat (D * num_features,) as sampled from the prior.
        Returns f_theta(t, x) of shape (..., D).
        """
        theta = theta.reshape(self.D, self.num_features)
        return self.library(x) @ theta.T

    def drift_fn(self, theta):
        """Drift with fixed theta and signature (t, x), as GaussianPathSDE and the ELBO expect."""
        return lambda t, x: self.drift(t, x, theta)
