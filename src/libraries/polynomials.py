import itertools
import numpy as np
import jax.numpy as jnp

class BasisLibrary:
    """
    A library to generate basis function matrices for regression models.
    """
    
    @staticmethod
    def polynomial(x, degree=3):
        """
        Generates a polynomial feature matrix.
        Input x: shape (n,)
        Output: shape (n, degree)
        """
        # Ensure x is a column vector
        x = x.reshape(-1, 1)
        
        # Create powers: [x^1, x^2, ..., x^degree]
        # We start from 1 to avoid the intercept (often handled separately)
        powers = jnp.arange(1, degree + 1)
        return jnp.power(x, powers)


class MultivariatePolynomialLibrary:
    """
    All monomials of D variables up to total degree `degree`, in graded order,
    e.g. D=2, degree=2: 1, x1, x2, x1^2, x1 x2, x2^2.
    Used as a drift library f(x) = Theta @ library(x) with Theta of shape (D, num_features).
    There are comb(D + degree, degree) features with the constant term, one fewer without.
    """

    def __init__(self, D, degree=3, include_bias=True):
        self.D = D
        self.degree = degree
        self.include_bias = include_bias

        # Each monomial as the tuple of variable indices it multiplies, e.g. x1^2 x2 -> (0, 0, 1)
        combos = [c for k in range(0 if include_bias else 1, degree + 1)
                  for c in itertools.combinations_with_replacement(range(D), k)]
        if not combos:
            raise ValueError("degree must be >= 1 when include_bias=False")

        # Pad every tuple to length `degree` with index D, which points at a constant 1 appended to x.
        # Products of gathered entries avoid x**0, whose gradient is NaN at x = 0.
        self.indices = jnp.array([c + (D,) * (degree - len(c)) for c in combos], dtype=jnp.int32)
        self.exponents = np.array([np.bincount(np.asarray(c, dtype=int), minlength=D) for c in combos])
        self.num_features = len(combos)

    def __call__(self, x):
        """x: (..., D) -> (..., num_features)."""
        x_ext = jnp.concatenate([x, jnp.ones(x.shape[:-1] + (1,), dtype=x.dtype)], axis=-1)
        return jnp.prod(x_ext[..., self.indices], axis=-1)

    def feature_names(self, variables=None):
        """Readable names, e.g. ['1', 'x1', 'x2', 'x1^2', 'x1 x2', 'x2^2']."""
        variables = variables or [f"x{i + 1}" for i in range(self.D)]
        names = []
        for e in self.exponents:
            terms = [v if p == 1 else f"{v}^{p}" for v, p in zip(variables, e) if p > 0]
            names.append(" ".join(terms) or "1")
        return names