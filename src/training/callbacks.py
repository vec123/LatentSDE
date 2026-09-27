"""
Callbacks for the Trainer: callables callback(step, params, loss, terms), run outside jit after every step.
"""
import os
import pickle

import jax
import numpy as np


class PrintLogger:
    """Prints the loss and the ELBO terms every `every` steps."""

    def __init__(self, every=100):
        self.every = every

    def __call__(self, step, params, loss, terms):
        if step % self.every == 0:
            parts = "  ".join(f"{name} {float(value):.4g}" for name, value in terms.items())
            print(f"step {step:6d}  loss {float(loss):.6g}  {parts}")


class History:
    """Records the loss and the ELBO terms every `every` steps in self.records, e.g. for plotting."""

    def __init__(self, every=1):
        self.every = every
        self.records = []

    def __call__(self, step, params, loss, terms):
        if step % self.every == 0:
            self.records.append({"step": step, "loss": float(loss),
                                 **{name: float(value) for name, value in terms.items()}})


class Checkpoint:
    """Pickles the parameters to directory/params_<step>.pkl every `every` steps."""

    def __init__(self, directory, every=1000):
        self.directory = directory
        self.every = every
        os.makedirs(directory, exist_ok=True)

    def __call__(self, step, params, loss, terms):
        if step % self.every == 0:
            with open(os.path.join(self.directory, f"params_{step:06d}.pkl"), "wb") as f:
                pickle.dump(jax.device_get(params), f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class EquationPrinter:
    """
    Prints the discovered drift f_i(x) = sum_k Theta_ik psi_k(x), keeping terms with |Theta_ik| > threshold.
    coefficients: function params -> Theta of shape (D, num_features), e.g. the posterior mean of the drift coefficients.
    feature_names: names of the library features, e.g. MultivariatePolynomialLibrary.feature_names().
    """

    def __init__(self, coefficients, feature_names, every=1000, threshold=1e-2, state_names=None):
        self.coefficients = coefficients
        self.feature_names = feature_names
        self.every = every
        self.threshold = threshold
        self.state_names = state_names

    def __call__(self, step, params, loss, terms):
        if step % self.every != 0:
            return
        theta = np.asarray(self.coefficients(params))
        state_names = self.state_names or [f"x{i + 1}" for i in range(theta.shape[0])]
        print(f"step {step}: discovered drift")
        for name, row in zip(state_names, theta):
            rhs = " + ".join(f"{c:.3g} {feature}" for c, feature in zip(row, self.feature_names)
                             if abs(c) > self.threshold)
            print(f"  d{name}/dt = {rhs or '0'}")
