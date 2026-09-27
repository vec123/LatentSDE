"""
Evaluation after training: the state estimate against the true trajectory, the discovered equations and a plot.
Writes metrics.json, equations.txt and state_estimate.png to the run directory.
"""
import json
import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def evaluate(experiment, params, run_dir, every=10):
    """every: evaluate on every `every`-th point of the true trajectory."""
    data = experiment.data
    prior = experiment.sde.sample(params["sde"], jax.random.PRNGKey(0))
    path = experiment.gp.build(params["gp"], prior.diffusion)

    t = data["t_true"][::every]
    x = data["x_true"][::every]
    m = jax.vmap(path.mean_fn)(t)
    sd = jnp.sqrt(jax.vmap(lambda ti: jnp.diag(path.Sigma_fn(ti)))(t))

    metrics = {
        "rmse": jnp.sqrt(jnp.mean((m - x) ** 2, axis=0)).tolist(),
        "coverage_2sd": jnp.mean(jnp.abs(m - x) <= 2 * sd, axis=0).tolist(),
        "observed_states": data["observed"],
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if hasattr(experiment.sde, "coefficients"):
        threshold = float(experiment.config["callbacks"]["equation_threshold"])
        write_equations(experiment.sde, params, os.path.join(run_dir, "equations.txt"), threshold)

    plot_state_estimate(np.asarray(t), np.asarray(x), np.asarray(m), np.asarray(sd), data,
                        os.path.join(run_dir, "state_estimate.png"))
    print(f"evaluation: {metrics}")
    return metrics


def write_equations(sde, params, path, threshold):
    """Posterior-mean drift, keeping terms with |coefficient| > threshold."""
    theta = np.asarray(sde.coefficients(params["sde"]))
    names = sde.drift.library.feature_names()
    lines = []
    for i, row in enumerate(theta):
        rhs = " + ".join(f"{c:.4g} {name}" for c, name in zip(row, names) if abs(c) > threshold)
        lines.append(f"dx{i + 1}/dt = {rhs or '0'}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_state_estimate(t, x, m, sd, data, path):
    D = x.shape[1]
    fig, axes = plt.subplots(D, 1, figsize=(10, 3 * D), sharex=True, squeeze=False)
    for i in range(D):
        ax = axes[i, 0]
        ax.plot(t, x[:, i], "k", lw=1, label="true")
        ax.plot(t, m[:, i], "C0", label="posterior mean")
        ax.fill_between(t, m[:, i] - 2 * sd[:, i], m[:, i] + 2 * sd[:, i], color="C0", alpha=0.3, label="±2 sd")
        if i in data["observed"]:
            j = data["observed"].index(i)
            ax.plot(np.asarray(data["t_obs"]), np.asarray(data["y_obs"][:, j]), "C3.", ms=3, label="observations")
        ax.set_ylabel(f"x{i + 1}")
    axes[0, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
