"""
Equation-discovery runs from YAML configs. Each config trains in its own process with one CPU thread (the fastest
setting for these small tensors), and all configs run at the same time. Run from the repository root:

    python -m experiments.discovery.run experiments/discovery/configs/lorenz_port.yaml [more configs ...]
    python -m experiments.discovery.run --evaluate runs/<run_dir> [...]    # outputs again from model.pt, no training

A run writes runs/<name>_<YYYYmmdd-HHMMSS>/: config.yaml, log.txt, checkpoints/model_<step>.pt (training state),
model.pt (final model, pruned), history.json, equations.txt, metrics.json, variational_posterior.png,
posterior.png and learned_dynamics.png. A config can name a base config in the same folder (base: <file>); its own
keys override the base.
model.implementation "svise" uses the original SVISE (reference/svise, needs gpytorch) for parity checks.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from src.data import load_observations
from src.discovery import learner as port_learner
from src.discovery.evaluation import coefficient_errors, drift_coefficients, equations, prune
from src.discovery.learner import sample_paths
from src.discovery.train import Checkpoint, History, Log, fit
from src.plotting import plot_learned_dynamics, plot_posterior, plot_variational_posterior


# ------------------------------------------------------------------ known systems, for truth and coefficient errors
def _lorenz63(x):
    return np.array([10 * (x[1] - x[0]), x[0] * (28 - x[2]) - x[1], x[0] * x[1] - 8 / 3 * x[2]])


def _duffing(x):
    return np.array([x[1], x[0] - x[0] ** 3 - 0.35 * x[1]])


# true coefficients {(state, feature): value} in SVISE's feature order (experiment_util.py)
SYSTEMS = {
    "lorenz63": (_lorenz63, {(0, 1): -10.0, (0, 2): 10.0, (1, 1): 28.0, (1, 6): -1.0, (1, 2): -1.0, (2, 5): 1.0,
                             (2, 3): -8 / 3}),
    "duffing": (_duffing, {(0, 2): 1.0, (1, 1): 1.0, (1, 6): -1.0, (1, 2): -0.35}),
}


def rk4(rhs, x0, t_end, h=1e-3, n_out=1001):
    """Dense true path from x0 on [0, t_end], with n_out points, as in the data generators."""
    x, xs = np.array(x0, dtype=float), [np.array(x0, dtype=float)]
    for _ in range(round(t_end / h)):
        k1 = rhs(x); k2 = rhs(x + h / 2 * k1); k3 = rhs(x + h / 2 * k2); k4 = rhs(x + h * k3)  # noqa: E702
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    step = round(t_end / h) // (n_out - 1)
    return torch.linspace(0, t_end, round(t_end / h) + 1)[::step], torch.tensor(np.array(xs))[::step]


# ------------------------------------------------------------------ config, data, model
def load_config(path: str) -> dict:
    config = yaml.safe_load(open(path))
    base = config.pop("base", None)
    if base is None:
        return config
    merged = load_config(os.path.join(os.path.dirname(path), base))
    for key, value in config.items():
        merged[key] = {**merged[key], **value} if isinstance(value, dict) and key in merged else value
    return merged


def load_data(cfg: dict) -> dict:
    states = cfg["states"]
    return load_observations(cfg["path"], cfg["trajectory"], states, time_column=cfg.get("time_column", "t"),
                             truth_columns=[s + cfg.get("truth_suffix", "_true") for s in states],
                             obs_std_columns=[s + cfg.get("obs_std_suffix", "_obs_std") for s in states])


def learner_module(implementation: str):
    if implementation == "port":
        return port_learner
    if implementation == "svise":
        sys.path.insert(0, os.path.join("reference", "svise"))
        from svise import sde_learning
        return sde_learning
    raise ValueError(f"Unknown implementation '{implementation}', choose 'port' or 'svise'")


def build_model(cfg: dict, data: dict):
    """SparsePolynomialSDE with a known identity observation matrix; cfg holds its other arguments."""
    cfg = dict(cfg)
    module = learner_module(cfg.pop("implementation", "port"))
    t, y = data["t"], data["y"]
    d = y.shape[1]
    return module.SparsePolynomialSDE(d, (float(t.min()), float(t.max())), G=torch.eye(d), num_meas=d,
                                      measurement_noise=data["obs_std"].pow(2), train_t=t, train_x=y, **cfg)


def _setup():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)


# ------------------------------------------------------------------ training and evaluation of one run
def train(run_dir: str) -> None:
    _setup()
    config = load_config(os.path.join(run_dir, "config.yaml"))
    out = config["output"]
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    data = load_data(config["data"])
    sde = build_model(config["model"], data)
    history = History(out["history_every"])
    start = time.time()
    fit(sde, data["t"], data["y"], **config["train"],
        callbacks=[Checkpoint(run_dir, out["checkpoint_every"]), history, Log(out["log_every"])])
    seconds = time.time() - start
    prune(sde)
    torch.save(sde, os.path.join(run_dir, "model.pt"))
    json.dump({"step": history.steps, "loss": history.losses}, open(os.path.join(run_dir, "history.json"), "w"))
    evaluate(run_dir, {"seconds": seconds, "final_loss": history.losses[-1]})


def evaluate(run_dir: str, extra: dict = None) -> dict:
    """equations.txt, metrics.json and both figures from <run_dir>/model.pt; no training."""
    _setup()
    config = load_config(os.path.join(run_dir, "config.yaml"))
    states, ev = config["data"]["states"], config.get("evaluate", {})
    data = load_data(config["data"])
    sde = torch.load(os.path.join(run_dir, "model.pt"), weights_only=False)
    metrics_path = os.path.join(run_dir, "metrics.json")
    metrics = {**(json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}), **(extra or {})}

    W = drift_coefficients(sde)
    lines = equations(sde, states)
    with open(os.path.join(run_dir, "equations.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    metrics.update({"equations": lines, "W": W.tolist()})
    if "system" in ev:
        rhs, terms = SYSTEMS[ev["system"]]
        W_true = torch.zeros_like(W)
        for (i, j), value in terms.items():
            W_true[i, j] = value
        metrics.update(coefficient_errors(W, W_true))
        t_dense, x_dense = rk4(rhs, ev["x0"], ev["t_end"])
    else:
        t_dense, x_dense = data["t_true"], data["x_true"]
    truth, observations, name = (t_dense, x_dense), (data["t"], data["y"]), os.path.basename(run_dir)

    with torch.no_grad():
        mean = sde.marginal_sde.mean(t_dense)
        sd = sde.marginal_sde.K(t_dense).diagonal(dim1=-2, dim2=-1).sqrt()
    metrics["state_error"] = ((mean - x_dense).norm() / x_dense.norm()).item()
    metrics["variational_posterior"] = plot_variational_posterior(
        os.path.join(run_dir, "variational_posterior.png"), t_dense, mean, sd, truth, observations, states,
        f"{name}\napproximate variational posterior q(z(t))")
    torch.manual_seed(23)
    paths = sample_paths(sde, t_dense, ev.get("n_paths", 256))
    metrics["posterior"] = plot_posterior(
        os.path.join(run_dir, "posterior.png"), t_dense, paths, truth, observations, states,
        f"{name}\nposterior: SDE paths from z(t_0) ~ q(z(t_0)), drift and diffusion weights drawn from q")
    metrics["posterior"]["diverged_paths"] = int((~torch.isfinite(paths[-1]).all(-1)).sum())
    # the learned ODE dz = W phi(z) dt from the true start, on the grid of the true path
    features = sde.sde_prior.features
    with torch.no_grad():
        t_learned, x_learned = rk4(lambda x: (features(torch.as_tensor(x)) @ W.t()).numpy(), x_dense[0].numpy(),
                                   float(t_dense[-1] - t_dense[0]))
    metrics["learned_dynamics"] = plot_learned_dynamics(
        os.path.join(run_dir, "learned_dynamics.png"), t_learned + t_dense[0], x_learned, truth, observations, states,
        f"{name}\nlearned ODE (posterior-mean coefficients, no noise) and true ODE from the same start")
    json.dump(metrics, open(metrics_path, "w"), indent=1)
    print(name, {k: metrics[k] for k in ("rer", "mismatched", "state_error") if k in metrics}, flush=True)
    print("\n".join(lines), flush=True)
    return metrics


# ------------------------------------------------------------------ parallel launch
def launch(config_paths, stamp: str, root: str = "runs") -> int:
    """One process per config, all at the same time; each writes its output to <run_dir>/log.txt."""
    runs = []
    for path in config_paths:
        config = load_config(path)
        run_dir = os.path.join(root, f"{config['name']}_{stamp}")
        os.makedirs(run_dir)
        yaml.safe_dump(config, open(os.path.join(run_dir, "config.yaml"), "w"), sort_keys=False)
        log = open(os.path.join(run_dir, "log.txt"), "w")
        process = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "experiments.discovery.run",
                                    "--train", run_dir], stdout=log, stderr=subprocess.STDOUT)
        runs.append((run_dir, process))
        print(f"started {run_dir} (pid {process.pid})", flush=True)
    failed = 0
    for run_dir, process in runs:
        code = process.wait()
        failed += code != 0
        print(f"finished {run_dir}: exit code {code}", flush=True)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", help="YAML configs, one parallel run each")
    parser.add_argument("--evaluate", nargs="+", metavar="RUN_DIR", help="outputs again from model.pt")
    parser.add_argument("--runs", default="runs", help="folder for the run folders (default: runs)")
    parser.add_argument("--train", metavar="RUN_DIR", help=argparse.SUPPRESS)  # one run, started by launch
    args = parser.parse_args()
    if args.train:
        return train(args.train)
    if args.evaluate:
        for run_dir in args.evaluate:
            evaluate(run_dir)
        return 0
    names = [load_config(path)["name"] for path in args.configs]
    if not names or len(set(names)) < len(names):
        parser.error("give at least one config, and each config needs its own name")
    return launch(args.configs, time.strftime("%Y%m%d-%H%M%S"), args.runs)


if __name__ == "__main__":
    raise SystemExit(main())
