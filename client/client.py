"""
Client: runs an experiment described by a YAML config.

    python -m client.client client/configs/vdp_x1_only.yaml

Loads the YAML, fills in defaults and validates it, assembles the experiment (assembly.py), trains it and
writes everything to runs/<name>_<timestamp>/: the resolved config, checkpoints, the final parameters,
the training history and the evaluation (evaluate.py).
"""
import argparse
import copy
import json
import os
import pickle
import time

import jax
import yaml

from client.assembly import GP_PATHS, SDES, assemble
from client.evaluate import evaluate

DEFAULTS = {
    "seed": 0,
    "data": {"trajectory": 0, "every": 1, "observed": None, "obs_std": 0.05, "add_noise": False},
    "gp": {},
    "sde": {},
    "loss": {"num_samples": 16, "quad_nodes": 3, "quad_intervals": 100},
    "train": {"batch_size": None, "beta_warmup": 0, "lr": {"gp": 1e-3, "sde": 1e-2}, "lr_decay": None},
    "callbacks": {"log_every": 500, "history_every": 10, "checkpoint_every": None,
                  "equations_every": None, "equation_threshold": 0.01},
}
REQUIRED = [("name",), ("data", "path"), ("data", "dt"), ("gp", "type"), ("sde", "type"), ("train", "num_steps")]


def merge(defaults, config):
    """Recursively fills config with defaults; values in config win."""
    merged = copy.deepcopy(defaults)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path):
    """Loads the YAML, fills in DEFAULTS and checks it, so that mistakes fail here and not inside JAX."""
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    unknown = set(config) - set(DEFAULTS) - {"name"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    config = merge(DEFAULTS, config)

    for keys in REQUIRED:
        section = config
        for key in keys:
            if not isinstance(section, dict) or section.get(key) is None:
                raise ValueError(f"Missing required config entry: {'.'.join(keys)}")
            section = section[key]

    if config["gp"]["type"] not in GP_PATHS:
        raise ValueError(f"Unknown gp.type '{config['gp']['type']}', choose from {sorted(GP_PATHS)}")
    if config["sde"]["type"] not in SDES:
        raise ValueError(f"Unknown sde.type '{config['sde']['type']}', choose from {sorted(SDES)}")
    return config


def make_run_dir(runs_dir, name):
    run_dir = os.path.join(runs_dir, f"{name}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir)
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run an experiment from a YAML config.")
    parser.add_argument("config", help="path to the experiment YAML")
    parser.add_argument("--runs-dir", default="runs", help="directory in which the run directory is created")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_dir = make_run_dir(args.runs_dir, config["name"])
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"run directory: {run_dir}")

    experiment = assemble(config, run_dir)
    data = experiment.data
    params, _ = experiment.trainer.fit(jax.random.PRNGKey(int(config["seed"])), (data["t_obs"], data["y_obs"]),
                                       int(config["train"]["num_steps"]), batch_size=config["train"]["batch_size"])

    with open(os.path.join(run_dir, "params.pkl"), "wb") as f:
        pickle.dump(jax.device_get(params), f)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(experiment.history.records, f)
    evaluate(experiment, params, run_dir)


if __name__ == "__main__":
    main()
