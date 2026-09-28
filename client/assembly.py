"""
Assembly: experiment config (a dict parsed from YAML by client.py) -> data, Gaussian path, prior SDE, loss and trainer.
Components are looked up by their `type` in the registries below; add new parameterizations there.
"""
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from src.loader.loader import load_trajectories
from src.sde.GaussianPaths.matern_path import MaternGaussianPath
from src.sde.PathSpacePriors.sparse_symbolic_sde import SparseSymbolicSDE
from src.training.callbacks import Checkpoint, EquationPrinter, History, PrintLogger
from src.training.losses.elbo import make_elbo_loss
from src.training.losses.unbiased_quadrature import UnbiasedGaussLegendreQuad
from src.training.trainer import Trainer, linear_warmup

GP_PATHS = {"matern": MaternGaussianPath}
SDES = {"sparse_symbolic": SparseSymbolicSDE}


@dataclass
class Experiment:
    config: dict
    data: dict
    gp: object
    sde: object
    loss_fn: object
    trainer: Trainer
    history: History


def build_data(cfg, key):
    """
    One trajectory of the CSV, subsampled every cfg["every"] steps and observed through H (the identity rows of
    the states in cfg["observed"]), with measurement noise R = obs_std^2 I.
    """
    trajs = load_trajectories(cfg["path"])
    x_true = trajs[int(cfg["trajectory"])]
    num_steps, D = x_true.shape
    t_true = jnp.arange(num_steps) * float(cfg["dt"])

    observed = list(range(D)) if cfg["observed"] is None else [int(i) for i in cfg["observed"]]
    H = jnp.eye(D)[jnp.array(observed)]
    obs_std = float(cfg["obs_std"])
    R = obs_std ** 2 * jnp.eye(len(observed))

    idx = jnp.arange(0, num_steps, int(cfg["every"]))
    t_obs = t_true[idx]
    y_obs = x_true[idx] @ H.T
    if cfg["add_noise"]:
        y_obs = y_obs + obs_std * jax.random.normal(key, y_obs.shape)

    return {"t_obs": t_obs, "y_obs": y_obs, "H": H, "R": R, "observed": observed,
            "T": float(t_true[-1]), "D": D, "t_true": t_true, "x_true": x_true}


def build_gp(cfg, data):
    cfg = dict(cfg)
    gp_type = cfg.pop("type")
    init_mean_from_data = cfg.pop("init_mean_from_data", True)
    gp = GP_PATHS[gp_type](D=data["D"], T=data["T"], **cfg)

    if init_mean_from_data and hasattr(gp, "t_support"):
        # Start the mean at the observations (linearly interpolated), unobserved states at zero
        Hm = jnp.zeros((gp.t_support.shape[0], data["D"]))
        for j, i in enumerate(data["observed"]):
            Hm = Hm.at[:, i].set(jnp.interp(gp.t_support, data["t_obs"], data["y_obs"][:, j]))
        gp.Hm_init = Hm
    return gp


def build_sde(cfg, data):
    cfg = dict(cfg)
    sde_type = cfg.pop("type")
    horseshoe = cfg.pop("horseshoe", None)
    return SDES[sde_type](D=data["D"], horseshoe_kwargs=horseshoe, **cfg)


def build_loss(cfg, gp, sde, data, batch_size):
    quad = UnbiasedGaussLegendreQuad(0.0, data["T"], int(cfg["quad_nodes"]),
                                     int(cfg["quad_intervals"]),
                                     num_mc_times=int(cfg.get("num_mc_times", 0)))
    num_data = data["t_obs"].shape[0] if batch_size is not None else None
    return make_elbo_loss(gp, sde, data["H"], data["R"], quad,
                          num_samples=int(cfg["num_samples"]), num_data=num_data)


def build_optimizers(cfg):
    """One Adam per parameter group, optionally with staircase exponential learning-rate decay."""
    optimizers = {}
    for group in ("gp", "sde"):
        lr = float(cfg["lr"][group])
        if cfg["lr_decay"]:
            lr = optax.exponential_decay(lr, transition_steps=int(cfg["lr_decay"]["every"]),
                                         decay_rate=float(cfg["lr_decay"]["rate"]), staircase=True)
        optimizers[group] = optax.adam(lr)
    return optimizers


def build_callbacks(cfg, sde, run_dir):
    history = History(every=int(cfg["history_every"]))
    callbacks = [PrintLogger(every=int(cfg["log_every"])), history]
    if cfg["checkpoint_every"]:
        callbacks.append(Checkpoint(os.path.join(run_dir, "checkpoints"), every=int(cfg["checkpoint_every"])))
    if cfg["equations_every"] and hasattr(sde, "coefficients"):
        callbacks.append(EquationPrinter(lambda params: sde.coefficients(params["sde"]),
                                         sde.drift.library.feature_names(),
                                         every=int(cfg["equations_every"]),
                                         threshold=float(cfg["equation_threshold"])))
    return callbacks, history


def assemble(config, run_dir):
    """Builds every component of the experiment described by a validated config (see client.load_config)."""
    key = jax.random.PRNGKey(int(config["seed"]))
    data = build_data(config["data"], jax.random.fold_in(key, 1))
    gp = build_gp(config["gp"], data)
    sde = build_sde(config["sde"], data)
    loss_fn = build_loss(config["loss"], gp, sde, data, config["train"]["batch_size"])
    callbacks, history = build_callbacks(config["callbacks"], sde, run_dir)
    trainer = Trainer(gp, sde, loss_fn, build_optimizers(config["train"]),
                      beta_schedule=linear_warmup(int(config["train"]["beta_warmup"])), callbacks=callbacks)
    return Experiment(config, data, gp, sde, loss_fn, trainer, history)
