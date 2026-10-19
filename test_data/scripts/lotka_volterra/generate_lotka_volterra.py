"""
Stochastic Lotka-Volterra data with a diffusion that depends on the state, to test the effect of that dependence.
Setup of Course & Nair (2023), Bartosh et al. (2025, appendix) and arXiv 2610.04390 (appendix):

    dz = (alpha z1 - beta z1 z2,  delta z1 z2 - gamma z2) dt + G(z) dW,   G(z) = sigma diag(z) L_rho,
    L_rho = [[1, 0], [rho, sqrt(1 - rho^2)]].

rho = 0 is the benchmark of the papers: G is diagonal. rho != 0 correlates the noise of the two species: G is full,
and row i depends on z only through z_i (the form of src/latent/diffusion.py's NetCholeskyDiffusion).

The same paths in log coordinates l = log z have the constant diffusion sigma L_rho (Ito's lemma), so they are the
control with a state-independent diffusion. The simulation is Euler-Maruyama in log coordinates, which keeps the
populations positive.

The CSV has the format of src/data/trajectories.py (experiments/discovery/run.py), one row per observation:
    t, trajectory_id, time_step
    x1, x2         populations with Gaussian observation noise    (G depends on the state)
    l1, l2         log-populations with Gaussian observation noise (G is constant)
    <state>_true   without observation noise; <state>_obs_std the noise std
Each trajectory has its own noise path and initial state. In a discovery config: states: [x1, x2] or [l1, l2].

    python -m test_data.scripts.lotka_volterra.generate_lotka_volterra
    python -m test_data.scripts.lotka_volterra.generate_lotka_volterra --rho 0.8 --name lotka_volterra_correlated

With --split, the file stem is <name>_<split>: the latent splits (experiments/latent/configs/lotka_volterra_*.yaml)
    python -m test_data.scripts.lotka_volterra.generate_lotka_volterra --name lotka_volterra_latent \\
        --split train --n-traj 1024 --seed 1   (val: --n-traj 64 --seed 2, test: --n-traj 128 --seed 3)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
PREY, PREDATOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="lotka_volterra", help="output folder and file stem in test_data/data")
    p.add_argument("--split", default=None, help="appended to the file stem, e.g. train, val or test")
    p.add_argument("--rho", type=float, default=0.0, help="correlation of the noise of the two species")
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--params", type=float, nargs=4, default=(2 / 3, 4 / 3, 1.0, 1.0),
                   metavar=("ALPHA", "BETA", "DELTA", "GAMMA"))
    p.add_argument("--z0", type=float, nargs=2, default=(0.9, 0.2))
    p.add_argument("--z0-std", type=float, default=0.05)
    p.add_argument("--T", type=float, default=40.0)
    p.add_argument("--dt", type=float, default=1e-3, help="Euler-Maruyama step")
    p.add_argument("--dt-obs", type=float, default=0.25, help="observation spacing, a multiple of dt")
    p.add_argument("--obs-std", type=float, default=0.1)
    p.add_argument("--n-traj", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261015)
    return p.parse_args()


def drift(z, alpha, beta, delta, gamma):
    return np.stack([alpha * z[..., 0] - beta * z[..., 0] * z[..., 1],
                     delta * z[..., 0] * z[..., 1] - gamma * z[..., 1]], -1)


def simulate(rng, args):
    """t (n_obs,) and log z (n_traj, n_obs, 2), Euler-Maruyama in log coordinates."""
    L = np.array([[1.0, 0.0], [args.rho, np.sqrt(1 - args.rho ** 2)]])
    every = round(args.dt_obs / args.dt)
    n_obs = round(args.T / args.dt_obs) + 1
    # z0_2 is only 4 std above zero, and the log needs a positive start
    z0 = np.maximum(np.array(args.z0) + args.z0_std * rng.standard_normal((args.n_traj, 2)), 1e-3)
    l = np.log(z0)
    out = np.empty((args.n_traj, n_obs, 2))
    out[:, 0] = l
    for k in range(1, (n_obs - 1) * every + 1):
        # Ito: dl = f(z) / z - sigma^2 / 2 (each row of L_rho has norm 1) + sigma L_rho dW
        dl = drift(np.exp(l), *args.params) / np.exp(l) - 0.5 * args.sigma ** 2
        l = l + dl * args.dt + args.sigma * np.sqrt(args.dt) * rng.standard_normal((args.n_traj, 2)) @ L.T
        if k % every == 0:
            out[:, k // every] = l
    return np.arange(n_obs) * args.dt_obs, out


def deterministic_cycle(args, t_end=10.0):
    """sigma = 0 from the mean initial state, RK4, a bit more than one period (8.0 for the defaults)."""
    f = lambda z: drift(z, *args.params)
    z, path = np.array(args.z0, dtype=float), [np.array(args.z0, dtype=float)]
    for _ in range(round(t_end / args.dt)):
        k1 = f(z); k2 = f(z + args.dt / 2 * k1); k3 = f(z + args.dt / 2 * k2); k4 = f(z + args.dt * k3)  # noqa: E702
        z = z + args.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        path.append(z)
    return np.array(path)


def to_frame(t, log_z, rng, args):
    columns = {"t": np.tile(t, args.n_traj),
               "trajectory_id": np.repeat(np.arange(args.n_traj), len(t)),
               "time_step": np.tile(np.arange(len(t)), args.n_traj)}
    for prefix, clean in (("x", np.exp(log_z)), ("l", log_z)):
        noisy = clean + args.obs_std * rng.standard_normal(clean.shape)
        for i in range(2):
            name = f"{prefix}{i + 1}"
            columns[name] = noisy[..., i].reshape(-1)
            columns[name + "_true"] = clean[..., i].reshape(-1)
            columns[name + "_obs_std"] = args.obs_std
    return pd.DataFrame(columns)


def plot(path, frame, cycle, args, n_paths=20):
    shown = frame[frame.trajectory_id < n_paths]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for i, (label, color) in enumerate((("prey x1", PREY), ("predator x2", PREDATOR))):
        for k, (_, traj) in enumerate(shown.groupby("trajectory_id")):
            ax.plot(traj.t, traj[f"x{i + 1}_true"], color=color, lw=0.8, alpha=0.5,
                    label=f"{label}, true" if k == 0 else None)
    ax.set(xlabel="t", ylabel="population", title=f"{min(n_paths, args.n_traj)} trajectories")

    for ax, prefix, transform, title in (
            (axes[1], "x", lambda z: z, r"Populations: $G = \sigma\,\mathrm{diag}(z)\,L_\rho$"),
            (axes[2], "l", np.log, r"Log-populations: $G = \sigma L_\rho$ (control)")):
        for k, (_, traj) in enumerate(shown.groupby("trajectory_id")):
            ax.plot(traj[f"{prefix}1_true"], traj[f"{prefix}2_true"], color=PREY, lw=0.6, alpha=0.35,
                    label="stochastic paths, true" if k == 0 else None)
        ax.plot(*transform(cycle).T, color=INK, lw=1.5, label=r"deterministic cycle ($\sigma = 0$)")
        ax.set(xlabel=f"{prefix}1 (prey)", ylabel=f"{prefix}2 (predator)", title=title)

    for ax in axes:
        ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(rf"Stochastic Lotka-Volterra, $\sigma = {args.sigma:g}$, $\rho = {args.rho:g}$, "
                 f"observation noise std = {args.obs_std:g}, {min(n_paths, args.n_traj)} of {args.n_traj} "
                 "trajectories", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = DATA / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t, log_z = simulate(rng, args)
    frame = to_frame(t, log_z, rng, args)
    stem = args.name if args.split is None else f"{args.name}_{args.split}"
    csv_path = out_dir / f"{stem}.csv"
    frame.to_csv(csv_path, index=False, float_format="%.10g")
    plot(out_dir / f"{stem}.png", frame, deterministic_cycle(args), args)
    print(f"Wrote {len(frame):,} rows ({args.n_traj} trajectories x {len(t)} times) to {csv_path}")


if __name__ == "__main__":
    main()
