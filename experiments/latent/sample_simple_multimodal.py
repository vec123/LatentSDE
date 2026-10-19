"""
Samples of the learned SDE (red) and of the true SDE (blue) of simple_multimodal_test, all from one start, with no
observations after it: the spread of each ensemble shows the law of its SDE. The learned SDE starts at the posterior
mean m(0) of one test trajectory with both modalities, the true SDE at its decoded value (x, y1, y2), so both
ensembles start at the same point.

    python -m experiments.latent.sample_simple_multimodal runs/<run_dir> [--split test] [--n 20] [--traj 0]
        [--checkpoint STEP]

True SDE: dx = KAPPA y1 dt + X_NOISE dW_0 (the x_noise of the data, 0 by default),
dy_j = -y_j / TAU dt + SIGMA sqrt(2 / TAU) dW_j on the truth grid of the split (exact
OU steps, x the integral of y1, as in test_data/scripts/simple_multimodal_test/generate.py). Learned SDE:
Euler-Maruyama on the same grid, decoder mean (no observation noise).

Writes <run_dir>/samples_<split>.png.
"""
import argparse
import os

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.multimodal import load_multimodal
from test_data.src.multimodal_reaction import integral
from .evaluate_simple_multimodal import SIM_COLOR, TRUE_COLOR, load_model, posterior_mean, simulate_paths
from .run import INK, load_config


def true_paths(t: np.ndarray, start: np.ndarray, kappa: float, tau: float, sigma: float, x_noise: float, n: int,
               rng: np.random.Generator) -> np.ndarray:
    """n paths (n, len(t), 3) of (x, y1, y2) of the true SDE from start (3,)."""
    a = np.exp(-np.diff(t) / tau)
    y = np.empty((n, len(t), 2))
    y[:, 0] = start[1:]
    for k in range(len(t) - 1):
        y[:, k + 1] = a[k] * y[:, k] + sigma * np.sqrt(1 - a[k] ** 2) * rng.standard_normal((n, 2))
    dw = np.sqrt(np.diff(t)) * rng.standard_normal((n, len(t) - 1))
    w = np.concatenate([np.zeros((n, 1)), np.cumsum(dw, 1)], 1)
    x = start[0] + kappa * integral(t, y[..., 0]) + x_noise * w
    return np.concatenate([x[..., None], y], -1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=20, help="samples of each SDE")
    p.add_argument("--traj", type=int, default=0, help="which trajectory with both modalities gives the start")
    p.add_argument("--checkpoint", type=int, help="step of a checkpoint to read instead of model.pt")
    args = p.parse_args()

    config = load_config(os.path.join(args.run_dir, "config.yaml"))
    model, run_name, suffix = load_model(args.run_dir, args.checkpoint)
    cfg = config["data"][args.split]
    (batch,), _ = load_multimodal(cfg["path"], cfg["modalities"], 1)
    with h5py.File(cfg["path"], "r") as f:
        t, group = f["truth/t"][:], f["group"][:]
        kappa, tau, sigma = (float(f.attrs[k]) for k in ("kappa", "tau", "sigma"))
        x_noise = float(f.attrs.get("x_noise", 0.0))  # data generated before --x-noise has none

    b = int(np.flatnonzero(group == 0)[args.traj])
    sub = {k: v[b:b + 1] for k, v in batch.items()}
    m0 = posterior_mean(model, sub, torch.zeros(1, 1, dtype=batch["t"].dtype))[:, 0]
    torch.manual_seed(0)
    zs = simulate_paths(model, t, m0.repeat(args.n, 1), 1)
    with torch.no_grad():
        start = model.decoder(m0, m0.new_zeros(1, 0)).mean[0].numpy()
        learned = model.decoder(zs, zs.new_zeros(*zs.shape[:2], 0)).mean.numpy()
    true = true_paths(t, start, kappa, tau, sigma, x_noise, args.n, np.random.default_rng(0))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for k, (ax, name) in enumerate(zip(axes, ("x", "y1", "y2"))):
        for i in range(args.n):
            ax.plot(t, true[i, :, k], color=TRUE_COLOR, lw=0.9, alpha=0.6,
                    label=f"true SDE ({args.n} samples)" if i == 0 else None)
            ax.plot(t, learned[i, :, k], color=SIM_COLOR, lw=0.9, alpha=0.6,
                    label=f"learned SDE ({args.n} samples)" if i == 0 else None)
        ax.plot(t[0], start[k], "o", color=INK, ms=6, mec="white", mew=1, label="start" if k == 0 else None, zorder=3)
        ax.set(ylabel=name)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    axes[-1].set(xlabel="t")
    fig.suptitle(f"{run_name}, {args.split} trajectory {b}: samples of the "
                 f"true and the learned SDE from one start (x, y1, y2) = ({start[0]:.2f}, {start[1]:.2f}, "
                 f"{start[2]:.2f})", color=INK, fontsize=11)
    out = os.path.join(args.run_dir, f"samples_{args.split}{suffix}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
