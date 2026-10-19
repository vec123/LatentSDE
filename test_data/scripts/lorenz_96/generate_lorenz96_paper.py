"""Generate the Lorenz-96 equation-discovery datasets from Course & Nair (2023), example 5.

The system is x_k' = x_{k-1} (x_{k+1} - x_{k-2}) - x_k + F with F = 10 and k = 1, ..., 1024 (periodic), with
x(0) ~ N(0, I) on t in [0, 10]. The states are not measured directly: each dataset has 512 measurements
y = G x + noise with a random low-rank matrix G = (1/r) sum_i u_i u_i^T, u_i ~ N(0, I), and a noise std of
2% of the half range of each clean measurement channel. The paper uses ranks 256, 512 and 1024. The other
settings follow reference/svise/experiments/5_extended_goveq_discovery/generate_data.py (5 datasets per rank,
all from the same trajectory; only G and the noise change).

One file for each rank, lorenz96_discovery_rank<rank>.npz, with
    t               (512,)            observation times
    y               (5, 512, 1024)    noisy measurements, one block for each dataset
    U               (5, rank, 1024)   measurement vectors u_i, G = U.T @ U / rank
    obs_std         (5, 1024)         noise std of each measurement channel
    x_true          (512, 1024)       clean states
    x0              (1024,)           initial state
    F, rank, noise_fraction           scalars
y, U and x_true are float32 to keep the files small. y comes from the stored float32 U, so G = U.T @ U / rank
in float64 gives the clean part of y.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


DIM = 1024
FORCING = 10.0
T0, T1 = 0.0, 10.0
NUM_OBSERVATIONS = 512
SUBSTEPS = 20  # RK4 steps between two observations, a step of about 1e-3
RANKS = (256, 512, 1024)
NOISE_FRACTION = 0.02
NUM_DATASETS = 5
SEED = 20260930


def lorenz96_rhs(x):
    return np.roll(x, 1) * (np.roll(x, -1) - np.roll(x, 2)) - x + FORCING


def rk4_trajectory(x0, times):
    h = (times[1] - times[0]) / SUBSTEPS
    states = np.empty((len(times), DIM), dtype=float)
    states[0] = x = x0
    for i in range(1, len(times)):
        for _ in range(SUBSTEPS):
            k1 = lorenz96_rhs(x)
            k2 = lorenz96_rhs(x + 0.5 * h * k1)
            k3 = lorenz96_rhs(x + 0.5 * h * k2)
            k4 = lorenz96_rhs(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        states[i] = x
    return states


def clean_measurements(x_true, u):
    """x_true @ G for G = u.T @ u / rank, without the dense 1024 x 1024 matrix."""
    return (x_true @ u.T) @ u / len(u)


def make_rank(x_true, rank, rng):
    U = rng.standard_normal((NUM_DATASETS, rank, DIM)).astype(np.float32)
    y = np.empty((NUM_DATASETS, len(x_true), DIM), dtype=np.float32)
    obs_std = np.empty((NUM_DATASETS, DIM), dtype=float)
    for j in range(NUM_DATASETS):
        clean = clean_measurements(x_true, U[j].astype(float))
        obs_std[j] = NOISE_FRACTION * 0.5 * (clean.max(axis=0) - clean.min(axis=0))
        y[j] = clean + rng.normal(0.0, obs_std[j], size=clean.shape)
    return U, y, obs_std


def main():
    output_dir = Path(__file__).resolve().parents[2] / "data" / "lorenz_96"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x0 = rng.standard_normal(DIM)
    times = np.linspace(T0, T1, NUM_OBSERVATIONS)
    x_true = rk4_trajectory(x0, times)

    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(len(RANKS), 2, width_ratios=[1.2, 1])
    ax = fig.add_subplot(grid[:, 0])
    image = ax.imshow(x_true, origin="lower", aspect="auto", extent=(0, DIM, T0, T1),
                      cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0), interpolation="nearest")
    fig.colorbar(image, ax=ax, label="x_k")
    ax.set_title("Clean states")
    ax.set_xlabel("state index k")
    ax.set_ylabel("t")

    for row, rank in enumerate(RANKS):
        U, y, obs_std = make_rank(x_true, rank, rng)
        npz_path = output_dir / f"lorenz96_discovery_rank{rank:04}.npz"
        np.savez_compressed(npz_path, t=times, y=y, U=U, obs_std=obs_std, x_true=x_true.astype(np.float32),
                            x0=x0, F=FORCING, rank=rank, noise_fraction=NOISE_FRACTION)
        print(f"Wrote {NUM_DATASETS} datasets of rank {rank} to {npz_path}")

        clean = clean_measurements(x_true, U[0].astype(float))[:, 0]
        axes = fig.add_subplot(grid[row, 1])
        axes.plot(times, clean, color="black", lw=1.2, label="G x (clean)")
        axes.scatter(times, y[0, :, 0], s=5, alpha=0.5, label="y (observed)")
        axes.set_title(f"Rank {rank}, dataset 0, channel 0; noise std = {obs_std[0, 0]:.3f}", fontsize=9)
        axes.set_ylabel("measurement")
        axes.legend(fontsize=7, loc="upper right")
    axes.set_xlabel("t")

    fig.suptitle("Lorenz-96 equation-discovery data, 1,024 states (Course & Nair, 2023)")
    fig.tight_layout()
    png_path = output_dir / "lorenz96_discovery.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot to {png_path}")


if __name__ == "__main__":
    main()
