"""Generate the Lorenz-63 symbolic-discovery datasets described in Course & Nair (2023).

The paper uses 20 independent datasets in each of two regimes: 64 observations with
1%-of-half-range noise, and 2,048 observations with 25%-of-half-range noise.
Each CSV stores noisy observations in x1/x2/x3 and clean states in x1_true/etc.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INITIAL_STATE = np.array([-8.0, 7.0, 27.0])
T0, T1 = 0.0, 10.0
INTEGRATION_DT = 0.001
NUM_DATASETS = 20
SEED = 20260928


def lorenz_rhs(x):
    a, b, c = x
    return np.array([10.0 * (b - a), a * (28.0 - c) - b, a * b - (8.0 / 3.0) * c])


def rk4_trajectory():
    n_steps = round((T1 - T0) / INTEGRATION_DT)
    times = np.linspace(T0, T1, n_steps + 1)
    states = np.empty((n_steps + 1, 3), dtype=float)
    states[0] = INITIAL_STATE
    for k in range(n_steps):
        x = states[k]
        h = INTEGRATION_DT
        k1 = lorenz_rhs(x)
        k2 = lorenz_rhs(x + 0.5 * h * k1)
        k3 = lorenz_rhs(x + 0.5 * h * k2)
        k4 = lorenz_rhs(x + h * k3)
        states[k + 1] = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return times, states


def make_regime(times_dense, states_dense, n_observations, noise_fraction, name, rng, output_dir):
    times = np.linspace(T0, T1, n_observations)
    truth = np.column_stack([
        np.interp(times, times_dense, states_dense[:, d]) for d in range(3)
    ])
    half_range = 0.5 * (states_dense.max(axis=0) - states_dense.min(axis=0))
    obs_std = noise_fraction * half_range

    frames = []
    for dataset_id in range(NUM_DATASETS):
        noisy = truth + rng.normal(0.0, obs_std, size=truth.shape)
        frame = pd.DataFrame({
            "t": times,
            "x1": noisy[:, 0], "x2": noisy[:, 1], "x3": noisy[:, 2],
            "x1_true": truth[:, 0], "x2_true": truth[:, 1], "x3_true": truth[:, 2],
            "trajectory_id": dataset_id,
            "time_step": np.arange(n_observations),
            "x1_obs_std": obs_std[0], "x2_obs_std": obs_std[1], "x3_obs_std": obs_std[2],
        })
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / f"lorenz63_discovery_{name}.csv"
    result.to_csv(csv_path, index=False, float_format="%.10g")
    return csv_path, times, truth, obs_std, result


def main():
    output_dir = Path(__file__).resolve().parents[2] / "data" / "lorenz_63"
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_times, dense_states = rk4_trajectory()
    rng = np.random.default_rng(SEED)

    low_path, low_t, low_truth, low_std, low_data = make_regime(
        dense_times, dense_states, 64, 0.01, "low_noise_low_data", rng, output_dir)
    high_path, high_t, high_truth, high_std, high_data = make_regime(
        dense_times, dense_states, 2048, 0.25, "high_noise_large_data", rng, output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, (label, frame, n, std) in enumerate([
        ("Low noise / low data (64 points)", low_data, 64, low_std),
        ("High noise / large data (2,048 points)", high_data, 2048, high_std),
    ]):
        first = frame[frame.trajectory_id == 0]
        truth = first[["x1_true", "x2_true", "x3_true"]].to_numpy()
        observed = first[["x1", "x2", "x3"]].to_numpy()
        t = first["t"].to_numpy()
        for d in range(3):
            axes[row, 0].plot(t, truth[:, d], lw=1.5, label=f"x{d + 1} true")
            axes[row, 0].scatter(t, observed[:, d], s=7 if n > 100 else 15,
                                 alpha=0.5, label=f"x{d + 1} observed")
        axes[row, 0].set_title(f"{label}; noise std = {np.array2string(std, precision=2)}")
        axes[row, 0].set_xlabel("t")
        axes[row, 0].set_ylabel("state")
        axes[row, 0].legend(ncol=3, fontsize=7)
        axes[row, 1].plot(truth[:, 0], truth[:, 1], color="black", lw=1.5, label="true")
        axes[row, 1].scatter(observed[:, 0], observed[:, 1], s=7 if n > 100 else 15,
                             alpha=0.35, label="observed")
        axes[row, 1].set_title("Lorenz attractor projection (x1, x2)")
        axes[row, 1].set_xlabel("x1")
        axes[row, 1].set_ylabel("x2")
        axes[row, 1].legend()

    fig.suptitle("Lorenz-63 equation-discovery data (Course & Nair, 2023)")
    fig.tight_layout()
    png_path = output_dir / "lorenz63_discovery.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {len(low_data):,} rows to {low_path}")
    print(f"Wrote {len(high_data):,} rows to {high_path}")
    print(f"Wrote plot to {png_path}")


if __name__ == "__main__":
    main()
