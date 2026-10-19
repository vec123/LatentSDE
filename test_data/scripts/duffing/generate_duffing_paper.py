"""Generate the low-noise Duffing oscillator discovery datasets from Course & Nair (2023).

The system is x1' = x2, x2' = -(x1^3 - x1) - 0.35 x2 with x(0) = [3, 2] on t in [0, 20].
The paper's low-noise regime uses 20 independent datasets of 32 observations with
1%-of-half-range noise. Each CSV stores noisy observations in x1/x2 and clean states
in x1_true/x2_true.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INITIAL_STATE = np.array([3.0, 2.0])
DAMPING = 0.35
T0, T1 = 0.0, 20.0
INTEGRATION_DT = 0.001
NUM_OBSERVATIONS = 32
NOISE_FRACTION = 0.01
NUM_DATASETS = 20
SEED = 20260929


def duffing_rhs(x):
    a, b = x
    return np.array([b, -(a ** 3 - a) - DAMPING * b])


def rk4_trajectory():
    n_steps = round((T1 - T0) / INTEGRATION_DT)
    times = np.linspace(T0, T1, n_steps + 1)
    states = np.empty((n_steps + 1, 2), dtype=float)
    states[0] = INITIAL_STATE
    for k in range(n_steps):
        x = states[k]
        h = INTEGRATION_DT
        k1 = duffing_rhs(x)
        k2 = duffing_rhs(x + 0.5 * h * k1)
        k3 = duffing_rhs(x + 0.5 * h * k2)
        k4 = duffing_rhs(x + h * k3)
        states[k + 1] = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return times, states


def main():
    output_dir = Path(__file__).resolve().parents[2] / "data" / "duffing"
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_times, dense_states = rk4_trajectory()
    rng = np.random.default_rng(SEED)

    times = np.linspace(T0, T1, NUM_OBSERVATIONS)
    truth = np.column_stack([
        np.interp(times, dense_times, dense_states[:, d]) for d in range(2)
    ])
    half_range = 0.5 * (truth.max(axis=0) - truth.min(axis=0))
    obs_std = NOISE_FRACTION * half_range

    frames = []
    for dataset_id in range(NUM_DATASETS):
        noisy = truth + rng.normal(0.0, obs_std, size=truth.shape)
        frames.append(pd.DataFrame({
            "t": times,
            "x1": noisy[:, 0], "x2": noisy[:, 1],
            "x1_true": truth[:, 0], "x2_true": truth[:, 1],
            "trajectory_id": dataset_id,
            "time_step": np.arange(NUM_OBSERVATIONS),
            "x1_obs_std": obs_std[0], "x2_obs_std": obs_std[1],
        }))
    data = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / "duffing_discovery_low_noise_low_data.csv"
    data.to_csv(csv_path, index=False, float_format="%.10g")

    first = data[data.trajectory_id == 0]
    observed = first[["x1", "x2"]].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for d in range(2):
        axes[0].plot(dense_times, dense_states[:, d], lw=1.5, label=f"x{d + 1} true")
        axes[0].scatter(times, observed[:, d], s=15, alpha=0.7, label=f"x{d + 1} observed")
    axes[0].set_title(
        f"Low noise / low data ({NUM_OBSERVATIONS} points); "
        f"noise std = {np.array2string(obs_std, precision=3)}"
    )
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("state")
    axes[0].legend(ncol=2, fontsize=7)
    axes[1].plot(dense_states[:, 0], dense_states[:, 1], color="black", lw=1.5, label="true")
    axes[1].scatter(observed[:, 0], observed[:, 1], s=15, alpha=0.6, label="observed")
    axes[1].set_title("Phase portrait (x1, x2)")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    axes[1].legend()

    fig.suptitle("Duffing oscillator equation-discovery data (Course & Nair, 2023)")
    fig.tight_layout()
    png_path = output_dir / "duffing_discovery.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {len(data):,} rows to {csv_path}")
    print(f"Wrote plot to {png_path}")


if __name__ == "__main__":
    main()
