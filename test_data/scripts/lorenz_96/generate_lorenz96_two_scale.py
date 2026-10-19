"""Generate the two-scale (two-level) Lorenz-96 system.

Slow variables X_i (i = 1..K, cyclic) couple to fast variables Y_{j,i}
(j = 1..J per slow variable, cyclic within each group and across groups):

    dX_i/dt = (X_{i+1} - X_{i-2}) X_{i-1} - X_i + F - (hc/b) sum_j Y_{j,i}
    dY_{j,i}/dt = cb (Y_{j-1,i} - Y_{j+2,i}) Y_{j+1,i} - c Y_{j,i} + (hc/b) X_i

Parameters K=36, J=10, F=10, h=1, b=10, c=10 follow Lorenz (1996). Only the
slow variables X are observed (noisy); the fast variables Y stand in for the
unresolved sub-grid process and are kept in the companion npz for reference.

Outputs, both under test_data/data/lorenz_96:
    lorenz96_two_scale_trajectories.csv   noisy/true X, several noise draws of one trajectory
    lorenz96_two_scale_full_state.npz     t, X_true (T,K), Y_true (T,J*K), x0, y0, params
    lorenz96_two_scale.png                slow-variable heatmap and a scale-separation close-up
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


K = 36          # number of slow variables
J = 10          # fast variables per slow variable
FORCING = 10.0
H = 1.0
B = 10.0
C = 10.0
T_SPINUP = 5.0
T0, T1 = 0.0, 4.0
INTEGRATION_DT = 1e-3
OBS_SUBSTEPS = 10  # RK4 steps between observations, so dt_obs = 1e-2
NUM_OBSERVATIONS = round((T1 - T0) / (INTEGRATION_DT * OBS_SUBSTEPS)) + 1
NOISE_FRACTION = 0.02
NUM_DATASETS = 10
SEED = 20261015


def rhs(state):
    x, y = state[:K], state[K:]
    dx = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + FORCING - (H * C / B) * y.reshape(K, J).sum(axis=1)
    dy = C * B * (np.roll(y, 1) - np.roll(y, -2)) * np.roll(y, -1) - C * y + (H * C / B) * np.repeat(x, J)
    return np.concatenate([dx, dy])


def rk4_step(state, h):
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * h * k1)
    k3 = rhs(state + 0.5 * h * k2)
    k4 = rhs(state + h * k3)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_trajectory(state0, num_steps, dt):
    states = np.empty((num_steps + 1, state0.size), dtype=float)
    states[0] = state = state0
    for i in range(num_steps):
        state = rk4_step(state, dt)
        states[i + 1] = state
    return states


def main():
    output_dir = Path(__file__).resolve().parents[2] / "data" / "lorenz_96"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    state0 = np.concatenate([
        FORCING * np.ones(K) + rng.standard_normal(K) * 0.5,
        rng.standard_normal(K * J) * 0.5,
    ])
    spinup_steps = round(T_SPINUP / INTEGRATION_DT)
    state0 = rk4_trajectory(state0, spinup_steps, INTEGRATION_DT)[-1]

    dense_steps = round((T1 - T0) / INTEGRATION_DT)
    dense = rk4_trajectory(state0, dense_steps, INTEGRATION_DT)
    dense_times = T0 + np.arange(dense_steps + 1) * INTEGRATION_DT

    obs_idx = np.arange(0, dense_steps + 1, OBS_SUBSTEPS)
    times = dense_times[obs_idx]
    x_true = dense[obs_idx, :K]
    y_true = dense[obs_idx, K:]

    half_range = 0.5 * (x_true.max(axis=0) - x_true.min(axis=0))
    obs_std = NOISE_FRACTION * half_range

    frames = []
    for dataset_id in range(NUM_DATASETS):
        noisy = x_true + rng.normal(0.0, obs_std, size=x_true.shape)
        frame = pd.DataFrame(noisy, columns=[f"X{i + 1}" for i in range(K)])
        for i in range(K):
            frame[f"X{i + 1}_true"] = x_true[:, i]
        frame["t"] = times
        frame["trajectory_id"] = dataset_id
        frame["time_step"] = np.arange(len(times))
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / "lorenz96_two_scale_trajectories.csv"
    data.to_csv(csv_path, index=False, float_format="%.10g")

    npz_path = output_dir / "lorenz96_two_scale_full_state.npz"
    np.savez_compressed(npz_path, t=times, X_true=x_true.astype(np.float32), Y_true=y_true.astype(np.float32),
                         x0=state0[:K], y0=state0[K:], K=K, J=J, F=FORCING, h=H, b=B, c=C,
                         noise_fraction=NOISE_FRACTION, obs_std=obs_std)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    image = axes[0].imshow(x_true, origin="lower", aspect="auto", extent=(1, K, T0, T1),
                            cmap="RdBu_r", interpolation="nearest")
    fig.colorbar(image, ax=axes[0], label="X_i")
    axes[0].set_title("Slow variables X_i")
    axes[0].set_xlabel("slow index i")
    axes[0].set_ylabel("t")

    probe = 2  # 0-based slow index to zoom in on
    axes[1].plot(times, x_true[:, probe], color="black", lw=1.5, label=f"X{probe + 1} true")
    axes[1].scatter(times, data[data.trajectory_id == 0][f"X{probe + 1}"], s=10, alpha=0.6,
                     label=f"X{probe + 1} observed")
    for j in range(J):
        axes[1].plot(times, y_true[:, probe * J + j] / B + FORCING, color="tab:orange", lw=0.4, alpha=0.5,
                      label="Y_{j,i}/b + F" if j == 0 else None)
    axes[1].set_title(f"Scale separation at slow index {probe + 1}")
    axes[1].set_xlabel("t")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"Two-scale Lorenz-96, K={K}, J={J}, F={FORCING:g}, h={H:g}, b={B:g}, c={C:g}")
    fig.tight_layout()
    png_path = output_dir / "lorenz96_two_scale.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {len(data):,} rows to {csv_path}")
    print(f"Wrote full state to {npz_path}")
    print(f"Wrote plot to {png_path}")


if __name__ == "__main__":
    main()
