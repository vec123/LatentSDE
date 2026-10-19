"""Generate overlapping noisy Lorenz-63 observations and save CSV/PNG artifacts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def lorenz_rhs(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])


def rk4_step(state, dt):
    k1 = lorenz_rhs(state)
    k2 = lorenz_rhs(state + 0.5 * dt * k1)
    k3 = lorenz_rhs(state + 0.5 * dt * k2)
    k4 = lorenz_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(initial_state, dt, num_steps):
    states = np.empty((num_steps + 1, 3), dtype=float)
    states[0] = initial_state
    for i in range(num_steps):
        states[i + 1] = rk4_step(states[i], dt)
    return states


def main():
    output_dir = Path(__file__).resolve().parents[2] / "data" / "lorenz_63"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_trajectories = 20
    dt = 0.01
    duration = 30.0
    num_steps = round(duration / dt)
    observation_noise_std = 1.0
    # Small perturbations around the standard Lorenz initial condition expose
    # the system's sensitivity while keeping every trajectory in a comparable range.
    initial_states = [
        np.array([1.0 + 0.05 * i, 1.0 - 0.03 * i, 1.0 + 0.02 * i])
        for i in range(num_trajectories)
    ]
    times = np.arange(num_steps + 1) * dt
    trajectories = np.stack([simulate(x0, dt, num_steps) for x0 in initial_states])
    rng = np.random.default_rng(20260927)
    noisy_trajectories = trajectories + rng.normal(
        loc=0.0, scale=observation_noise_std, size=trajectories.shape
    )

    frame = pd.DataFrame({
        "t": np.tile(times, num_trajectories),
        "x1": noisy_trajectories[:, :, 0].reshape(-1),
        "x2": noisy_trajectories[:, :, 1].reshape(-1),
        "x3": noisy_trajectories[:, :, 2].reshape(-1),
        "x1_true": trajectories[:, :, 0].reshape(-1),
        "x2_true": trajectories[:, :, 1].reshape(-1),
        "x3_true": trajectories[:, :, 2].reshape(-1),
        "trajectory_id": np.repeat(np.arange(num_trajectories), num_steps + 1),
        "time_step": np.tile(np.arange(num_steps + 1), num_trajectories),
    })
    csv_path = output_dir / "lorenz_trajectories.csv"
    frame.to_csv(csv_path, index=False)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.turbo(np.linspace(0.02, 0.98, num_trajectories))
    for trajectory, color in zip(noisy_trajectories, colors):
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=color, alpha=0.55, linewidth=0.55)
    ax.set_title(
        "Noisy Lorenz-63 trajectories "
        f"(σ=10, ρ=28, β=8/3; observation noise sd={observation_noise_std:g})"
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    fig.tight_layout()
    png_path = output_dir / "lorenz_trajectories.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {num_trajectories} trajectories and {len(frame):,} rows to {csv_path}")
    print(f"Wrote trajectory plot to {png_path}")


if __name__ == "__main__":
    main()
