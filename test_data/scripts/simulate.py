import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import os
from test_data.src.euler_mayurama import simulate_batch
from test_data.src.systems import linear_2d, nonlinear_2d, vanderpol_2d

# --- 1. Run Simulation ---
N = 50
t_span = (0, 10)
dt = 0.001
key = jax.random.PRNGKey(42)
x0 = jax.random.multivariate_normal(key, jnp.array([0.1, 0.1]), jnp.eye(2) * 0.001, (N,))

# Run the simulation (using the simulate_batch function from previous step)
trajectories = simulate_batch(vanderpol_2d, x0, t_span, dt, key) # Shape: (N, steps, 2)

# --- 2. Save Trajectories as CSV ---
def save_to_csv(trajs, filename="trajectories.csv"):
    # Reshape: (N * steps, 2)
    data = trajs.reshape(-1, 2)
    df = pd.DataFrame(data, columns=["x1", "x2"])
    
    # Add index info (Trajectory ID and Time step)
    steps = trajs.shape[1]
    df["trajectory_id"] = jnp.repeat(jnp.arange(N), steps)
    df["time_step"] = jnp.tile(jnp.arange(steps), N)
    
    df.to_csv(filename, index=False)
    print(f"Trajectories saved to {filename}")

# --- 3. Save Plot as PNG ---
def save_plot(trajs, filename="trajectories.png"):
    plt.figure(figsize=(8, 6))
    for i in range(trajs.shape[0]):
        plt.plot(trajs[i, :, 0], trajs[i, :, 1], alpha=0.3, color='blue')
    
    plt.title("2D System Trajectories")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved to {filename}")

# Execute
save_to_csv(trajectories)
save_plot(trajectories)