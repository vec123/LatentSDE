import jax
import jax.numpy as jnp

from src.sde.GaussianPaths.core.gp_path import GaussianPathSDE
from src.sde.tests.gp_ground_truths import get_ground_truth_cov, get_ground_truth_mean
from src.sde.tests.plot_helper import plot_covar_components_dxd, plot_trajs
from src.sde.tests.data_helper import generate_vanderpol_data

from src.sde.tests.gaussian_path.gp_helpers import run_sde_verification

# Diffusion matrix (2x2)
def G_fn(t):
    return jnp.array([[0.1, 0.0], [0.1, 0.1]])


if __name__ == "__main__":
    D = 2
    T = 20.0 
    eps = 1e-6
    NUM_TIMES = 100
    t_array = jnp.linspace(0, T, NUM_TIMES)

    vanderpol_traj = generate_vanderpol_data(num_trajectories=1,T =T, num_points=NUM_TIMES, noise_scale = 0.0)
    plot_trajs(vanderpol_traj)
    t_eval, full_traj = vanderpol_traj[0] # x1_noisy is shape (500,)
    
    dt = t_array[1] - t_array[0]
    x2_approx = jnp.gradient(full_traj, dt)
    
    def vdp_mean_fn(t):
        return jnp.array([
            jnp.interp(t, t_eval, full_traj[:, 0]),
            jnp.interp(t, t_eval, full_traj[:, 1])  
        ])


    sde_bridge = GaussianPathSDE(
        mean_fn=vdp_mean_fn,
        # Note: You need a wrapper to make the Jacobian of the function work
        dmean_dt_fn=lambda t: jax.jacobian(vdp_mean_fn)(t),
        Sigma_fn=get_ground_truth_cov,
        dSigma_dt_fn=lambda t: jax.jacobian(get_ground_truth_cov)(t),
        G_fn=G_fn
    )
    key = jax.random.PRNGKey(42)
    trajs = sde_bridge.sample_trajectories(num_trajectories=100, t_eval= t_eval, key=key)
    plot_trajs(trajs, filename="simulated_vanderpol_traj")
   # run_sde_verification(sde_bridge,t_array,100)