import jax
import jax.numpy as jnp

from src.sde.GaussianPaths.core.gp_path import GaussianPathSDE
from src.sde.GaussianPaths.means.matern_52_mean import MaternMeanModel
from src.sde.GaussianPaths.Variances.matern_52_var import MaternCovariance

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

   # mu_true = jax.vmap(get_ground_truth_mean)(t_array)

    trajs = generate_vanderpol_data(num_trajectories=100,T =T, num_points=NUM_TIMES, noise_scale = 1.0)
    plot_trajs(trajs)
    all_t = jnp.concatenate([t for t, y in trajs])
    all_y = jnp.concatenate([y for t, y in trajs])

    mean_model = MaternMeanModel(all_t, all_y, length_scale=1, sigma_f=1.0)
    var_model = MaternCovariance(D=D, time_interval=(0.0, T), type="full")
    params_R = jax.random.normal(jax.random.PRNGKey(0), (var_model.dim_R, NUM_TIMES + 1)) * 1e-2
    params_L = jax.random.normal(jax.random.PRNGKey(1), (D, NUM_TIMES + 1)) * 0.1

    
    sde_bridge = GaussianPathSDE(
        mean_fn=vdp_mean_fn,
        dmean_dt_fn=lambda t: jax.jacobian(vdp_mean_fn)(t),
        Sigma_fn=get_ground_truth_cov,
        dSigma_dt_fn=lambda t: jax.jacobian(get_ground_truth_cov)(t),
        G_fn=G_fn
    )
    run_sde_verification(sde_bridge,t_array,100)