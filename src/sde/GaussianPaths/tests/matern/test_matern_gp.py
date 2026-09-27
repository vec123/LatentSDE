import jax
import jax.numpy as jnp
import optax

from src.sde.GaussianPaths.core.gp_path import GaussianPathSDE
from src.sde.GaussianPaths.means.matern_52_mean import MaternMeanModel
from src.sde.GaussianPaths.Variances.matern_52_var import MaternCovariance

from src.sde.tests.gp_ground_truths import get_ground_truth_cov, get_ground_truth_mean
from src.sde.tests.plot_helper import plot_covar_components_dxd, plot_trajs, plot_gp_comparison
from src.sde.tests.plot_helper import plot_gp
from src.sde.tests.data_helper import generate_vanderpol_data
from src.sde.tests.gaussian_path.gp_helpers import run_sde_verification

from src.sde.tests.gaussian_path.gp_helpers import run_sde_verification

from jax import config
config.update("jax_enable_x64", True)


if __name__ == "__main__":
    D = 2
    T = 20.0 
    eps = 1e-6
    NUM_TIMES = 50
    NUM_BASIS = 40 
    t_array = jnp.linspace(0, T, NUM_TIMES)

    mean_traj = generate_vanderpol_data(num_trajectories=1,T =T, num_points=NUM_TIMES, noise_scale = 0.0)
    plot_trajs(mean_traj, filename="mean_vanderpol_traj")
    t_eval, full_traj = mean_traj[0]
    
    dt = t_array[1] - t_array[0]
    x2_approx = jnp.gradient(full_traj, dt)
    def vdp_mean_fn(t):
        return jnp.array([
            jnp.interp(t, t_eval, full_traj[:, 0]), 
            jnp.interp(t, t_eval, full_traj[:, 1])  
        ])

    mu_true = jax.vmap(vdp_mean_fn)(t_array) 
    S_true = jax.vmap(get_ground_truth_cov)(t_array)     
    plot_gp(t_array,vdp_mean_fn, get_ground_truth_cov, filename ="vanderpol_gp.png" )
    def G_fn(t):
        return jnp.array([[0.1, 0.0], [0.1, 0.1]])                   
    sde_bridge_true = GaussianPathSDE(
        mean_fn=vdp_mean_fn,
        dmean_dt_fn=lambda t: jax.jacobian(vdp_mean_fn)(t),
        Sigma_fn=get_ground_truth_cov,
        dSigma_dt_fn=lambda t: jax.jacobian(get_ground_truth_cov)(t),
        G_fn=G_fn
    )

    key = jax.random.PRNGKey(42)
    ground_truth_trajs = sde_bridge_true.sample_trajectories(num_trajectories=100, t_eval= t_eval, key=key)
    plot_trajs(ground_truth_trajs, filename="simulated_ground_truth_vanderpol_traj")

    all_t = jnp.concatenate([t for t, y in ground_truth_trajs])
    all_y = jnp.concatenate([y for t, y in ground_truth_trajs])
    mean_model = MaternMeanModel(all_t, all_y, length_scale=1, sigma_f=1.0, noise_var=1e-3)
    var_model = MaternCovariance(
        D=D, time_interval=(0.0, T),
        type="full", 
        noise_var=1e-3)
    basis_centers = jnp.linspace(0, T, NUM_BASIS)
    params_R = jax.random.normal(jax.random.PRNGKey(0), (var_model.dim_R, NUM_BASIS + 1)) * 1e-2
    params_L = jax.random.normal(jax.random.PRNGKey(1), (D, NUM_BASIS + 1)) * 0.1
    alpha, beta, _sigma_ = 1.0, 1.0, 1.0

    params ={
        "params_R": params_R,
        "params_L": params_L,
        "basis_centers": basis_centers,
        "logit_length_scale": jnp.array(0.0),
        }
    
    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(params)

    def compute_loss(params, ground_truth_trajs, mean_model):
        W_R, W_L = params["params_R"], params["params_L"]
        basis_centers = params["basis_centers"]
        length_scale = jax.nn.sigmoid(params["logit_length_scale"])
        
        def Sigma_fn(t):
            return var_model.get_cov(t, W_R, W_L, basis_centers, length_scale, alpha, beta, _sigma_)
        H = jnp.array([[1.0, 0.0]])
        
        def logpdf_full(y, mu, sig):
            mu_obs = H @ mu
            sig_obs = H @ sig @ H.T
            eps = 1e-6
            return -0.5 * (jnp.log(sig_obs[0, 0] + eps) + 
                        (y - mu_obs[0])**2 / (sig_obs[0, 0] + eps) + 
                        jnp.log(2 * jnp.pi))

        total_loss = 0.0
        for t_data, y_data in ground_truth_trajs:
            mu_t = jax.vmap(mean_model)(t_data)
            Sigma_t = jax.vmap(Sigma_fn)(t_data)
          
            loss = jax.vmap(logpdf_full)(y_data, mu_t, Sigma_t)
            total_loss += jnp.sum(loss)
                    
        return -total_loss
    
    @jax.jit(static_argnames=['mean_model'])
    def train_step(params, opt_state, trajs, mean_model):
        loss, grads = jax.value_and_grad(compute_loss)(params, trajs, mean_model)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for step in range(1000):
        params, opt_state, loss = train_step(params, opt_state, ground_truth_trajs, mean_model)
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

            W_R, W_L = params["params_R"], params["params_L"]
            basis_centers =params["basis_centers"]
            length_scale = jax.nn.sigmoid(params["logit_length_scale"])
            mu_pred = jax.vmap(mean_model)(t_array)
            S_pred = jax.vmap(
                var_model.get_cov, 
                in_axes=(0, None, None, None, None, None, None, None)
            )(t_array, W_R, W_L, basis_centers, length_scale, alpha, beta, _sigma_)
    
            plot_gp_comparison(t_array, 
                                mu_true=mu_true,
                                S_true=S_true,
                                mu_pred=mu_pred,
                                S_pred=S_pred,
                                dim=2, 
                                trajs=None,
                                filename =f"gp_comparison_{step}.png")
           
            def current_Sigma_fn(t):
                return var_model.get_cov(t, W_R, W_L, basis_centers, length_scale, alpha, beta, _sigma_)
            def current_dSigma_dt_fn(t):
                return jax.jacobian(current_Sigma_fn)(t)
            
            # 4. Initialize SDE bridge with these functional closures
            sde_bridge_approx = GaussianPathSDE(
                mean_fn=mean_model,
                dmean_dt_fn=lambda t: jax.jacobian(mean_model)(t),
                Sigma_fn=current_Sigma_fn,
                dSigma_dt_fn=current_dSigma_dt_fn,
                G_fn=G_fn
            )
            
            # 5. Sample and Plot
            key = jax.random.PRNGKey(42)
            approximated_trajs = sde_bridge_approx.sample_trajectories(num_trajectories=100, t_eval=t_eval, key=key)
            plot_trajs(approximated_trajs, filename=f"simulated_approx_vanderpol_traj_{step}.png")