"""
Figures of the two posteriors over latent paths, one panel per state.

    plot_variational_posterior   the approximate variational posterior q(z(t)) = N(m(t), diag v(t)): m and m +- 2 sd
    plot_posterior               the posterior, the learned SDE dz = f(z) dt + g dW: paths from z(t_0) ~ q(z(t_0)),
                                 with the drift weights drawn from their approximate variational posterior (one draw
                                 for each path) or fixed point estimates; median and 2.5 % to 97.5 % quantiles
    plot_learned_dynamics        a path of the learned ODE dz = f(z) dt (no noise) against the true path from the same
                                 start, over time and as a phase portrait

Only numpy and matplotlib: arrays can come from numpy, torch or JAX, and the module loads by its file path (it has
no package imports), e.g. from the JAX project. With the true state, the title gives the RMSE of the centre line and
the fraction of the true path in the band.
"""
import numpy as np
from matplotlib.figure import Figure

TRUTH, OBSERVATIONS, GRID = "#0b0b0b", "#52514e", "#e4e3df"
VARIATIONAL, POSTERIOR, LEARNED = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 of the dataviz palette


def _numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def plot_variational_posterior(path, t, mean, sd, truth=None, observations=None, names=None, title=None) -> dict:
    """
    Mean and mean +- 2 sd of q(z(t)) on t (T,); mean and sd (T, d). truth = (t_true, x_true) and
    observations = (t_obs, y_obs) are optional, y_obs (n, d) with NaN where a state is not observed.
    Returns the metrics (empty without truth).
    """
    mean, sd = _numpy(mean), _numpy(sd)
    return _plot(path, _numpy(t), mean, mean - 2 * sd, mean + 2 * sd, VARIATIONAL, ("mean", "±2 sd"),
                 title or "approximate variational posterior q(z(t))", truth, observations, names)


def plot_posterior(path, t, paths, truth=None, observations=None, names=None, title=None, n_shown=5) -> dict:
    """Median and 2.5 % to 97.5 % quantiles of the SDE paths (T, n, d) on t (T,), and n_shown single paths."""
    paths = _numpy(paths)
    lower, median, upper = np.nanquantile(paths, [0.025, 0.5, 0.975], axis=1)
    return _plot(path, _numpy(t), median, lower, upper, POSTERIOR, ("median", "95 % of paths"),
                 title or "posterior: SDE paths from z(t_0) ~ q(z(t_0))", truth, observations, names,
                 shown=paths[:, :n_shown])


def plot_learned_dynamics(path, t, x, truth, observations=None, names=None, title=None, phase=(0, -1)) -> dict:
    """
    Path x (T, d) of the learned ODE on t (T,) against the true path truth = (t_true, x_true) from the same start,
    one panel per state and a phase portrait of the states phase. Returns the RMSE per state and valid_time: the
    first time at which |x - x_true| exceeds 0.4 times the RMS spread of the true path about its mean (None if it
    never does).
    """
    t, x = _numpy(t), _numpy(x)
    t_true, x_true = _numpy(truth[0]), _numpy(truth[1])
    d = x.shape[1]
    x = np.where(np.isfinite(x), x, np.nan)  # a diverging path ends where it leaves the floats
    on_t = np.stack([np.interp(t, t_true, x_true[:, i]) for i in range(d)], -1)
    error = np.linalg.norm(x - on_t, axis=1)
    spread = np.sqrt(np.mean(np.sum((on_t - on_t.mean(0)) ** 2, 1)))
    beyond = ~(error <= 0.4 * spread)
    metrics = {"rmse": np.sqrt(np.nanmean((x - on_t) ** 2, 0)).tolist(),
               "valid_time": float(t[np.argmax(beyond)]) if beyond.any() else None}
    names = names or [f"z{i + 1}" for i in range(d)]
    title = (title or "learned ODE (posterior-mean coefficients, no noise) and true ODE from the same start")
    valid = (f"until t = {metrics['valid_time']:.2f}" if metrics["valid_time"] is not None
             else f"on all of [{t[0]:.3g}, {t[-1]:.3g}]")
    title += f"\nRMSE {np.round(metrics['rmse'], 3)}, error below 0.4 x RMS spread of the true path {valid}"

    fig = Figure(figsize=(14, 2.6 * d + 0.8), layout="constrained")
    grid = fig.add_gridspec(d, 2, width_ratios=(2.2, 1))
    axes = [fig.add_subplot(grid[i, 0]) for i in range(d)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    for i, ax in enumerate(axes):
        ax.plot(t_true, x_true[:, i], color=TRUTH, lw=1, label="true ODE")
        ax.plot(t, x[:, i], color=LEARNED, lw=1.5, label="learned ODE")
        visible = [x_true[:, i]]
        if observations is not None:
            ax.plot(_numpy(observations[0]), _numpy(observations[1])[:, i], "o", color=OBSERVATIONS, ms=3.5,
                    mec="white", mew=0.5, label="observations")
            visible.append(_numpy(observations[1])[:, i])
        _limits(ax.set_ylim, visible)  # the truth and the data set the limits, so a diverging path cannot
        ax.set_ylabel(names[i])
        _style(ax)
    axes[0].set_title(title, fontsize="medium", loc="left")
    axes[-1].set_xlabel("t")

    a, b = phase
    ax = fig.add_subplot(grid[:, 1])
    ax.plot(x_true[:, a], x_true[:, b], color=TRUTH, lw=0.8)
    ax.plot(x[:, a], x[:, b], color=LEARNED, lw=1.2)
    start, = ax.plot(x_true[0, a], x_true[0, b], "o", color=TRUTH, ms=8, mec="white", mew=1)
    _limits(ax.set_xlim, [x_true[:, a]])
    _limits(ax.set_ylim, [x_true[:, b]])
    ax.set_xlabel(names[a])
    ax.set_ylabel(names[b])
    ax.set_title("phase portrait", fontsize="medium")
    _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles + [start], labels + ["start"], loc="outside lower center", ncols=4, fontsize="small",
               frameon=False)
    fig.savefig(path, dpi=150)
    return metrics


def _limits(set_limits, arrays) -> None:
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if values.size:
        low, high = values.min(), values.max()
        pad = 0.05 * (high - low) or 1.0
        set_limits(low - pad, high + pad)


def _style(ax) -> None:
    ax.grid(color=GRID, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def _plot(path, t, centre, lower, upper, color, labels, title, truth, observations, names, shown=None) -> dict:
    d = centre.shape[1]
    fig = Figure(figsize=(10, 2.6 * d + 0.8), layout="constrained")
    axes = fig.subplots(d, 1, sharex=True, squeeze=False)[:, 0]
    metrics = {}
    if truth is not None:
        t_true, x_true = _numpy(truth[0]), _numpy(truth[1])
        x = np.stack([np.interp(t, t_true, x_true[:, i]) for i in range(d)], -1)
        metrics = {"rmse": np.sqrt(np.mean((centre - x) ** 2, 0)).tolist(),
                   "coverage": np.mean((lower <= x) & (x <= upper), 0).tolist()}
        title += (f"\nRMSE of the {labels[0]} {np.round(metrics['rmse'], 3)}, "
                  f"true path in the band {np.round(metrics['coverage'], 2)}")
    if observations is not None:
        t_obs, y_obs = _numpy(observations[0]), _numpy(observations[1])
    for i, ax in enumerate(axes):
        ax.fill_between(t, lower[:, i], upper[:, i], color=color, alpha=0.25, lw=0, label=labels[1])
        if shown is not None:
            for k in range(shown.shape[1]):
                ax.plot(t, shown[:, k, i], color=color, lw=0.6, alpha=0.6, label=None if k else "single paths")
        ax.plot(t, centre[:, i], color=color, lw=1.5, label=labels[0])
        visible = [lower[:, i], upper[:, i]]
        if truth is not None:
            ax.plot(t_true, x_true[:, i], color=TRUTH, lw=1, label="true")
            visible.append(x_true[:, i])
        if observations is not None:
            ax.plot(t_obs, y_obs[:, i], "o", color=OBSERVATIONS, ms=3.5, mec="white", mew=0.5, label="observations")
            visible.append(y_obs[:, i])
        # the band, the truth and the observations set the limits, so a diverging single path cannot
        visible = np.concatenate(visible)
        visible = visible[np.isfinite(visible)]
        if visible.size:
            low, high = visible.min(), visible.max()
            pad = 0.05 * (high - low) or 1.0
            ax.set_ylim(low - pad, high + pad)
        ax.set_ylabel(names[i] if names else f"z{i + 1}")
        ax.grid(color=GRID, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small", frameon=False)
    axes[0].set_title(title, fontsize="medium")
    axes[-1].set_xlabel("t")
    fig.savefig(path, dpi=150)
    return metrics
