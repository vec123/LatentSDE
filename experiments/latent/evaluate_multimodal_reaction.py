"""
Does a LatentSDE trained on multimodal_reaction_test (configs/multimodal_reaction_*.yaml) learn the correct latent
dynamics? The latent coordinates are the model's own, so every check goes through the decoder, into the two
quantities that define the truth: the rotation phase (from the decoded image, -arg c1(R0) as in
test_data/src/multimodal_reaction.py) and gamma (the decoded biomarker).

Posterior, conditioned on both modalities, on the images only, or on the biomarker only, at the truth's frame
times: phase error, the fraction of imaging gaps with the correct number of turns, and R^2 of gamma.
Prior, simulated from each trajectory's own m(t0) (both modalities) with no observations after it: the true
dynamics are d phase = (omega(beta) + gamma) dt and gamma an OU process (time scale tau, std), so the rate of the
decoded phase against the decoded gamma must have slope 1 and intercept omega(beta), and gamma the true std and
autocorrelation time.

    python -m experiments.latent.evaluate_multimodal_reaction runs/<run_dir> [--split test] [--paths 8]

Writes <run_dir>/latent_dynamics_<split>.png and .json, and prints the metrics.
"""
import argparse
import json
import os

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.multimodal import load_multimodal_reaction
from src.data.pca import PCA
from src.latent.diffusion import diffuse
from test_data.src.multimodal_reaction import R0, angular_mode, angular_mode_matrix
from .run import INK, MUTED, load_config

BOTH, IMAGES, BIO = "#1baf7a", "#2a78d6", "#eb6834"
CONDITIONS = {"both": (), "images only": ("biomarker",), "biomarker only": ("imaging",)}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def decode(model, z, mu, columns, pcas, K0, shape):
    """Decoded phase (..., Q) unwrapped along Q, and decoded gamma (..., Q), of latent paths z (..., Q, dim_z), one
    path at a time (the decoded images of all paths at once would not fit in memory)."""
    lead, Q = z.shape[:-2], z.shape[-2]
    z, mu = z.reshape(-1, Q, z.shape[-1]), mu.reshape(-1, mu.shape[-1])
    phase, gamma = np.empty((len(z), Q)), np.empty((len(z), Q))
    with torch.no_grad():
        for r in range(len(z)):
            x = model.decoder(z[r], mu[r].expand(Q, -1)).mean
            images = pcas["imaging"].decode(x[:, columns["imaging"]]).reshape(Q, *shape)
            phase[r] = -np.unwrap(np.angle(angular_mode(images.numpy(), K0)[:, 0]))
            gamma[r] = pcas["biomarker"].decode(x[:, columns["biomarker"]])[:, 0].numpy()
    return phase.reshape(*lead, Q), gamma.reshape(*lead, Q)


def posterior_paths(model, batch, columns, drop, t_query):
    """Posterior mean m (B, Q, dim_z) at t_query (B, Q), conditioned with the modalities in drop masked out."""
    mask = batch["mask"].clone()
    for name in drop:
        mask[..., columns[name]] = 0
    model.eval()
    with torch.no_grad():
        model.posterior.condition({**batch, "x": batch["x"] * mask, "mask": mask})
        return model.posterior(t_query)[0]


def simulate(model, z0, mu, t, substeps, n_paths, seed=0):
    """Euler-Maruyama prior paths (B, n_paths, Q, dim_z) from z0 (B, dim_z) on the times t (Q,)."""
    torch.manual_seed(seed)
    B, d = z0.shape
    z = z0.repeat_interleave(n_paths, 0)
    mu = mu.repeat_interleave(n_paths, 0)
    u = mu.new_zeros(len(z), 0)
    zs = [z]
    with torch.no_grad():
        for t0, t1 in zip(t[:-1].tolist(), t[1:].tolist()):
            dt = (t1 - t0) / substeps
            for _ in range(substeps):
                G = model.diffusion(z, mu, torch.tensor(t0))
                z = z + model.drift(z, mu, u) * dt + diffuse(G, (dt ** 0.5) * torch.randn_like(z))
            zs.append(z)
    return torch.stack(zs, 1).reshape(B, n_paths, len(t), d)


def acf(x, max_lag):
    """Mean autocorrelation over the rows of x (n, Q), lags 0..max_lag."""
    x = x - x.mean(-1, keepdims=True)
    var = (x * x).mean(-1)
    return np.array([((x[:, :x.shape[1] - k] * x[:, k:]).mean(-1) / var).mean() for k in range(max_lag + 1)])


def time_scale(c, dt):
    """The lag at which the autocorrelation c first falls below 1/e, interpolated."""
    k = int(np.argmax(c < np.exp(-1)))
    return dt * (k - 1 + (c[k - 1] - np.exp(-1)) / (c[k - 1] - c[k])) if k > 0 else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--split", default="test")
    p.add_argument("--paths", type=int, default=8, help="prior paths per trajectory")
    p.add_argument("--substeps", type=int, default=10, help="Euler-Maruyama steps per truth frame")
    p.add_argument("--trajectory", type=int, default=0, help="trajectory shown in the posterior panels")
    args = p.parse_args()

    config = load_config(os.path.join(args.run_dir, "config.yaml"))
    model = torch.load(os.path.join(args.run_dir, "model.pt"), weights_only=False)
    cfg = config["data"][args.split]
    batches, columns = load_multimodal_reaction(cfg["path"], cfg["pca_dir"], 1)
    batch = batches[0]
    pcas = {}
    for name in columns:
        pcas[name] = PCA()
        pcas[name].load_state_dict(torch.load(os.path.join(cfg["pca_dir"], name, "model.pt"), weights_only=True))
    with h5py.File(cfg["path"].replace("observations_", "truth_"), "r") as f:
        t_frames, x_grid = f["t_frames"][:], f["x"][:]
        stride, tau, gamma_std = int(f.attrs["frame_stride"]), float(f.attrs["tau"]), float(f.attrs["gamma_std"])
        phase_true, gamma_true = f["phase"][:], f["gamma"][:, ::stride]
        omega = np.array([np.polyfit(t_frames, p0, 1)[0] for p0 in f["phase0"][:]])
    with h5py.File(cfg["path"], "r") as g:
        visits = [g["imaging/t"][i, :n] for i, n in enumerate(g["imaging/count"][:])]
        bio = [(g["biomarker/t"][i, :n], g["biomarker/y"][i, :n]) for i, n in enumerate(g["biomarker/count"][:])]
    K0, shape = angular_mode_matrix(x_grid, np.array([R0])), (len(x_grid), len(x_grid))
    B = batch["t"].shape[0]
    tq = torch.as_tensor(t_frames, dtype=batch["t"].dtype).repeat(B, 1)  # a copy: forward-mode AD needs one
    observed = batch["mask"].any(-1)
    span = [(batch["t"][b][observed[b]].min().item(), batch["t"][b][observed[b]].max().item()) for b in range(B)]
    inside = np.array([(t_frames >= a) & (t_frames <= b) for a, b in span])

    metrics, post = {"posterior": {}}, {}
    for name, drop in CONDITIONS.items():
        m = posterior_paths(model, batch, columns, drop, tq)
        phase, gamma = decode(model, m, batch["mu"], columns, pcas, K0, shape)
        post[name] = phase, gamma
        err = wrap(phase - phase_true)[inside]
        correct = []
        for b, v in enumerate(visits):
            k = np.searchsorted(t_frames, v)
            correct += list(np.abs(np.diff(phase[b, k]) - np.diff(phase_true[b, k])) < np.pi)
        resid = (gamma - gamma_true)[inside]
        metrics["posterior"][name] = {
            "phase_rmse_rad": float(np.sqrt(np.mean(err ** 2))),
            "imaging_gaps_with_correct_turns": float(np.mean(correct)),
            "gamma_r2": float(1 - np.mean(resid ** 2) / np.var(gamma_true[inside])),
        }

    t = torch.as_tensor(t_frames, dtype=batch["t"].dtype)
    m0 = posterior_paths(model, batch, columns, (), torch.as_tensor([[a] for a, _ in span], dtype=t.dtype))[:, 0]
    i0 = [int(np.searchsorted(t_frames, a)) for a, _ in span]
    start = min(i0)
    z = simulate(model, m0, batch["mu"], t[start:], args.substeps, args.paths)
    phase_sim, gamma_sim = decode(model, z, batch["mu"].unsqueeze(1).expand(-1, args.paths, -1), columns, pcas, K0,
                                  shape)
    w = 5  # frames: rate and gamma averaged over 1 time unit
    dt = t_frames[1] - t_frames[0]
    rate = (phase_sim[..., w:] - phase_sim[..., :-w]) / (w * dt)
    gamma_w = np.stack([gamma_sim[..., k:k + w].mean(-1) for k in range(gamma_sim.shape[-1] - w)], -1)
    excess = rate - omega[:, None, None]
    slope, intercept = np.polyfit(gamma_w.ravel(), excess.ravel(), 1)
    max_lag = int(round(4 * tau / dt))
    acf_sim, acf_true = acf(gamma_sim.reshape(-1, gamma_sim.shape[-1]), max_lag), acf(gamma_true, max_lag)
    metrics["prior"] = {
        "rate_minus_omega_vs_gamma_slope": float(slope), "rate_minus_omega_vs_gamma_intercept": float(intercept),
        "gamma_std": float(gamma_sim.std()), "gamma_std_true": gamma_std,
        "gamma_time_scale": float(time_scale(acf_sim, dt)), "gamma_time_scale_true": tau,
    }
    metrics["diffusion_trajectory_0"] = model.diffusion(m0, batch["mu"], t[:1])[0].detach().numpy().round(4).tolist()

    b = args.trajectory
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    ax = axes[0, 0]
    for name, color in (("images only", IMAGES), ("both", BOTH)):
        turns = np.where(inside[b], (post[name][0][b] - phase_true[b]) / (2 * np.pi), np.nan)
        ax.plot(t_frames, turns, color=color, lw=1.5, label=f"posterior, {name}")
    for s in visits[b]:
        ax.axvline(s, color=IMAGES, lw=0.8, alpha=0.3)
    ax.set(xlabel="t", ylabel="turns", title=f"phase error of the decoded image, {args.split} trajectory {b}")
    ax = axes[0, 1]
    ax.plot(t_frames, gamma_true[b], color=INK, lw=1, label="true gamma")
    ax.plot(t_frames, np.where(inside[b], post["both"][1][b], np.nan), color=BOTH, lw=1.5,
            label="posterior, both")
    ax.plot(*bio[b], "o", color=BIO, ms=3.5, mec="white", mew=0.5, label="biomarker sample")
    ax.set(xlabel="t", ylabel="gamma", title="decoded gamma")
    ax = axes[1, 0]
    pick = np.random.default_rng(0).choice(gamma_w.size, min(4000, gamma_w.size), replace=False)
    ax.plot(gamma_w.ravel()[pick], excess.ravel()[pick], "o", color=BOTH, ms=2, alpha=0.3, mew=0,
            label="prior paths, 1-unit means")
    g = np.linspace(gamma_w.min(), gamma_w.max(), 2)
    ax.plot(g, g, color=INK, lw=1.2, label="truth: rate - omega = gamma")
    ax.plot(g, slope * g + intercept, color=MUTED, lw=1.2, ls="--", label=f"fit: slope {slope:.2f}")
    ax.set(xlabel="decoded gamma", ylabel="rate - omega(beta)", title="prior: rotation rate against gamma")
    ax = axes[1, 1]
    lags = dt * np.arange(max_lag + 1)
    ax.plot(lags, np.exp(-lags / tau), color=INK, lw=1.2, label=f"truth: exp(-lag / {tau:g})")
    ax.plot(lags, acf_true, color=MUTED, lw=1.2, ls="--", label="true gamma paths")
    ax.plot(lags, acf_sim, color=BOTH, lw=1.5, label="prior paths")
    ax.set(xlabel="lag", ylabel="autocorrelation", title="prior: autocorrelation of gamma")
    for ax in axes.ravel():
        ax.legend(fontsize=8, frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    post_b = metrics["posterior"]
    fig.suptitle(f"{os.path.basename(args.run_dir)}, {args.split}: correct turns between visits: both "
                 f"{post_b['both']['imaging_gaps_with_correct_turns']:.2f}, images only "
                 f"{post_b['images only']['imaging_gaps_with_correct_turns']:.2f}; gamma R^2 (both) "
                 f"{post_b['both']['gamma_r2']:.2f}", color=INK, fontsize=11)
    out = os.path.join(args.run_dir, f"latent_dynamics_{args.split}")
    fig.savefig(out + ".png", dpi=130)
    plt.close(fig)
    json.dump(metrics, open(out + ".json", "w"), indent=1)
    print(json.dumps(metrics, indent=1))
    print(f"Wrote {out}.png and .json")


if __name__ == "__main__":
    main()
