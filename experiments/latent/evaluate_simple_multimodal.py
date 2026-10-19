"""
Does a LatentSDE trained on simple_multimodal_test (configs/simple_multimodal*.yaml) learn the true dynamics,
dx = KAPPA y1 dt (+ X_NOISE dW) with y1, y2 OU processes, and does it combine the modalities? The latent
coordinates are the model's own, so every check is in the units of the observations: A (x) and B (y1, y2).

Learned dynamics: at the posterior states of the split's trajectories, the drift pushed through the decoder,
d(decoded)/dt = J_decoder(z) f(z), regressed on the decoded values, for each component only at the states of the
trajectories where its modality is observed (dA/dt with A, dB/dt with B): where a modality is missing, its latent
coordinates are free, and the posterior can park them where the drift does nothing (in
runs/simple_multimodal_neighbors_fixed_x_diffusion_20261017-180049 x of the B-only trajectories at 0, where the
learned dx/dt did not depend on y1), which a pooled fit mixes in. The truth is dA/dt = KAPPA B1 (and not B2),
dB_j/dt = -B_j / TAU; and the diffusion pushed through the decoder, (J G^2 J^T) diagonal, is X_NOISE for A (the
x_noise of the data, 0 by default) and SIGMA sqrt(2 / TAU) for B.
Posterior: the RMSE of the decoded x, y1, y2 against the truth, per group (both, A only, B only), and for the
trajectories with both modalities also with B masked out (A only), the gain of combining.

    python -m experiments.latent.evaluate_simple_multimodal runs/<run_dir> [--split test] [--checkpoint STEP]

--checkpoint reads checkpoints/model_<STEP>.pt (also while the run still trains) for model.pt, and adds _<STEP>
to the names of the files it writes.

Simulation: the learned SDE (drift and diffusion, no observations) from the posterior mean m(0) of each of the
first --n-sim trajectories with A, against their true x, y1, y2.

Writes <run_dir>/dynamics_<split>.png (the posterior of one trajectory with a band of 2 std of the decoded value)
and .json, and simulation_<split>.png, and prints the metrics.
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

from src.data.multimodal import load_multimodal
from src.latent.diffusion import diffuse
from .run import INK, MUTED, load_config

A_COLOR, Y1_COLOR, Y2_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
TRUE_COLOR, SIM_COLOR = "#2a78d6", "#e34948"
GROUPS = ("both", "A only", "B only")
fmt = lambda v: "not separable" if np.isnan(v) else f"{v:+.2f}"
NAMES = ("A", "B1", "B2")


def posterior_mean(model, batch, t_query, drop=None, scale=False):
    """Posterior mean m (B, Q, dim_z) at t_query, conditioned with the columns in drop masked out. Then the steps
    with no value left are removed: the GRU reads every step, and no training batch has an empty step (only the
    padding at the end), so empty steps would be inputs it never saw. Each row needs one value left. With scale,
    also the posterior's scale: log v (B, Q, dim_z), or L (B, Q, dim_z, dim_z) for a full covariance."""
    mask = batch["mask"].clone()
    if drop is not None:
        mask[..., drop] = 0
        keep = mask.any(-1)
        count = keep.sum(1, keepdim=True)
        order = torch.argsort((~keep).long(), dim=1, stable=True)[:, :int(count.max())]  # kept steps first, in order
        pick = lambda v: v.gather(1, order.view(*order.shape, *[1] * (v.dim() - 2)).expand(*order.shape, *v.shape[2:]))
        inside = torch.arange(order.shape[1]) < count
        t = pick(batch["t"])
        batch = {**batch, "t": torch.where(inside, t, t.gather(1, count - 1)),  # padding: the last time repeated
                 "x": pick(batch["x"]), "u": pick(batch["u"])}
        mask = pick(mask)
    model.eval()
    with torch.no_grad():
        model.posterior.condition({**batch, "x": batch["x"] * mask, "mask": mask})
        out = model.posterior(t_query)
    return (out[0], out[1]) if scale else out[0]


def decoded_band(model, m, scale, n: int = 256):
    """The decoded posterior mean, decoder(m), and the std of the decoded value under the posterior N(m, diag v)
    (scale log v) or N(m, L L^T) (scale L), from n samples: each (B, Q, D). The samples share their noise over
    the times, so the sampling error of the std is smooth in t and does not hide its shape."""
    gen = torch.Generator().manual_seed(0)
    eps = torch.randn(n, m.shape[0], 1, m.shape[-1], dtype=m.dtype, generator=gen)
    z = m + ((scale @ eps.unsqueeze(-1)).squeeze(-1) if scale.dim() > m.dim() else (0.5 * scale).exp() * eps)
    with torch.no_grad():
        center = model.decoder(m, m.new_zeros(*m.shape[:-1], 0)).mean
        spread = model.decoder(z, z.new_zeros(*z.shape[:-1], 0)).mean.std(0)
    return center.numpy(), spread.numpy()


def implied(model, z):
    """Decoded values (n, 3), the drift pushed through the decoder (n, 3), and the diffusion pushed through it,
    the square root of the diagonal of J G G^T J^T (n, 3), at latent states z (n, dim_z); G diagonal or full."""
    mu, u = z.new_zeros(len(z), 0), z.new_zeros(len(z), 0)
    decode = lambda zz: model.decoder(zz, mu).mean
    with torch.no_grad():
        f = model.drift(z, mu, u)
        G = model.diffusion(z, mu, z.new_zeros(1))
        x, dx = torch.func.jvp(decode, (z,), (f,))
        J = torch.func.vmap(torch.func.jacrev(lambda zz: model.decoder(zz[None], mu[:1]).mean[0]))(z)
    JG = J @ G if G.dim() > z.dim() else J * G.unsqueeze(1)
    return x.numpy(), dx.numpy(), JG.pow(2).sum(-1).sqrt().numpy()


def simulate_paths(model, t: np.ndarray, z0, substeps: int):
    """Euler-Maruyama paths of the learned prior SDE, with no observations, from z0 (n, dim_z) over the grid t (Q,),
    substeps steps between two grid points: (n, Q, dim_z)."""
    mu, u = z0.new_zeros(len(z0), 0), z0.new_zeros(len(z0), 0)
    z, zs = z0, [z0]
    with torch.no_grad():
        for t0, t1 in zip(t[:-1].tolist(), t[1:].tolist()):
            dt = (t1 - t0) / substeps
            for _ in range(substeps):
                G = model.diffusion(z, mu, torch.tensor(t0))
                z = z + model.drift(z, mu, u) * dt + diffuse(G, dt ** 0.5 * torch.randn_like(z))
            zs.append(z)
    return torch.stack(zs, 1)


def plot_simulation(out: str, t_true, truth, t_sim, sims, title: str):
    """Three panels, x, y1 and y2: the true trajectories (blue) and the simulated ones (red), one each per
    trajectory; truth (n, len(t_true), 3), sims (n, len(t_sim), 3)."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for k, (ax, name) in enumerate(zip(axes, ("x", "y1", "y2"))):
        for i in range(len(truth)):
            ax.plot(t_true, truth[i, :, k], color=TRUE_COLOR, lw=0.7, alpha=0.45,
                    label=f"true ({len(truth)} test trajectories)" if i == 0 else None)
            ax.plot(t_sim, sims[i, :, k], color=SIM_COLOR, lw=0.7, alpha=0.45,
                    label="simulated (learned SDE, from the posterior mean m(0) of each)" if i == 0 else None)
        ax.set(ylabel=name)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    axes[-1].set(xlabel="t")
    fig.suptitle(title, color=INK, fontsize=11)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def load_model(run_dir: str, checkpoint: int = None):
    """model.pt of a run, or checkpoints/model_<checkpoint>.pt (saved on the training device) on the CPU; the name of
    the run for the titles; and the suffix of the output files: "" for model.pt, "_<checkpoint>" for a checkpoint."""
    name = os.path.basename(os.path.normpath(run_dir))
    if checkpoint is None:
        return torch.load(os.path.join(run_dir, "model.pt"), weights_only=False), name, ""
    path = os.path.join(run_dir, "checkpoints", f"model_{checkpoint:06d}.pt")
    return torch.load(path, map_location="cpu", weights_only=False), f"{name} (step {checkpoint})", f"_{checkpoint:06d}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--split", default="test")
    p.add_argument("--every", type=int, default=10, help="truth grid points between two evaluation times")
    p.add_argument("--n-sim", type=int, default=20, help="trajectories in simulation_<split>.png")
    p.add_argument("--checkpoint", type=int, help="step of a checkpoint to read instead of model.pt")
    args = p.parse_args()

    config = load_config(os.path.join(args.run_dir, "config.yaml"))
    model, run_name, suffix = load_model(args.run_dir, args.checkpoint)
    cfg = config["data"][args.split]
    (batch,), columns = load_multimodal(cfg["path"], cfg["modalities"], 1)
    with h5py.File(cfg["path"], "r") as f:
        t_true, x_true, y_true, group = f["truth/t"][:], f["truth/x"][:], f["truth/y"][:], f["group"][:]
        kappa, tau, sigma = (float(f.attrs[k]) for k in ("kappa", "tau", "sigma"))
        x_noise = float(f.attrs.get("x_noise", 0.0))  # data generated before --x-noise has none
        a_t = [f["A/t"][i, :n] for i, n in enumerate(f["A/count"][:])]
        a_y = [f["A/y"][i, :n, 0] for i, n in enumerate(f["A/count"][:])]
        b_t = [f["B/t"][i, :n] for i, n in enumerate(f["B/count"][:])]
        b_y = [f["B/y"][i, :n] for i, n in enumerate(f["B/count"][:])]
    t_eval = t_true[::args.every]
    truth = np.concatenate([x_true[:, ::args.every, None], y_true[:, ::args.every]], -1)  # (B, Q, 3)
    observed = batch["mask"].any(-1)
    span = [(batch["t"][b][observed[b]].min().item(), batch["t"][b][observed[b]].max().item())
            for b in range(len(group))]
    inside = np.array([(t_eval >= a) & (t_eval <= c) for a, c in span])
    tq = torch.as_tensor(t_eval, dtype=batch["t"].dtype).repeat(len(group), 1)

    m = posterior_mean(model, batch, tq)
    z = m[torch.as_tensor(inside)]
    decoded, rate, noise = implied(model, z)
    # the states of the trajectories where each modality is observed (group 0 both, 1 A only, 2 B only)
    traj = np.nonzero(inside)[0]
    seen = {"A": (group != 2)[traj], "B": (group != 1)[traj]}
    # a regressor must be separable: a column the model does not represent (its decoded value varies by less than a
    # tenth of its noise std: it is explained as noise), or one that is a copy of the columns before it, would make
    # the regression ill-conditioned; columns enter in the order A, B1, B2 while the standardized design stays
    # well-conditioned
    with torch.no_grad():
        obs_std = model.decoder(z[:1], z.new_zeros(1, 0)).stddev[0].numpy()
    represented = np.zeros(3, dtype=bool)
    for k in range(3):
        if decoded[:, k].std() <= 0.1 * obs_std[k]:
            continue
        trial = represented.copy()
        trial[k] = True
        cols = decoded[:, trial]
        if np.linalg.cond((cols - cols.mean(0)) / cols.std(0)) < 100:
            represented = trial
    design = np.concatenate([np.ones((len(z), 1)), decoded[:, represented]], 1)
    coef, r2 = {}, {}
    for k, name in enumerate(NAMES):
        sel = seen[name[0]]
        X, y = design[sel], rate[sel, k]
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        coef[name] = np.full(4, np.nan)
        coef[name][np.r_[True, represented]] = c
        r2[name] = 1 - np.sum((y - X @ c) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    true_coef = {"A": [0, 0, kappa, 0], "B1": [0, 0, -1 / tau, 0], "B2": [0, 0, 0, -1 / tau]}
    metrics = {"separable (A, B1, B2)": represented.tolist(),
               "dynamics": {f"d{name}/dt": {"learned (1, A, B1, B2), null: not separable":
                                             [None if np.isnan(v) else round(float(v), 3) for v in coef[name]],
                                             "r2": round(float(r2[name]), 3), "true": true_coef[name]}
                            for name in NAMES},
               "fitted at": "dA/dt: the trajectories with A; dB1/dt, dB2/dt: the trajectories with B",
               "diffusion_in_observation_units": {"learned (A, B1, B2)": [
                   round(float(np.median(noise[seen[n[0]], k])), 3) for k, n in enumerate(NAMES)],
                                                  "true": [x_noise] + [round(sigma * (2 / tau) ** 0.5, 3)] * 2}}

    both = group == 0  # B masked only where A is left: a B-only trajectory would have no observation at all
    with torch.no_grad():
        dec = {"all": model.decoder(m, batch["mu"][:, None].expand(-1, m.shape[1], -1)).mean.numpy()}
        sub = {k: v[torch.as_tensor(both)] for k, v in batch.items()}
        m_a = posterior_mean(model, sub, tq[torch.as_tensor(both)], drop=columns["B"])
        dec["B masked"] = np.full_like(dec["all"], np.nan)
        dec["B masked"][both] = model.decoder(m_a, sub["mu"][:, None].expand(-1, m.shape[1], -1)).mean.numpy()
    rmse = {}
    for g, gname in enumerate(GROUPS):
        sel = (group == g)[:, None] & inside
        rmse[gname] = {n: round(float(np.sqrt(np.mean((dec["all"][..., k] - truth[..., k])[sel] ** 2))), 3)
                       for k, n in enumerate(("x", "y1", "y2"))}
    sel = both[:, None] & inside
    rmse["both, B masked"] = {n: round(float(np.sqrt(np.mean((dec["B masked"][..., k] - truth[..., k])[sel] ** 2))),
                                       3) for k, n in enumerate(("x", "y1", "y2"))}
    metrics["posterior_rmse"] = rmse

    # the posterior of one trajectory on the fine truth grid, so that its spread between two B observations shows
    b = int(np.argmax(group == 0))
    one = {k: v[b:b + 1] for k, v in batch.items()}
    t_fine = torch.as_tensor(t_true, dtype=batch["t"].dtype)[None]
    band = {"all": decoded_band(model, *posterior_mean(model, one, t_fine, scale=True)),
            "B masked": decoded_band(model, *posterior_mean(model, one, t_fine, drop=columns["B"], scale=True))}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    ax = axes[0, 0]
    ax.plot(t_true, x_true[b], color=INK, lw=1, label="true x")
    ax.plot(a_t[b], a_y[b], "o", color=A_COLOR, ms=6, mec="white", mew=0.8, label="A", zorder=3)
    for key, color, ls, lw, label in (("B masked", MUTED, "--", 1.2, "posterior, B masked"),
                                      ("all", A_COLOR, "-", 1.5, "posterior, A and B")):
        center, spread = band[key][0][0, :, 0], band[key][1][0, :, 0]
        ax.fill_between(t_true, center - 2 * spread, center + 2 * spread, color=color, alpha=0.15, lw=0)
        ax.plot(t_true, center, color=color, lw=lw, ls=ls, label=f"{label} (band: 2 std)")
    ax.set(xlabel="t", ylabel="x", title=f"{args.split} trajectory {b} (both modalities): x")
    ax = axes[0, 1]
    for j, color in ((0, Y1_COLOR), (1, Y2_COLOR)):
        center, spread = band["all"][0][0, :, 1 + j], band["all"][1][0, :, 1 + j]
        ax.plot(t_true, y_true[b, :, j], color=color, lw=0.8, alpha=0.5, label=f"true y{j + 1}")
        ax.fill_between(t_true, center - 2 * spread, center + 2 * spread, color=color, alpha=0.2, lw=0)
        ax.plot(b_t[b], b_y[b][:, j], "o", color=color, ms=3, mec="white", mew=0.4, zorder=3)
        ax.plot(t_true, center, color=color, lw=1.5, label=f"posterior y{j + 1} (band: 2 std)")
    ax.set(xlabel="t", ylabel="y", title="the drivers (dots: B)")
    ax = axes[1, 0]
    with_a = np.flatnonzero(seen["A"])
    pick = np.random.default_rng(0).choice(with_a, min(3000, len(with_a)), replace=False)
    for j, color in ((0, Y1_COLOR), (1, Y2_COLOR)):
        ax.plot(decoded[pick, 1 + j], rate[pick, 0], "o", color=color, ms=2.5, alpha=0.35, mew=0,
                label=f"against decoded y{j + 1}: slope {fmt(coef['A'][2 + j])} (true {true_coef['A'][2 + j]:g})")
    lim = np.abs(decoded[with_a, 1:]).max()
    ax.plot([-lim, lim], [-kappa * lim, kappa * lim], color=INK, lw=1, label=f"truth: dx/dt = {kappa:g} y1")
    ax.set(xlabel="decoded driver", ylabel="learned dx/dt",
           title="learned drift of x, in the units of A (trajectories with A)")
    ax = axes[1, 1]
    labels = [f"d{n}/dt <- {c}" for n in NAMES for c in ("A", "B1", "B2")]
    learned = [coef[n][1 + k] for n in NAMES for k in range(3)]
    true = [true_coef[n][1 + k] for n in NAMES for k in range(3)]
    ypos = np.arange(len(labels))[::-1]
    ax.barh(ypos, np.nan_to_num(learned), color=A_COLOR, height=0.6, label="learned")
    for y_, v in zip(ypos, learned):
        if np.isnan(v):
            ax.text(0, y_, " not separable", va="center", fontsize=8, color=MUTED)
    ax.plot(true, ypos, "|", color=INK, ms=18, mew=2, label="true")
    ax.set_yticks(ypos, labels)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set(xlabel="coefficient", title="learned dynamics: regression of d(decoded)/dt")
    for ax in axes.ravel():
        ax.legend(fontsize=8, frameon=False, loc="lower right" if ax is axes[1, 1] else "best")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    r = rmse
    fig.suptitle(f"{run_name}, {args.split}: x RMSE (both) {r['both']['x']:.2f}, with B masked "
                 f"{r['both, B masked']['x']:.2f}; dx/dt on y1 {fmt(coef['A'][2])}, on y2 {fmt(coef['A'][3])} "
                 f"(R^2 {r2['A']:.2f})",
                 color=INK, fontsize=11)
    out = os.path.join(args.run_dir, f"dynamics_{args.split}{suffix}")
    fig.savefig(out + ".png", dpi=130)
    plt.close(fig)
    json.dump(metrics, open(out + ".json", "w"), indent=1)
    print(json.dumps(metrics, indent=1))
    print(f"Wrote {out}.png and .json")

    # simulations: the first n_sim trajectories with A (with no A, nothing gives the start of x), each simulated
    # once from its own posterior mean m(0), with no observations after it
    pick = np.flatnonzero(group != 2)[:args.n_sim]
    torch.manual_seed(0)
    zs = simulate_paths(model, t_eval, m[torch.as_tensor(pick), 0], args.every)
    with torch.no_grad():
        sims = model.decoder(zs, zs.new_zeros(*zs.shape[:2], 0)).mean.numpy()
    out = os.path.join(args.run_dir, f"simulation_{args.split}{suffix}.png")
    plot_simulation(out, t_true, np.concatenate([x_true[pick, :, None], y_true[pick]], -1), t_eval, sims,
                    f"{run_name}, {args.split}: true trajectories and simulations of the "
                    f"learned SDE (decoder mean, no observation noise)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
