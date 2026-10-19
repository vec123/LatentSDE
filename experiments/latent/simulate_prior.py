"""
Simulates a trajectory from the trained LatentSDE, Euler-Maruyama forward through the learned drift and
diffusion only -- no encoder or posterior feeding it observations as it goes, unlike reconstruct() in run.py,
which decodes the posterior mean at the data's own times.

z0 is (--init posterior, the default) the window's own posterior mean m(t0) from its real frame 0, so the
simulation starts exactly where the compared trajectory does and only the roll-forward is prior (no
observations after t0); or (--init prior) a draw from the learned N(z0_mean, exp(z0_logvar)), the same
unconditional start regardless of which trajectory it is compared against.

    python -m experiments.latent.simulate_prior runs/<run_dir> [--split test] [--window 0] [--substeps 10]
                                                [--init posterior|prior]

Writes <run_dir>/prior_simulation_<split>.png: the real window side by side with one simulated sample, decoded.
"""
import argparse
import os

import torch

from src.decoders.field import PCADecoder
from src.latent.diffusion import diffuse
from src.plotting import plot_learned_dynamics
from .run import CMAP, INK, MUTED, field_axes, load_config, load_split


def encode_z0(model, t: torch.Tensor, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Posterior mean m(t0) (1, dim_z) from the window's own real frame 0 -- the true encoding, a point estimate,
    not a sample."""
    model.eval()
    t, x = t.unsqueeze(0), x.unsqueeze(0)
    mask = torch.ones_like(x)
    with torch.no_grad():
        feat = x if model.encoder is None else model.encoder(x)
        feat_mask = mask.any(-1, keepdim=True).expand_as(feat).to(feat.dtype)
        model.posterior.condition({"t": t, "x": feat, "mask": feat_mask, "mu": mu})
        m, _, _, _ = model.posterior(t[:, :1])
    return m[:, 0]


def simulate(model, t: torch.Tensor, mu: torch.Tensor, substeps: int, z0: torch.Tensor = None) -> torch.Tensor:
    """z(t) (N, dim_z), Euler-Maruyama from z0 (the prior's own draw if None) over the fine grid between t's
    points, one sample."""
    model.eval()
    u = mu.new_zeros(1, 0)  # run.py's build_model always gives the drift dim_u=0, no forcing
    with torch.no_grad():
        if z0 is None:
            z0 = (model.z0_mean + (0.5 * model.z0_logvar).exp() * torch.randn_like(model.z0_mean)).unsqueeze(0)
        z = z0
        zs = [z]
        for t0, t1 in zip(t[:-1].tolist(), t[1:].tolist()):
            dt = (t1 - t0) / substeps
            for _ in range(substeps):
                G = model.diffusion(z, mu, torch.tensor(t0))
                z = z + model.drift(z, mu, u) * dt + diffuse(G, (dt ** 0.5) * torch.randn_like(z))
            zs.append(z)
        return torch.cat(zs, 0)


def plot_comparison(out: str, original, simulated, titles, shape, start_label: str):
    import matplotlib.pyplot as plt
    n = len(titles)
    side, gap, left, bottom, row_gap, top = 1.6, 0.25, 0.5, 0.4, 0.35, 0.4
    W, H = left + n * (side + gap) + 0.7, top + bottom + 2 * side + row_gap + 0.6
    fig = plt.figure(figsize=(W, H))
    fig.suptitle(f"prior simulation: learned drift and diffusion, no observations ({start_label})", color=INK,
                fontsize=11, y=(H - top / 2) / H, wrap=True)
    for i, (orig, sim, title) in enumerate(zip(original, simulated, titles)):
        x0 = (left + i * (side + gap)) / W
        ax_top = fig.add_axes([x0, (bottom + side + row_gap) / H, side / W, side / H])
        ax_bot = fig.add_axes([x0, bottom / H, side / W, side / H])
        field_axes(ax_top, orig.reshape(shape), title=title)
        im = field_axes(ax_bot, sim.reshape(shape))
        if i == 0:
            ax_top.set_ylabel("original", color=MUTED, fontsize=9)
            ax_bot.set_ylabel("prior sample", color=MUTED, fontsize=9)
    cax = fig.add_axes([(left + n * (side + gap)) / W, bottom / H, 0.1 / W, (2 * side + row_gap) / H])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.outline.set_visible(False)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--split", default="test")
    parser.add_argument("--window", type=int, default=0, help="index of the window to simulate (and compare to)")
    parser.add_argument("--substeps", type=int, default=10, help="Euler-Maruyama steps between data points")
    parser.add_argument("--snapshots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", choices=["posterior", "prior"], default="posterior",
                        help="posterior: the window's own true frame-0 encoding, m(t0); prior: z0 ~ "
                             "N(z0_mean, exp(z0_logvar))")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = load_config(os.path.join(args.run_dir, "config.yaml"))
    model = torch.load(os.path.join(args.run_dir, "model.pt"), weights_only=False)
    batches, shape = load_split(config["data"][args.split])
    t = torch.cat([b["t"] for b in batches])[args.window]
    x = torch.cat([b["x"] for b in batches])[args.window]
    mu = torch.cat([b["mu"] for b in batches])[args.window:args.window + 1]

    z0 = encode_z0(model, t, x, mu) if args.init == "posterior" else None
    z = simulate(model, t, mu, args.substeps, z0=z0)
    mu_t = mu.expand(z.shape[0], -1)
    with torch.no_grad():
        sim = model.decoder(z, mu_t).mean

    start_label = "start: true frame-0 encoding" if args.init == "posterior" else "start: prior sample"
    if mu.numel():
        start_label += f", mu={[round(v, 3) for v in mu[0].tolist()]}"
    idx = torch.linspace(0, len(t) - 1, min(args.snapshots, len(t))).round().long()
    out = os.path.join(args.run_dir, f"prior_simulation_{args.split}_w{args.window}.png")
    plot_comparison(out, x[idx], sim[idx], [f"{args.split} [{i}]" for i in idx.tolist()], shape, start_label)
    print(out)

    if isinstance(model.decoder, PCADecoder):
        with torch.no_grad():
            true_pca = model.encoder.pca.encode(x)
            sim_pca = model.decoder.to_code(torch.cat([z, mu_t], -1))
        pca_out = os.path.join(args.run_dir, f"prior_simulation_pca_{args.split}_w{args.window}.png")
        names = [f"pca{i + 1}" for i in range(sim_pca.shape[-1])]
        plot_learned_dynamics(pca_out, t, sim_pca, (t, true_pca), names=names,
                              title=f"prior-simulated PCA components ({start_label}) against the true "
                                    "components", phase=(0, 1))
        print(pca_out)


if __name__ == "__main__":
    main()
