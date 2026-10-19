"""
Fits a PCA (src/data/pca.py) on the u frames of reaction_diffusion_batch.h5, and writes test_data/data/
reaction_diffusion/PCA/: model.pt (its state_dict, so an experiment can load it once instead of refitting it
every run) and reconstructions.png (a few original frames against their PCA reconstruction).

    python -m test_data.scripts.reaction_diffusion.fit_pca
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

from src.data.pca import PCA
from src.data.reaction_diffusion import load_reaction_diffusion

DATA = Path(__file__).resolve().parents[2] / "data" / "reaction_diffusion"

# Same look as render_reaction_diffusion.py.
SURFACE, INK, MUTED = "#fcfcfb", "#222220", "#6b6b66"
CMAP = LinearSegmentedColormap.from_list(
    "blue_red", [(0.0, "#0b3a75"), (0.25, "#2a78d6"), (0.5, "#f0efec"), (0.75, "#e34948"), (1.0, "#7d1716")])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "font.size": 10,
})


def field_axes(ax, data, title=None):
    im = ax.imshow(data, cmap=CMAP, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, color=INK, fontsize=9)
    return im


def plot_reconstructions(out: Path, originals, reconstructions, titles, rank: int, error: float):
    n = len(titles)
    side, gap, left, bottom, row_gap, top = 1.6, 0.25, 0.5, 0.4, 0.35, 0.4
    W = left + n * (side + gap) + 0.7
    H = top + bottom + 2 * side + row_gap + 0.6
    fig = plt.figure(figsize=(W, H))
    fig.suptitle(f"PCA, {rank} dimensions, relative reconstruction RMS error {error:.4f}", color=INK, fontsize=11,
                y=(H - top / 2) / H)
    for i, (orig, recon, title) in enumerate(zip(originals, reconstructions, titles)):
        x0 = (left + i * (side + gap)) / W
        ax_top = fig.add_axes([x0, (bottom + side + row_gap) / H, side / W, side / H])
        ax_bot = fig.add_axes([x0, bottom / H, side / W, side / H])
        field_axes(ax_top, orig, title=title)
        im = field_axes(ax_bot, recon)
        if i == 0:
            ax_top.set_ylabel("original", color=MUTED, fontsize=9)
            ax_bot.set_ylabel("reconstruction", color=MUTED, fontsize=9)
    cax = fig.add_axes([(left + n * (side + gap)) / W, bottom / H, 0.1 / W, (2 * side + row_gap) / H])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1], label="u")
    cb.outline.set_visible(False)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA / "reaction_diffusion_batch.h5")
    p.add_argument("--out-dir", type=Path, default=DATA / "PCA")
    p.add_argument("--field", default="u")
    p.add_argument("--percent-cutoff", type=float, default=0.999, help="0.95 (PCA.create's own default) keeps"
                   " only rank 2 for this data, 16%% relative reconstruction error; 0.999 keeps rank 6, 2%%")
    p.add_argument("--max-evecs", type=int, default=30)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--n-snapshots", type=int, default=6, help="snapshots shown in reconstructions.png")
    args = p.parse_args()

    t, x, _ = load_reaction_diffusion(str(args.data), field=args.field)
    n_traj, n_steps = x.shape[:2]
    frames = x.flatten(0, 1).flatten(1)  # every (trajectory, step) frame, flattened to (n_traj * n_steps, H * W)
    print(f"Fitting PCA on {frames.shape[0]} frames of dimension {frames.shape[1]} from {args.data} ...")

    pca, z = PCA.create(frames, percent_cutoff=args.percent_cutoff, max_evecs=args.max_evecs,
                        random_state=args.random_state)
    rank = z.shape[1]
    recon = pca.decode(z)
    error = (recon - frames).pow(2).mean().sqrt() / frames.pow(2).mean().sqrt()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "model.pt"
    torch.save(pca.state_dict(), model_path)
    print(f"Wrote {model_path}  (rank {rank}, relative reconstruction RMS error {error:.4f})")

    # n_snapshots spread across both trajectory and time, so late-time dynamics show too, not just t ~ 0 or ~ end.
    traj_idx = torch.linspace(0, n_traj - 1, args.n_snapshots).round().long()
    time_idx = torch.linspace(0, n_steps - 1, args.n_snapshots).round().long()
    idx = traj_idx * n_steps + time_idx
    shape = x.shape[-2:]
    originals = frames[idx].reshape(-1, *shape)
    reconstructions = recon[idx].reshape(-1, *shape)
    titles = [f"traj {i // n_steps}, t[{i % n_steps}]" for i in idx.tolist()]
    plot_path = args.out_dir / "reconstructions.png"
    plot_reconstructions(plot_path, originals, reconstructions, titles, rank, error.item())
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
