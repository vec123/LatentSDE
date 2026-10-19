"""Render a reaction-diffusion trajectory as an MP4 (one 2-D frame per snapshot) plus a snapshot strip.

Works on the single-trajectory file (reaction_diffusion.h5, with a train/val/test split and no --trajectory
needed) and on a multi-trajectory batch file (reaction_diffusion_batch.h5 and the like: pass --trajectory to
pick one; if the file has a "beta" dataset, that trajectory's value is shown in the title, e.g. to compare two
--trajectory renders at different beta side by side).
"""

import argparse
from pathlib import Path

import h5py
import imageio_ffmpeg
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap

DATA = Path(__file__).resolve().parents[2] / "data" / "reaction_diffusion"

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

SURFACE = "#fcfcfb"
INK = "#222220"
MUTED = "#6b6b66"
# Diverging blue <-> red with a neutral gray midpoint: u is signed, 0 means "no excitation".
CMAP = LinearSegmentedColormap.from_list(
    "blue_red", [(0.0, "#0b3a75"), (0.25, "#2a78d6"), (0.5, "#f0efec"), (0.75, "#e34948"), (1.0, "#7d1716")]
)
SPLITS = [("train", "#dcdbd7"), ("val", "#c4c3be"), ("test", "#acaba6")]
RUN = ("run", "#dcdbd7")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "font.size": 10,
})


def field_axes(ax, data, x, y, title=None):
    dx = x[1] - x[0]
    im = ax.imshow(data, cmap=CMAP, vmin=-1, vmax=1, origin="lower", interpolation="nearest",
                   extent=[x[0], x[-1] + dx, y[0], y[-1] + dx])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, color=INK, fontsize=10)
    return im


def timeline_axes(ax, T, train_end=None, val_end=None):
    """One labelled segment per split, or one plain, unlabelled bar if the file has no train/val/test split."""
    if train_end is not None and val_end is not None:
        bounds = [0.0, train_end, val_end, T]
        segments = list(zip(SPLITS, bounds[:-1], bounds[1:]))
    else:
        segments = [(RUN, 0.0, T)]
    for (name, color), a, b in segments:
        ax.barh(0, b - a, left=a, height=1, color=color, edgecolor=SURFACE, linewidth=2)
        if len(segments) > 1:
            ax.text((a + b) / 2, 0, name, ha="center", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks(sorted({0.0, train_end, val_end, T} - {None}) if train_end is not None
                 else np.linspace(0, T, 5).round(1))
    ax.set_yticks([])
    ax.tick_params(labelsize=8, length=0)
    ax.set_xlabel("t", labelpad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax.axvline(0, color=INK, linewidth=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA / "reaction_diffusion.h5")
    p.add_argument("--field", choices=["u", "v"], default="u")
    p.add_argument("--trajectory", type=int, default=None, help="index into a multi-trajectory file's leading"
                   " axis; omit for the single-trajectory file")
    p.add_argument("--stride", type=int, default=6, help="render every k-th snapshot")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    suffix = f"{args.field}" if args.trajectory is None else f"traj{args.trajectory}_{args.field}"
    out = args.out or args.data.with_name(f"{args.data.stem}_{suffix}.mp4")

    with h5py.File(args.data, "r") as f:
        t, x, y = f["t"][:], f["x"][:], f["y"][:]
        field = f[args.field][:]
        beta = float(f["beta"][args.trajectory]) if "beta" in f and args.trajectory is not None else None
        train_end, val_end = f.attrs.get("train_end"), f.attrs.get("val_end")
    if args.trajectory is not None:
        field = field[args.trajectory]
    T = t[-1]
    if train_end is not None and val_end is not None:
        split = ["train" if ti <= train_end else "val" if ti <= val_end else "test" for ti in t]
    else:
        split = ["run"] * len(t)
    beta_suffix = "" if beta is None else f", beta = {beta:.2f}"

    # Snapshot strip: six times spread evenly over the trajectory's own span.
    picks = np.linspace(t[0], t[-1], 6)
    side, gap, left, bottom = 1.8, 0.3, 0.6, 0.55
    W, H = left + len(picks) * (side + gap) + 0.75, bottom + side + 0.4
    fig = plt.figure(figsize=(W, H))
    for i, tp in enumerate(picks):
        ax = fig.add_axes([(left + i * (side + gap)) / W, bottom / H, side / W, side / H])
        k = int(abs(t - tp).argmin())
        im = field_axes(ax, field[k], x, y, title=f"t = {t[k]:.1f}  ({split[k]})")
        ax.set_xticks([-10, 0, 10])
        ax.set_yticks([-10, 0, 10])
        if i > 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    cax = fig.add_axes([(left + len(picks) * (side + gap)) / W, bottom / H, 0.12 / W, side / H])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1], label=args.field)
    cb.outline.set_visible(False)
    fig.suptitle(f"λ–ω reaction–diffusion{beta_suffix}", color=INK, fontsize=11)
    strip = out.with_name(f"{out.stem}_snapshots.png")
    fig.savefig(strip, dpi=150)
    plt.close(fig)
    print(f"Wrote {strip}")

    # Video: axes placed in inches so the field is square and the colorbar matches its height.
    W, H = 6.0, 7.0
    fig = plt.figure(figsize=(W, H), dpi=160)
    left, side, bottom = 0.7, 4.3, 1.6

    def inches(l, b, w, h):
        return fig.add_axes([l / W, b / H, w / W, h / H])

    ax = inches(left, bottom, side, side)
    im = field_axes(ax, field[0], x, y)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    cb = fig.colorbar(im, cax=inches(left + side + 0.15, bottom, 0.15, side), ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label(args.field)
    cb.outline.set_visible(False)
    cursor = timeline_axes(inches(left, 0.5, side + 0.3, 0.3), T, train_end, val_end)
    fig.text(left / W, 6.72 / H, f"λ–ω reaction–diffusion  ·  {args.field}(x, y, t){beta_suffix}", fontsize=12,
            color=INK, weight="bold")
    label = fig.text(left / W, 6.42 / H, "", fontsize=10, color=MUTED, family="monospace")

    frames = range(0, len(t), args.stride)
    writer = FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    with writer.saving(fig, str(out), dpi=160):
        for i, k in enumerate(frames):
            im.set_data(field[k])
            cursor.set_xdata([t[k], t[k]])
            label.set_text(f"t = {t[k]:7.2f}   snapshot {k:5d}/{len(t) - 1}   {split[k]}")
            writer.grab_frame()
            if i % 250 == 0:
                print(f"  frame {i}/{len(frames)}")
    plt.close(fig)
    print(f"Wrote {out}  ({len(frames)} frames, {len(frames) / args.fps:.0f}s at {args.fps} fps)")


if __name__ == "__main__":
    main()
