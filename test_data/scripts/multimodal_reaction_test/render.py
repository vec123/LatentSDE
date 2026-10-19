"""
Render one trajectory of multimodal_reaction_test as an MP4: the forced field u(x, y, t) (truth_frames), and below
it the rotation rate d(phase)/dt over the whole trajectory, drawn up to the current time. The dashed line is the
unforced rate omega(beta), so the distance between the two is the hidden forcing gamma(t). The observations of
observations_<split>.h5: the blue lines are the imaging visits, the orange dots the biomarker samples y of gamma,
drawn at omega + y so that they sit on the rate curve (up to their noise); both solid up to the current time.

    python -m test_data.scripts.multimodal_reaction_test.render [--split train] [--trajectory 0] [--stride 1]

Writes <dir>/<split>_traj<trajectory>.mp4.
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

from test_data.scripts.reaction_diffusion.render_reaction_diffusion import INK, MUTED, field_axes
from test_data.src.multimodal_reaction import truth_frames

DATA = Path(__file__).resolve().parents[2] / "data" / "multimodal_reaction_test"
IMAGING, BIOMARKER, FUTURE, GRID = "#2a78d6", "#eb6834", "#c3c2b7", "#e1e0d9"  # as in sample_modalities.py

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DATA)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--trajectory", type=int, default=0)
    p.add_argument("--stride", type=int, default=1, help="render every k-th stored frame")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or args.dir / f"{args.split}_traj{args.trajectory}.mp4"
    i = args.trajectory

    with h5py.File(args.dir / f"truth_{args.split}.h5", "r") as f, \
            h5py.File(args.dir / f"observations_{args.split}.h5", "r") as g:
        t, x, y = f["t_frames"][:], f["x"][:], f["y"][:]
        u = truth_frames(f, i)
        beta, phase, phase0 = float(f["beta"][i]), f["phase"][i], f["phase0"][i]
        visits = g["imaging/t"][i, :g["imaging/count"][i]]
        n_bio = g["biomarker/count"][i]
        t_bio, y_bio = g["biomarker/t"][i, :n_bio], g["biomarker/y"][i, :n_bio]
    rate = np.gradient(phase, t)
    omega = np.polyfit(t, phase0, 1)[0]

    W, H = 8.0, 8.0
    fig = plt.figure(figsize=(W, H), dpi=160)

    def inches(l, b, w, h):
        return fig.add_axes([l / W, b / H, w / W, h / H])

    left, side, bottom = 0.8, 4.6, 2.75
    ax = inches(left + 0.9, bottom, side, side)
    im = field_axes(ax, u[0], x, y)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    cb = fig.colorbar(im, cax=inches(left + 0.9 + side + 0.15, bottom, 0.15, side), ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("u")
    cb.outline.set_visible(False)

    ax_rate = inches(left, 0.6, W - left - 0.3, 1.4)
    visit_lines = [ax_rate.axvline(s, color=IMAGING, lw=1.2, alpha=0.25) for s in visits]
    ax_rate.plot([], [], color=IMAGING, lw=1.2, label="imaging visit")
    ax_rate.plot(t_bio, omega + y_bio, "o", color=BIOMARKER, ms=3.5, alpha=0.3, mew=0)
    bio_past, = ax_rate.plot([], [], "o", color=BIOMARKER, ms=4.5, mec="white", mew=0.6,
                             label="biomarker sample, at ω + y")
    ax_rate.axhline(omega, color=MUTED, lw=1, ls="--", label=f"unforced rate ω(β) = {omega:.2f}")
    ax_rate.plot(t, rate, color=FUTURE, lw=1)
    past, = ax_rate.plot(t[:1], rate[:1], color=INK, lw=1.4, label="rotation rate d(phase)/dt")
    dot, = ax_rate.plot(t[:1], rate[:1], "o", color=INK, ms=6, mec="white", mew=1)
    ax_rate.set_xlim(t[0], t[-1])
    ax_rate.set_xlabel("t", labelpad=2)
    ax_rate.set_ylabel("rate")
    ax_rate.grid(color=GRID, lw=0.6)
    ax_rate.set_axisbelow(True)
    for side_name in ("top", "right"):
        ax_rate.spines[side_name].set_visible(False)
    ax_rate.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2, frameon=False, fontsize=8, ncol=4,
                   columnspacing=1.2, handlelength=1.6)

    fig.text(left / W, 7.8 / H, f"multimodal_reaction_test  ·  forced u(x, y, t), {args.split} trajectory {i}, "
             f"β = {beta:.2f}", fontsize=11, color=INK, weight="bold")
    label = fig.text(left / W, 7.53 / H, "", fontsize=10, color=MUTED, family="monospace")

    frames = range(0, len(t), args.stride)
    writer = FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    with writer.saving(fig, str(out), dpi=160):
        for n, k in enumerate(frames):
            im.set_data(u[k])
            past.set_data(t[:k + 1], rate[:k + 1])
            dot.set_data([t[k]], [rate[k]])
            seen = t_bio <= t[k]
            bio_past.set_data(t_bio[seen], omega + y_bio[seen])
            for line, s in zip(visit_lines, visits):
                line.set_alpha(0.8 if s <= t[k] else 0.25)
            label.set_text(f"t = {t[k]:7.2f}   rate {rate[k]:+.2f}   γ = rate - ω {rate[k] - omega:+.2f}")
            writer.grab_frame()
            if n % 250 == 0:
                print(f"  frame {n}/{len(frames)}", flush=True)
    plt.close(fig)
    print(f"Wrote {out}  ({len(frames)} frames, {len(frames) / args.fps:.0f}s at {args.fps} fps)")


if __name__ == "__main__":
    main()
