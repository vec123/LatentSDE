"""Render the hotel dataset as an MP4 (floor-by-floor room temperatures) plus a static overview figure."""

import argparse
from pathlib import Path

import h5py
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

DATA = Path(__file__).resolve().parents[2] / "data" / "hotel"
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

SURFACE = "#fcfcfb"
INK = "#222220"
MUTED = "#6b6b66"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
CMAP = LinearSegmentedColormap.from_list(
    "blue_red", [(0.0, "#0b3a75"), (0.25, "#2a78d6"), (0.5, "#f0efec"), (0.75, "#e34948"), (1.0, "#7d1716")]
)
CMAP.set_bad(SURFACE)
SPLITS = [("train", "#ecebe8"), ("val", "#dcdbd7"), ("test", "#c9c8c3")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "font.size": 10,
})


def clean(ax, keep=()):
    for name, s in ax.spines.items():
        s.set_visible(name in keep)


def shade_splits(ax, t_days, split, label=True):
    for code, (name, color) in enumerate(SPLITS):
        sel = t_days[split == code]
        if sel.size:
            ax.axvspan(sel[0], sel[-1], color=color, lw=0, zorder=0)
            if label:
                ax.text((sel[0] + sel[-1]) / 2, 0.97, name, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=8, color=MUTED)


def overview(f, out, days=7.0):
    t = f["t"][:] / 24
    sel = t < days
    t = t[sel]
    S, sp, ci = f["sensor_temp"][sel], f["setpoint"][sel], f["checked_in"][sel]
    duty, power = f["heater_duty"][sel], f["rooms/heater_power"][:]
    side, number = f["rooms/side"][:], f["rooms/number"][:]

    heated = (ci & ~np.isnan(sp)).mean(0)
    off = (ci & np.isnan(sp)).sum(0)
    picks = [(int(np.argmax(np.where(side == 1, heated, -1))), "south, guest-heated"),
             (int(np.argmax(np.where(side == 0, heated, -1))), "north, guest-heated")]
    if off.max() > 0:
        picks.append((int(np.argmax(off)), "guest switched heating off"))

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, height_ratios=[1, 2, 1], constrained_layout=True)
    ax = axes[0]
    ax.plot(t, f["t_out"][sel], color=INK, lw=1.5)
    ax.set_ylabel("outdoor (°C)")
    ax.set_title(f"Hotel simulation · first {days:g} days", loc="left", color=INK, weight="bold")

    ax = axes[1]
    ends = []
    for (i, desc), color in zip(picks, SERIES):
        ax.plot(t, S[:, i], color=color, lw=1.2, label=f"room {number[i]} ({desc})")
        ends.append([S[-12:, i].mean(), number[i]])
    ends.sort()
    min_gap = 0.06 * np.ptp(S[:, [i for i, _ in picks]])
    for k in range(1, len(ends)):  # nudge end labels apart
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + min_gap)
    for y, num in ends:
        ax.annotate(f"{num}", (t[-1], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_ylabel("sensor reading (°C)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9, ncol=3, borderaxespad=0.2)

    ax = axes[2]
    per_hour = round(1 / (24 * (t[1] - t[0])))
    hours = len(t) // per_hour
    kw = (duty * power).sum(1)[:hours * per_hour].reshape(hours, per_hour).mean(1) / 1000
    ax.fill_between(t[:hours * per_hour:per_hour], kw, color="#c4c3be", lw=0, step="post")
    ax.set_ylabel("room heating (kW,\nhourly mean)")
    ax.set_xlabel("time (days)")
    for ax in axes:
        clean(ax, keep=("bottom",))
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA / "hotel.h5")
    p.add_argument("--stride", type=int, default=3, help="render every k-th sample")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--vmin", type=float, default=10.0)
    p.add_argument("--vcenter", type=float, default=18.0)
    p.add_argument("--vmax", type=float, default=26.0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or args.data.with_suffix(".mp4")

    with h5py.File(args.data, "r") as f:
        ov = out.with_name(f"{out.stem}_overview.png")
        overview(f, ov)
        print(f"Wrote {ov}")
        t = f["t"][:]
        S, duty, ci, occ = f["sensor_temp"][:], f["heater_duty"][:], f["checked_in"][:], f["occupants"][:]
        corridor, t_out, split = f["corridor_temp"][:], f["t_out"][:], f["split"][:]
        floor, side, pos = f["rooms/floor"][:], f["rooms/side"][:], f["rooms/position"][:]
        F, n_cols = int(f.attrs["n_floors"]), int(f.attrs["n_cols"])
    M = S.shape[1]
    norm = TwoSlopeNorm(vcenter=args.vcenter, vmin=args.vmin, vmax=args.vmax)

    # Figure laid out in inches: [north rooms | corridor | south rooms | colorbar], timeline below.
    W, H = 10.0, 6.4
    fig = plt.figure(figsize=(W, H), dpi=120)

    def inches(l, b, w, h):
        return fig.add_axes([l / W, b / H, w / W, h / H])

    left, bottom, gap, corr_w, plan_h = 0.75, 2.05, 0.3, 0.45, 3.3
    side_w = (W - left - 1.05 - 2 * gap - corr_w) / 2
    panels, images = {}, {}
    x0 = left
    for key, width, title in [("north", side_w, "North rooms"), ("corridor", corr_w, "Corridor"), ("south", side_w, "South rooms")]:
        ax = inches(x0, bottom, width, plan_h)
        cols = 1 if key == "corridor" else n_cols
        images[key] = ax.imshow(np.full((F, cols), np.nan), cmap=CMAP, norm=norm, origin="lower", aspect="auto",
                                extent=[-0.5, cols - 0.5, -0.5, F - 0.5])
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        n_rooms = 1 if key == "corridor" else int(pos[side == (key == "south")].max()) + 1
        ax.set_xticks(range(n_rooms), [str(c + 1) for c in range(n_rooms)] if key != "corridor" else [""])
        ax.set_yticks(range(F), [str(f + 1) for f in range(F)] if key == "north" else [""] * F)
        ax.tick_params(length=0)
        for c in range(1, cols):
            ax.axvline(c - 0.5, color=SURFACE, lw=2)
        for r in range(1, F):
            ax.axhline(r - 0.5, color=SURFACE, lw=2)
        clean(ax)
        panels[key] = ax
        x0 += width + gap
    panels["north"].set_ylabel("floor")
    panels["north"].set_xlabel("room position along corridor")
    cax = inches(x0 - gap + 0.2, bottom, 0.15, plan_h)
    cb = fig.colorbar(images["north"], cax=cax)
    cb.set_label("temperature (°C)")
    cb.outline.set_visible(False)

    # Per-room glyphs: heater gauge along the bottom of the cell, occupancy dot in the top-right corner.
    gauges = []
    for i in range(M):
        ax = panels["north" if side[i] == 0 else "south"]
        x, y = pos[i] - 0.38, floor[i] - 0.38
        ax.add_patch(Rectangle((x, y), 0.76, 0.1, facecolor="white", edgecolor=INK, lw=0.6, zorder=3))
        gauges.append(ax.add_patch(Rectangle((x, y), 0.0, 0.1, facecolor=INK, lw=0, zorder=4)))
    dots = {}
    for key in ("north", "south"):
        dots[key, "in"] = panels[key].scatter([], [], s=34, facecolor=INK, edgecolor="white", lw=1.2, zorder=5)
        dots[key, "out"] = panels[key].scatter([], [], s=34, facecolor="white", edgecolor=INK, lw=1.2, zorder=5)
    fig.legend(handles=[Patch(facecolor=INK, edgecolor=INK, label="heater on (bar = share of time)"),
                        Line2D([], [], ls="", marker="o", ms=6, mfc=INK, mec="white", label="guest in room"),
                        Line2D([], [], ls="", marker="o", ms=6, mfc="white", mec=INK, label="checked in, out"),
                        Patch(facecolor="none", edgecolor="none", label="no dot = vacant")],
               loc="lower left", bbox_to_anchor=(left / W, 1.3 / H), ncol=4, frameon=False, fontsize=8.5,
               handlelength=1.4, columnspacing=1.6)

    # Outdoor temperature timeline.
    tl = inches(left, 0.45, W - left - 1.05, 0.75)
    td = t / 24
    shade_splits(tl, td, split)
    tl.plot(td, t_out, color=INK, lw=0.8)
    tl.set_xlim(td[0], td[-1])
    tl.set_ylabel("outdoor °C", fontsize=8)
    tl.set_xlabel("time (days)", labelpad=2)
    tl.tick_params(labelsize=8, length=0)
    clean(tl)
    cursor = tl.axvline(td[0], color=INK, lw=2)

    fig.text(left / W, 6.08 / H, f"Hotel · {M} rooms on {F} floors · room sensor readings (corridor: simulated air)",
             fontsize=12, color=INK, weight="bold")
    label = fig.text(left / W, 5.8 / H, "", fontsize=10, color=MUTED, family="monospace")

    north, south = side == 0, side == 1
    grid = {"north": np.full((F, n_cols), np.nan), "south": np.full((F, n_cols), np.nan)}
    frames = range(0, len(t), args.stride)
    writer = FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    with writer.saving(fig, str(out), dpi=120):
        for n, k in enumerate(frames):
            for key, mask in (("north", north), ("south", south)):
                grid[key][floor[mask], pos[mask]] = S[k, mask]
                images[key].set_data(grid[key])
                for state, sel in (("in", ci[k] & (occ[k] > 0)), ("out", ci[k] & (occ[k] == 0))):
                    sel = sel & mask
                    dots[key, state].set_offsets(np.c_[pos[sel] + 0.3, floor[sel] + 0.28])
            images["corridor"].set_data(corridor[k][:, None])
            d = duty[k:k + args.stride].mean(0)
            for g, di in zip(gauges, d):
                g.set_width(0.76 * di)
            cursor.set_xdata([td[k], td[k]])
            hh, mm = divmod(round((t[k] % 24) * 60), 60)
            label.set_text(f"day {int(td[k]):2d}  {hh:02d}:{mm:02d}   outdoor {t_out[k]:5.1f} °C   "
                           f"heaters on {int((d > 0).sum()):2d}/{M}   guests in {int((ci[k] & (occ[k] > 0)).sum()):2d}/{M}   "
                           f"{SPLITS[split[k]][0]}")
            writer.grab_frame()
            if n % 500 == 0:
                print(f"  frame {n}/{len(frames)}")
    plt.close(fig)
    print(f"Wrote {out}  ({len(frames)} frames, {len(frames) / args.fps:.0f}s at {args.fps} fps)")


if __name__ == "__main__":
    main()
