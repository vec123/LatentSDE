"""
Draw the two modalities of the multimodal_reaction_test toy from its dense truth (generate_truth.py), each at its
own irregular times:

    imaging    the forced field u plus noise, at rare visits (Gamma-distributed gaps, mean about 3 turns): shows
               the rotation phase at each visit, but the phase advance between two visits only modulo 2 pi, and
               nothing of the forcing gamma (it turns the field and leaves its shape alone)
    biomarker  gamma(t) plus noise, a scalar, at frequent times (Gamma-distributed gaps): shows the forcing, and
               nothing of the phase

The integral of gamma over one visit gap has a std of about one turn, so only both together give the phase
advance between visits, and so the latent trajectory. check_leak.py tests what the images show by themselves.

    python -m test_data.scripts.multimodal_reaction_test.sample_modalities [--seed 0]

Writes <dir>/observations_<split>.h5 next to each truth_<split>.h5 (padded to the longest trajectory):
    imaging/t (n_traj, max_n)          visit times on the truth's t_frames, NaN after count
    imaging/frame (n_traj, max_n)      index into t_frames, -1 after count
    imaging/u (n_traj, max_n, n, n)    noisy u, 0 after count
    imaging/count (n_traj,)
    biomarker/t, biomarker/y (n_traj, max_n), biomarker/count (n_traj,): times on the truth's t, NaN after count
    beta (n_traj,)                     the known constant of each trajectory (a model's mu)
and overview.png: trajectory 0 of the train split.
"""
import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from test_data.src.multimodal_reaction import (BIO_GAP_MEAN, BIO_GAP_SHAPE, BIO_NOISE, IMAGE_GAP_MEAN,
                                               IMAGE_GAP_SHAPE, IMAGE_NOISE, renewal_times, truth_frames)

DATA = Path(__file__).resolve().parents[2] / "data" / "multimodal_reaction_test"
SPLITS = ("train", "val", "test")

# Same look as test_data/scripts/reaction_diffusion; IMAGING and BIOMARKER as in the multimodal HS overview.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#222220", "#6b6b66", "#e1e0d9"
IMAGING, BIOMARKER = "#2a78d6", "#eb6834"
CMAP = LinearSegmentedColormap.from_list(
    "blue_red", [(0.0, "#0b3a75"), (0.25, "#2a78d6"), (0.5, "#f0efec"), (0.75, "#e34948"), (1.0, "#7d1716")])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "font.size": 10,
})


def snap(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of the nearest grid point to each time, without repeats."""
    return np.unique(np.clip(np.searchsorted(grid, times), 0, len(grid) - 1))


def pad(rows, fill, dtype):
    out = np.full((len(rows), max(len(r) for r in rows), *rows[0].shape[1:]), fill, dtype=dtype)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def sample_split(truth: Path, out: Path, rng: np.random.Generator, args):
    with h5py.File(truth, "r") as f:
        t, t_frames, gamma, beta = f["t"][:], f["t_frames"][:], f["gamma"][:], f["beta"][:]
        n_traj = gamma.shape[0]
        img_frame, img_u, bio_index, bio_y = [], [], [], []
        for i in range(n_traj):
            k = snap(renewal_times(t_frames[-1], args.image_gap_mean, args.image_gap_shape, rng), t_frames)
            u = truth_frames(f, i, k)
            img_frame.append(k)
            img_u.append((u + args.image_noise * rng.standard_normal(u.shape)).astype(np.float32))
            j = snap(renewal_times(t[-1], args.bio_gap_mean, args.bio_gap_shape, rng), t)
            bio_index.append(j)
            bio_y.append(gamma[i, j] + args.bio_noise * rng.standard_normal(len(j)))
    with h5py.File(out, "w") as g:
        img = g.create_group("imaging")
        img["t"] = pad([t_frames[k] for k in img_frame], np.nan, "f8")
        img["frame"] = pad(img_frame, -1, "i8")
        img["u"] = pad(img_u, 0.0, "f4")
        img["count"] = np.array([len(k) for k in img_frame])
        bio = g.create_group("biomarker")
        bio["t"] = pad([t[j] for j in bio_index], np.nan, "f8")
        bio["y"] = pad(bio_y, np.nan, "f8")
        bio["count"] = np.array([len(j) for j in bio_index])
        g["beta"] = beta
        g.attrs.update(
            truth=truth.name,
            imaging=f"forced u + N(0, {args.image_noise}^2) per pixel, at Gamma-distributed gaps (shape "
                    f"{args.image_gap_shape}, mean {args.image_gap_mean}), snapped to t_frames",
            biomarker=f"gamma + N(0, {args.bio_noise}^2), at Gamma-distributed gaps (shape {args.bio_gap_shape}, "
                      f"mean {args.bio_gap_mean}), snapped to t",
            seed=args.seed,
        )
    counts = [len(k) for k in img_frame], [len(j) for j in bio_index]
    print(f"{out.name}: {n_traj} trajectories, images per trajectory {min(counts[0])}-{max(counts[0])}, "
          f"biomarker samples {min(counts[1])}-{max(counts[1])}")


def plot_overview(truth: Path, obs: Path, out: Path, n_images: int = 6):
    with h5py.File(truth, "r") as f, h5py.File(obs, "r") as g:
        t, t_frames, gamma, phase = f["t"][:], f["t_frames"][:], f["gamma"][0], f["phase"][0]
        beta = f["beta"][0]
        n_img = g["imaging/count"][0]
        t_img, u_img = g["imaging/t"][0, :n_img], g["imaging/u"][0, :min(n_img, n_images)]
        n_bio = g["biomarker/count"][0]
        t_bio, y_bio = g["biomarker/t"][0, :n_bio], g["biomarker/y"][0, :n_bio]
    rate = np.gradient(phase, t_frames)

    fig = plt.figure(figsize=(12, 7.6), layout="constrained")
    grid = fig.add_gridspec(3, len(u_img), height_ratios=(1, 1, 1.15))
    ax_gamma = fig.add_subplot(grid[0, :])
    ax_rate = fig.add_subplot(grid[1, :], sharex=ax_gamma)
    ax_gamma.plot(t, gamma, color=INK, lw=1, label="gamma(t), hidden")
    ax_gamma.plot(t_bio, y_bio, "o", color=BIOMARKER, ms=4, mec=SURFACE, mew=0.6, label="biomarker")
    ax_rate.plot(t_frames, rate, color=INK, lw=1, label="rotation rate d(phase)/dt")
    for ax in (ax_gamma, ax_rate):
        for s in t_img:
            ax.axvline(s, color=IMAGING, lw=1.5, alpha=0.7)
        ax.plot([], [], color=IMAGING, lw=1.5, label="imaging visit")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2, frameon=False, fontsize=9, ncol=3)
    ax_gamma.set_ylabel("gamma")
    fig.suptitle(f"multimodal_reaction_test, train trajectory 0 (beta = {beta:.2f}): hidden phase-rate forcing, its "
                 "biomarker, and the imaging visits", x=0.01, ha="left", fontsize=11)
    ax_rate.set_ylabel("rate")
    ax_rate.set_xlabel("t")
    for c, (u, s) in enumerate(zip(u_img, t_img)):
        ax = fig.add_subplot(grid[2, c])
        ax.imshow(u, cmap=CMAP, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"visit t = {s:.1f}", fontsize=9, color=INK)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DATA, help="folder with truth_<split>.h5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-gap-mean", type=float, default=IMAGE_GAP_MEAN)
    p.add_argument("--image-gap-shape", type=float, default=IMAGE_GAP_SHAPE)
    p.add_argument("--image-noise", type=float, default=IMAGE_NOISE)
    p.add_argument("--bio-gap-mean", type=float, default=BIO_GAP_MEAN)
    p.add_argument("--bio-gap-shape", type=float, default=BIO_GAP_SHAPE)
    p.add_argument("--bio-noise", type=float, default=BIO_NOISE)
    args = p.parse_args()

    for split_id, split in enumerate(SPLITS):
        rng = np.random.default_rng([args.seed, split_id])
        sample_split(args.dir / f"truth_{split}.h5", args.dir / f"observations_{split}.h5", rng, args)
    plot_overview(args.dir / "truth_train.h5", args.dir / "observations_train.h5", args.dir / "overview.png")


if __name__ == "__main__":
    main()
