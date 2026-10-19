"""
Generate simple_multimodal_clock: the smallest test that a latent SDE learns dynamics which only the combination
of two modalities shows, with trajectories that do not all have both modalities. The hard version: the phase winds
several turns between A visits, so a model needs an angle latent and a posterior that can follow it
(src/latent/angles.py, src/encoders/integrated.py), and the loss has a local minimum for every wrong number of
turns. test_data/scripts/simple_multimodal_test is the version without winding.

Latent state (phi, gamma): a clock whose rate has a hidden part,

    d phi = (OMEGA + gamma) dt,   gamma a stationary OU process (time scale TAU, std SIGMA).

    A   (cos phi, sin phi) + noise, 2 values, at rare times (Gamma-distributed gaps, mean A_GAP_MEAN)
    B   gamma + noise, 1 value, at frequent times (Gamma-distributed gaps, mean B_GAP_MEAN)

A gives phi only modulo 2 pi, and over one A gap the integral of gamma has a std of about one turn: the wrapped
phase steps of A alone are close to uniform, so A alone shows almost nothing of the drift of phi. B alone shows
the OU dynamics of gamma but nothing of phi. Only both give OMEGA and the coupling "rate = OMEGA + gamma".
Each split mixes three groups: both modalities, A only, B only (--fractions); a trajectory without a modality
has count 0 for it. Long, regular A gaps (shape A_GAP_SHAPE) keep A alone uninformative; short gaps would not.

    python -m test_data.scripts.simple_multimodal_clock.generate [--n-train 100] [--n-val 20] [--n-test 30]

Writes test_data/data/simple_multimodal_clock/:
    <split>.h5      truth/t (nt,), truth/phi, truth/gamma (n_traj, nt): the dense truth, phi unwrapped
                    A/t (n_traj, max_n), A/y (n_traj, max_n, 2), A/count (n_traj,); B/t, B/y (..., 1), B/count:
                    times on truth/t, NaN (t) or 0 (y) after count
                    group (n_traj,): 0 both, 1 A only, 2 B only
    diagnostics.txt what each modality shows (the turns between A visits from: the gap only, B, the true gamma;
                    how uniform the wrapped A steps are), on the train split
    overview.png    one trajectory of each group
"""
import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from test_data.src.multimodal_reaction import integral, ou_path, renewal_times  # generic OU and sampling helpers

DATA = Path(__file__).resolve().parents[2] / "data" / "simple_multimodal_clock"
SPLITS = ("train", "val", "test")
GROUPS = ("both", "A only", "B only")

OMEGA, TAU, SIGMA = 1.0, 5.0, 0.5
A_GAP_MEAN, A_GAP_SHAPE, A_NOISE = 20.0, 8.0, 0.1
B_GAP_MEAN, B_GAP_SHAPE, B_NOISE = 2.0, 4.0, 0.05
T, DT = 200.0, 0.05

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#222220", "#6b6b66", "#e1e0d9"
A_COLOR, B_COLOR = "#2a78d6", "#eb6834"
LABEL_BOX = dict(facecolor=SURFACE, edgecolor="none", pad=3)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "font.size": 10,
})


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def pad(rows, fill, width=None):
    shape = (len(rows), max(1, max(len(r) for r in rows))) + (() if width is None else (width,))
    out = np.full(shape, fill, dtype=float)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def generate_split(split_id: int, n: int, fractions, seed: int):
    t = np.arange(0.0, T + DT / 2, DT)
    rng = np.random.default_rng(np.random.SeedSequence([seed, split_id]))
    counts = np.floor(np.asarray(fractions) / sum(fractions) * n).astype(int)
    counts[0] += n - counts.sum()
    group = rng.permutation(np.repeat(np.arange(3), counts))
    phi, gamma, a_t, a_y, b_t, b_y = [], [], [], [], [], []
    for g, stream in zip(group, np.random.SeedSequence([seed, split_id, 1]).spawn(n)):
        r = np.random.default_rng(stream)
        gm = ou_path(t, r, TAU, SIGMA)
        ph = r.uniform(0, 2 * np.pi) + OMEGA * t + integral(t, gm)
        ka = np.unique(np.searchsorted(t, renewal_times(T, A_GAP_MEAN, A_GAP_SHAPE, r)).clip(0, len(t) - 1))
        kb = np.unique(np.searchsorted(t, renewal_times(T, B_GAP_MEAN, B_GAP_SHAPE, r)).clip(0, len(t) - 1))
        ya = np.stack([np.cos(ph[ka]), np.sin(ph[ka])], -1) + A_NOISE * r.standard_normal((len(ka), 2))
        yb = gm[kb, None] + B_NOISE * r.standard_normal((len(kb), 1))
        if g == 2:
            ka, ya = ka[:0], ya[:0]
        if g == 1:
            kb, yb = kb[:0], yb[:0]
        phi.append(ph)
        gamma.append(gm)
        a_t.append(t[ka])
        a_y.append(ya)
        b_t.append(t[kb])
        b_y.append(yb)
    return t, group, np.array(phi), np.array(gamma), a_t, a_y, b_t, b_y


def diagnostics(t, group, phi, gamma, a_t, a_y, b_t, b_y) -> str:
    """Turns between consecutive A visits, from OMEGA and the gap only, from B in the gap, and from the true gamma;
    and the mean resultant length of the wrapped A steps after removing OMEGA * gap (0: uniform, A alone shows
    nothing of the rate; 1: A alone shows it exactly)."""
    resolved = {"gap length and the true OMEGA (chance)": [], "+ B in the gap": [], "+ true gamma": []}
    steps = []
    for g, ph, gm, ta, ya, tb, yb in zip(group, phi, gamma, a_t, a_y, b_t, b_y):
        if len(ta) < 2:
            continue
        seen = np.arctan2(ya[:, 1], ya[:, 0])
        gap, mod = np.diff(ta), wrap(np.diff(seen))
        true = np.diff(np.interp(ta, t, ph))
        steps.append(mod - OMEGA * gap)
        if g != 0:
            continue
        mean_true = np.array([np.interp(np.linspace(a, b, 400), t, gm).mean() for a, b in zip(ta[:-1], ta[1:])])
        mean_b = np.array([np.interp(np.linspace(a, b, 400), tb, yb[:, 0]).mean() for a, b in zip(ta[:-1], ta[1:])])
        for name, rate in zip(resolved, (0 * gap, mean_b, mean_true)):
            pred = (OMEGA + rate) * gap
            k = np.round((pred - mod) / (2 * np.pi))
            resolved[name] += list(np.abs(mod + 2 * np.pi * k - true) < np.pi)
    steps = np.concatenate(steps)
    lines = [f"train split: {len(group)} trajectories, groups (both, A only, B only) {np.bincount(group, minlength=3)}",
             "", "turns between consecutive A visits, trajectories with both modalities "
             f"({len(resolved['+ true gamma'])} gaps): fraction correct"]
    lines += [f"   {name:<42}{np.mean(v):.3f}" for name, v in resolved.items()]
    lines += ["", f"wrapped A steps minus OMEGA * gap, all trajectories with A ({len(steps)} steps): mean resultant "
                  f"length {np.abs(np.exp(1j * steps).mean()):.3f}",
              "   (0: uniform, A alone shows nothing of the rate; for comparison, 1 / sqrt(n) = "
              f"{1 / np.sqrt(len(steps)):.3f} is the value of a uniform sample of this size)"]
    return "\n".join(lines)


def plot_overview(out: Path, t, group, phi, gamma, a_t, a_y, b_t, b_y):
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), layout="constrained", sharex=True)
    for row, g in enumerate(range(3)):
        i = int(np.argmax(group == g))
        wrapped = np.mod(phi[i], 2 * np.pi)
        wrapped[1:][np.abs(np.diff(wrapped)) > np.pi] = np.nan  # no lines across the jumps from 2 pi to 0
        ax = axes[row, 0]
        ax.plot(t, wrapped, color=INK, lw=0.6, label="phi mod 2 pi, hidden")
        if len(a_t[i]):
            ax.plot(a_t[i], np.mod(np.arctan2(a_y[i][:, 1], a_y[i][:, 0]), 2 * np.pi), "o", color=A_COLOR, ms=6,
                    mec="white", mew=0.8, label="A: angle of the reading")
        else:
            ax.text(0.5, 0.5, "A not observed", transform=ax.transAxes, ha="center", color=MUTED, bbox=LABEL_BOX)
        ax.set_ylabel("phase")
        ax.set_title(f"trajectory {i}, {GROUPS[g]}", loc="left", fontsize=10)
        ax = axes[row, 1]
        ax.plot(t, gamma[i], color=INK, lw=0.8, label="gamma, hidden")
        if len(b_t[i]):
            ax.plot(b_t[i], b_y[i][:, 0], "o", color=B_COLOR, ms=3.5, mec="white", mew=0.5, label="B")
        else:
            ax.text(0.5, 0.9, "B not observed", transform=ax.transAxes, ha="center", color=MUTED, bbox=LABEL_BOX)
        ax.set(ylabel="gamma")
    for ax in axes.ravel():
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2, frameon=False, fontsize=8,
                      ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.suptitle(f"simple_multimodal_clock: d phi = ({OMEGA:g} + gamma) dt, gamma OU (tau {TAU:g}, std {SIGMA:g}); "
                 "A sees phi mod 2 pi rarely, B sees gamma often", x=0.01, ha="left", fontsize=11)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DATA)
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--fractions", type=float, nargs=3, default=(0.4, 0.3, 0.3), metavar=("BOTH", "A", "B"),
                   help="share of the trajectories with both modalities, A only, B only")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split_id, (split, n) in enumerate(zip(SPLITS, (args.n_train, args.n_val, args.n_test))):
        t, group, phi, gamma, a_t, a_y, b_t, b_y = generate_split(split_id, n, args.fractions, args.seed)
        with h5py.File(args.out_dir / f"{split}.h5", "w") as f:
            f["truth/t"], f["truth/phi"], f["truth/gamma"] = t, phi, gamma
            for name, times, ys, width in (("A", a_t, a_y, 2), ("B", b_t, b_y, 1)):
                f[f"{name}/t"] = pad(times, np.nan)
                f[f"{name}/y"] = pad(ys, 0.0, width)
                f[f"{name}/count"] = np.array([len(s) for s in times])
            f["group"] = group
            f.attrs.update(
                dynamics=f"d phi = ({OMEGA} + gamma) dt; gamma stationary OU, time scale {TAU}, std {SIGMA}; "
                         "phi(0) uniform in [0, 2 pi)",
                A=f"(cos phi, sin phi) + N(0, {A_NOISE}^2), Gamma-distributed gaps (shape {A_GAP_SHAPE}, mean "
                  f"{A_GAP_MEAN})",
                B=f"gamma + N(0, {B_NOISE}^2), Gamma-distributed gaps (shape {B_GAP_SHAPE}, mean {B_GAP_MEAN})",
                groups="0 both modalities, 1 A only, 2 B only", fractions=args.fractions, seed=args.seed,
                omega=OMEGA, tau=TAU, sigma=SIGMA,
            )
        print(f"{split}.h5: {n} trajectories, groups (both, A only, B only) {np.bincount(group, minlength=3)}")
        if split == "train":
            text = diagnostics(t, group, phi, gamma, a_t, a_y, b_t, b_y)
            (args.out_dir / "diagnostics.txt").write_text(text + "\n")
            print(text)
            plot_overview(args.out_dir / "overview.png", t, group, phi, gamma, a_t, a_y, b_t, b_y)


if __name__ == "__main__":
    main()
