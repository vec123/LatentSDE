"""
Generate simple_multimodal_test: a small test that a latent SDE learns dynamics which only the combination of two
modalities shows, with trajectories that do not all have both modalities, and with smooth latent paths (no
winding, unlike test_data/scripts/simple_multimodal_clock), so that a posterior which interpolates between the
observations (SequencePosterior) can follow them.

Latent state (x, y1, y2): x is driven by y1 only; y1 and y2 are independent OU processes with the same law,

    dx = KAPPA y1 dt + X_NOISE dW_0,   dy_j = -y_j / TAU dt + SIGMA sqrt(2 / TAU) dW_j   (j = 1, 2),
    x(0) ~ N(0, X0_STD^2).

X_NOISE (--x-noise) is 0 by default: x is then the integral of y1, with no noise of its own, the degenerate case
that a latent SDE with diagonal noise finds hard. Its noise comes from a separate random stream, so with
--x-noise > 0 only x changes: the paths of y, the observation times and the observation noise are the same.

    A   x + noise, 1 value, at rare times (Gamma-distributed gaps, mean A_GAP_MEAN)
    B   (y1, y2) + noise, 2 values, at frequent times (Gamma-distributed gaps, mean B_GAP_MEAN)

A alone shows how much x moves, but not which driver moves it, nor with which sign; B alone shows two OU processes
with the same law, and nothing of x. The data of the A-only and the B-only trajectories have exactly the same law
when y1 and y2 are swapped (or y1 changes sign), so "x is driven by +KAPPA y1, not by y2" can only be learned from
trajectories that have both modalities; and x between two A visits needs the integral of y1, from B. Each split
mixes three groups: both modalities, A only, B only (--fractions); a trajectory without a modality has count 0 for
it.

    python -m test_data.scripts.simple_multimodal_test.generate [--n-train 100] [--n-val 20] [--n-test 30]
        [--x-noise 0] [--out-dir test_data/data/simple_multimodal_test]

Writes test_data/data/simple_multimodal_test/:
    <split>.h5      truth/t (nt,), truth/x (n_traj, nt), truth/y (n_traj, nt, 2): the dense truth
                    A/t (n_traj, max_n), A/y (n_traj, max_n, 1), A/count (n_traj,); B/t, B/y (..., 2), B/count:
                    times on truth/t, NaN (t) or 0 (y) after count
                    group (n_traj,): 0 both, 1 A only, 2 B only
    diagnostics.txt what each modality shows, on the train split
    overview.png    one trajectory of each group
"""
import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from test_data.src.multimodal_reaction import ou_path, renewal_times  # generic OU and sampling helpers

DATA = Path(__file__).resolve().parents[2] / "data" / "simple_multimodal_test"
SPLITS = ("train", "val", "test")
GROUPS = ("both", "A only", "B only")

KAPPA, TAU, SIGMA, X0_STD = 1.0, 5.0, 0.3, 1.0
A_GAP_MEAN, A_GAP_SHAPE, A_NOISE = 10.0, 8.0, 0.1
B_GAP_MEAN, B_GAP_SHAPE, B_NOISE = 1.5, 4.0, 0.05
T, DT = 100.0, 0.05

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#222220", "#6b6b66", "#e1e0d9"
A_COLOR, Y1_COLOR, Y2_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
LABEL_BOX = dict(facecolor=SURFACE, edgecolor="none", pad=3)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "font.size": 10,
})


def pad(rows, fill, width):
    out = np.full((len(rows), max(1, max(len(r) for r in rows)), width), fill, dtype=float)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def generate_split(split_id: int, n: int, fractions, seed: int, x_noise: float = 0.0, a_gap_mean: float = A_GAP_MEAN):
    t = np.arange(0.0, T + DT / 2, DT)
    rng = np.random.default_rng(np.random.SeedSequence([seed, split_id]))
    counts = np.floor(np.asarray(fractions) / sum(fractions) * n).astype(int)
    counts[0] += n - counts.sum()
    group = rng.permutation(np.repeat(np.arange(3), counts))
    xs, ys, a_t, a_y, b_t, b_y = [], [], [], [], [], []
    for g, stream in zip(group, np.random.SeedSequence([seed, split_id, 1]).spawn(n)):
        r = np.random.default_rng(stream)
        y = np.stack([ou_path(t, r, TAU, SIGMA), ou_path(t, r, TAU, SIGMA)], -1)
        steps = 0.5 * (y[1:, 0] + y[:-1, 0]) * np.diff(t)  # exact integral of y1, linear between the times t
        x = X0_STD * r.standard_normal() + KAPPA * np.concatenate([[0.0], np.cumsum(steps)])
        if x_noise > 0:
            dw = np.sqrt(np.diff(t)) * np.random.default_rng(stream.spawn(1)[0]).standard_normal(len(t) - 1)
            x = x + x_noise * np.concatenate([[0.0], np.cumsum(dw)])
        ka = np.unique(np.searchsorted(t, renewal_times(T, a_gap_mean, A_GAP_SHAPE, r)).clip(0, len(t) - 1))
        kb = np.unique(np.searchsorted(t, renewal_times(T, B_GAP_MEAN, B_GAP_SHAPE, r)).clip(0, len(t) - 1))
        ya = x[ka, None] + A_NOISE * r.standard_normal((len(ka), 1))
        yb = y[kb] + B_NOISE * r.standard_normal((len(kb), 2))
        if g == 2:
            ka, ya = ka[:0], ya[:0]
        if g == 1:
            kb, yb = kb[:0], yb[:0]
        xs.append(x)
        ys.append(y)
        a_t.append(t[ka])
        a_y.append(ya)
        b_t.append(t[kb])
        b_y.append(yb)
    return t, group, np.array(xs), np.array(ys), a_t, a_y, b_t, b_y


def bridge(t_gap, xa, xb, rate):
    """x between two A readings xa, xb: the integral of KAPPA * rate from the left reading, with the mismatch at
    the right reading spread linearly over the gap; rate None is plain linear interpolation."""
    s = (t_gap - t_gap[0]) / (t_gap[-1] - t_gap[0])
    if rate is None:
        return xa + s * (xb - xa)
    I = KAPPA * np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t_gap))])
    return xa + I + s * (xb - xa - I[-1])


def diagnostics(t, group, x, y, a_t, a_y, b_t, b_y) -> str:
    estimators = ("A only: linear interpolation", "A + B, the wrong driver y2", "A + B, the driver y1",
                  "A + the true y1 (floor: the noise of A)")
    sq = {name: [] for name in estimators}
    inc, int1, int2 = [], [], []
    for g, xi, yi, ta, ya, tb, yb in zip(group, x, y, a_t, a_y, b_t, b_y):
        if g != 0:
            continue
        for k in range(len(ta) - 1):
            i0, i1 = np.searchsorted(t, ta[k]), np.searchsorted(t, ta[k + 1])
            tg = t[i0:i1 + 1]
            rates = (None, np.interp(tg, tb, yb[:, 1]), np.interp(tg, tb, yb[:, 0]), yi[i0:i1 + 1, 0])
            for name, rate in zip(estimators, rates):
                sq[name] += list((bridge(tg, ya[k, 0], ya[k + 1, 0], rate) - xi[i0:i1 + 1])[1:-1] ** 2)
            inc.append(ya[k + 1, 0] - ya[k, 0])
            int1.append(np.trapezoid(np.interp(tg, tb, yb[:, 0]), tg))
            int2.append(np.trapezoid(np.interp(tg, tb, yb[:, 1]), tg))
    b_only = np.concatenate([yb for g, yb in zip(group, b_y) if g == 2])
    lines = [f"train split: {len(group)} trajectories, groups (both, A only, B only) {np.bincount(group, minlength=3)}"]
    if not inc:
        return "\n".join(lines + ["", "no trajectory has both modalities: nothing shows which driver moves x"])
    lines += ["", f"x between consecutive A visits, trajectories with both modalities ({len(inc)} gaps): RMSE "
             f"(std of x over a whole trajectory {x.std(1).mean():.2f})"]
    lines += [f"   {name:<44}{np.sqrt(np.mean(v)):.3f}" for name, v in sq.items()]
    lines += ["", "correlation of the A step over a gap with the integral of B over the same gap (both modalities):",
              f"   y1 {np.corrcoef(inc, int1)[0, 1]:+.3f}, y2 {np.corrcoef(inc, int2)[0, 1]:+.3f}: x is driven by y1, "
              "which only trajectories with both modalities show",
              "", "B only trajectories: y1 and y2 have the same law, so B alone cannot tell the driver",
              f"   std y1 {b_only[:, 0].std():.3f}, y2 {b_only[:, 1].std():.3f} (true {np.hypot(SIGMA, B_NOISE):.3f} "
              "with the noise of B)"]
    return "\n".join(lines)


def plot_overview(out: Path, t, group, x, y, a_t, a_y, b_t, b_y, x_noise: float = 0.0):
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), layout="constrained", sharex=True)
    for row, g in enumerate(range(3)):
        i = int(np.argmax(group == g))
        ax = axes[row, 0]
        ax.plot(t, x[i], color=INK, lw=1, label="x, hidden")
        if len(a_t[i]):
            ax.plot(a_t[i], a_y[i][:, 0], "o", color=A_COLOR, ms=6, mec="white", mew=0.8, label="A")
        else:
            ax.text(0.5, 0.85, "A not observed", transform=ax.transAxes, ha="center", color=MUTED, bbox=LABEL_BOX)
        ax.set_ylabel("x")
        ax.set_title(f"trajectory {i}, {GROUPS[g]}", loc="left", fontsize=10)
        ax = axes[row, 1]
        for j, color in ((0, Y1_COLOR), (1, Y2_COLOR)):
            ax.plot(t, y[i, :, j], color=color, lw=0.8, alpha=0.6, label=f"y{j + 1}, hidden")
            if len(b_t[i]):
                ax.plot(b_t[i], b_y[i][:, j], "o", color=color, ms=3.5, mec="white", mew=0.5, label=f"B: y{j + 1}")
        if not len(b_t[i]):
            ax.text(0.5, 0.85, "B not observed", transform=ax.transAxes, ha="center", color=MUTED, bbox=LABEL_BOX)
        ax.set_ylabel("y")
    for ax in axes.ravel():
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2, frameon=False, fontsize=8, ncol=4)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    noise = f" + {x_noise:g} dW" if x_noise > 0 else ""
    fig.suptitle(f"simple_multimodal_test: dx = {KAPPA:g} y1 dt{noise}; y1, y2 OU (tau {TAU:g}, std {SIGMA:g}); "
                 "A sees x rarely, B sees (y1, y2) often", x=0.01, ha="left", fontsize=11)
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
    p.add_argument("--x-noise", type=float, default=0.0, help="diffusion of x: dx = KAPPA y1 dt + X_NOISE dW")
    p.add_argument("--a-gap-mean", type=float, default=A_GAP_MEAN, help="mean time between two A observations")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split_id, (split, n) in enumerate(zip(SPLITS, (args.n_train, args.n_val, args.n_test))):
        t, group, x, y, a_t, a_y, b_t, b_y = generate_split(split_id, n, args.fractions, args.seed, args.x_noise, args.a_gap_mean)
        with h5py.File(args.out_dir / f"{split}.h5", "w") as f:
            f["truth/t"], f["truth/x"], f["truth/y"] = t, x, y
            for name, times, values, width in (("A", a_t, a_y, 1), ("B", b_t, b_y, 2)):
                f[f"{name}/t"] = pad([s[:, None] for s in times], np.nan, 1)[..., 0]
                f[f"{name}/y"] = pad(values, 0.0, width)
                f[f"{name}/count"] = np.array([len(s) for s in times])
            f["group"] = group
            f.attrs.update(
                dynamics=f"dx = {KAPPA} y1 dt + {args.x_noise} dW; y1, y2 independent stationary OU, time scale "
                         f"{TAU}, std {SIGMA}; x(0) ~ N(0, {X0_STD}^2)",
                A=f"x + N(0, {A_NOISE}^2), Gamma-distributed gaps (shape {A_GAP_SHAPE}, mean {args.a_gap_mean})",
                B=f"(y1, y2) + N(0, {B_NOISE}^2), Gamma-distributed gaps (shape {B_GAP_SHAPE}, mean {B_GAP_MEAN})",
                groups="0 both modalities, 1 A only, 2 B only", fractions=args.fractions, seed=args.seed,
                kappa=KAPPA, tau=TAU, sigma=SIGMA, x_noise=args.x_noise,
            )
        print(f"{split}.h5: {n} trajectories, groups (both, A only, B only) {np.bincount(group, minlength=3)}")
        if split == "train":
            text = diagnostics(t, group, x, y, a_t, a_y, b_t, b_y)
            (args.out_dir / "diagnostics.txt").write_text(text + "\n")
            print(text)
            plot_overview(args.out_dir / "overview.png", t, group, x, y, a_t, a_y, b_t, b_y, args.x_noise)


if __name__ == "__main__":
    main()
