"""
Generate simple_multimodal_test_nonlinear: simple_multimodal_test_xnoise_dense_x with a nonlinear drive of x. The
paths of y1, y2, the observation times and the observation noise are the same as in simple_multimodal_test (same
seed); only x changes:

    dx = (y1 + C_QUAD (y2^2 - SIGMA^2)) dt + X_NOISE dW_0,   dy_j = -y_j / TAU dt + SIGMA sqrt(2 / TAU) dW_j.

The term y2^2 - SIGMA^2 has zero mean, so x does not drift away. The symmetry of the linear test is gone: x depends
on y1 linearly and on y2 quadratically, so a linear model cannot fit it, and only trajectories with both
modalities show the drive.

    python -m test_data.scripts.simple_multimodal_nonlinear.generate [--n-train 100] [--n-val 20] [--n-test 30]
        [--x-noise 0.2] [--a-gap-mean 3.3333] [--c-quad 3] [--out-dir test_data/data/simple_multimodal_test_nonlinear]

Output format: as simple_multimodal_test (see its generate.py).
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from test_data.scripts.simple_multimodal_test import generate as base
from test_data.src.multimodal_reaction import ou_path, renewal_times

DATA = Path(__file__).resolve().parents[2] / "data" / "simple_multimodal_test_nonlinear"
C_QUAD = 3.0


def drive(y, c_quad):
    return y[..., 0] + c_quad * (y[..., 1] ** 2 - base.SIGMA ** 2)


def generate_split(split_id, n, fractions, seed, x_noise, a_gap_mean, c_quad):
    t = np.arange(0.0, base.T + base.DT / 2, base.DT)
    rng = np.random.default_rng(np.random.SeedSequence([seed, split_id]))
    counts = np.floor(np.asarray(fractions) / sum(fractions) * n).astype(int)
    counts[0] += n - counts.sum()
    group = rng.permutation(np.repeat(np.arange(3), counts))
    xs, ys, a_t, a_y, b_t, b_y = [], [], [], [], [], []
    for g, stream in zip(group, np.random.SeedSequence([seed, split_id, 1]).spawn(n)):
        r = np.random.default_rng(stream)
        y = np.stack([ou_path(t, r, base.TAU, base.SIGMA), ou_path(t, r, base.TAU, base.SIGMA)], -1)
        f = drive(y, c_quad)
        steps = 0.5 * (f[1:] + f[:-1]) * np.diff(t)
        x = base.X0_STD * r.standard_normal() + np.concatenate([[0.0], np.cumsum(steps)])
        if x_noise > 0:
            dw = np.sqrt(np.diff(t)) * np.random.default_rng(stream.spawn(1)[0]).standard_normal(len(t) - 1)
            x = x + x_noise * np.concatenate([[0.0], np.cumsum(dw)])
        ka = np.unique(np.searchsorted(t, renewal_times(base.T, a_gap_mean, base.A_GAP_SHAPE, r)).clip(0, len(t) - 1))
        kb = np.unique(np.searchsorted(t, renewal_times(base.T, base.B_GAP_MEAN, base.B_GAP_SHAPE, r)).clip(0, len(t) - 1))
        ya = x[ka, None] + base.A_NOISE * r.standard_normal((len(ka), 1))
        yb = y[kb] + base.B_NOISE * r.standard_normal((len(kb), 2))
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


def bridge(tg, xa, xb, rate):
    """x between two A readings: the integral of rate from the left reading, mismatch spread linearly."""
    s = (tg - tg[0]) / (tg[-1] - tg[0])
    if rate is None:
        return xa + s * (xb - xa)
    I = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(tg))])
    return xa + I + s * (xb - xa - I[-1])


def diagnostics(t, group, x, y, a_t, a_y, b_t, b_y, c_quad) -> str:
    names = ("A only: linear interpolation", "A + B, linear drive y1", "A + B, the true drive from B",
             "A + the true drive (floor: the noise of A)")
    sq = {k: [] for k in names}
    for g, xi, yi, ta, ya, tb, yb in zip(group, x, y, a_t, a_y, b_t, b_y):
        if g != 0:
            continue
        for k in range(len(ta) - 1):
            i0, i1 = np.searchsorted(t, ta[k]), np.searchsorted(t, ta[k + 1])
            tg = t[i0:i1 + 1]
            yg = np.stack([np.interp(tg, tb, yb[:, 0]), np.interp(tg, tb, yb[:, 1])], -1)
            rates = (None, yg[:, 0], drive(yg, c_quad), drive(yi[i0:i1 + 1], c_quad))
            for name, rate in zip(names, rates):
                sq[name] += list((bridge(tg, ya[k, 0], ya[k + 1, 0], rate) - xi[i0:i1 + 1])[1:-1] ** 2)
    lines = [f"train split: {len(group)} trajectories, groups (both, A only, B only) {np.bincount(group, minlength=3)}",
             "", f"x between consecutive A visits, trajectories with both modalities: RMSE "
             f"(std of x over a whole trajectory {x.std(1).mean():.2f})"]
    lines += [f"   {k:<44}{np.sqrt(np.mean(v)):.3f}" for k, v in sq.items()]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DATA)
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--fractions", type=float, nargs=3, default=(0.4, 0.3, 0.3))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--x-noise", type=float, default=0.2)
    p.add_argument("--a-gap-mean", type=float, default=10.0 / 3)
    p.add_argument("--c-quad", type=float, default=C_QUAD)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split_id, (split, n) in enumerate(zip(base.SPLITS, (args.n_train, args.n_val, args.n_test))):
        t, group, x, y, a_t, a_y, b_t, b_y = generate_split(
            split_id, n, args.fractions, args.seed, args.x_noise, args.a_gap_mean, args.c_quad)
        with h5py.File(args.out_dir / f"{split}.h5", "w") as f:
            f["truth/t"], f["truth/x"], f["truth/y"] = t, x, y
            for name, times, values, width in (("A", a_t, a_y, 1), ("B", b_t, b_y, 2)):
                f[f"{name}/t"] = base.pad([s[:, None] for s in times], np.nan, 1)[..., 0]
                f[f"{name}/y"] = base.pad(values, 0.0, width)
                f[f"{name}/count"] = np.array([len(s) for s in times])
            f["group"] = group
            f.attrs.update(
                dynamics=f"dx = (y1 + {args.c_quad} (y2^2 - {base.SIGMA}^2)) dt + {args.x_noise} dW; y1, y2 independent "
                         f"stationary OU, time scale {base.TAU}, std {base.SIGMA}; x(0) ~ N(0, {base.X0_STD}^2)",
                A=f"x + N(0, {base.A_NOISE}^2), Gamma-distributed gaps (shape {base.A_GAP_SHAPE}, mean {args.a_gap_mean:.4f})",
                B=f"(y1, y2) + N(0, {base.B_NOISE}^2), Gamma-distributed gaps (shape {base.B_GAP_SHAPE}, mean {base.B_GAP_MEAN})",
                groups="0 both modalities, 1 A only, 2 B only", fractions=args.fractions, seed=args.seed,
                kappa=1.0, c_quad=args.c_quad, tau=base.TAU, sigma=base.SIGMA, x_noise=args.x_noise,
            )
        print(f"{split}.h5: {n} trajectories, groups {np.bincount(group, minlength=3)}")
        if split == "train":
            text = diagnostics(t, group, x, y, a_t, a_y, b_t, b_y, args.c_quad)
            (args.out_dir / "diagnostics.txt").write_text(text + "\n")
            print(text)
            base.plot_overview(args.out_dir / "overview.png", t, group, x, y, a_t, a_y, b_t, b_y, args.x_noise)


if __name__ == "__main__":
    main()
