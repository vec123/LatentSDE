"""
Tests what the images of the multimodal_reaction_test toy show by themselves. The hidden forcing gamma(t) only
turns the field (test_data/src/multimodal_reaction.py), so by symmetry no image shows gamma or its history; this
script checks that in numbers, the way it found the leak of the first design (a hidden beta(t), whose history the
spiral arms record).

Fitted on the train split, scored on the test split, on frames with the imaging noise added, with features that do
not change when the spiral rotates (|c1(r)|, the local wavenumber profile, the radial power spectrum). Each test
uses a linear model (ridge) and a nonlinear one (gradient boosting); a leak is the better of the two. The
unforced rotation rate omega(beta) of the trajectory (the slope of phase0) is known to every predictor: beta is
constant, the images show it in the spiral's pitch, and a model can take it as mu. Given as omega, not beta, so
that no predictor has to extrapolate omega(beta) outside the betas of the train split.

1. One image: R^2 of gamma(t), of the mean of gamma over [t - h, t] (past), and over [t, t + 5] (future).
2. Phase advance over a visit gap (Gamma-distributed gaps as in sample_modalities.py): the images give it modulo
   2 pi, a predictor of the unwrapped advance picks the integer. Fraction of gaps where the integer is correct,
   from: the gap length and omega only (chance), plus the images at both ends (leak), plus the biomarker in the gap
   (the intended source), plus both, and the true gamma (upper bound).

    python -m test_data.scripts.multimodal_reaction_test.check_leak [--dir test_data/data/multimodal_reaction_test]

Writes <dir>/leak_check.txt and prints it.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from test_data.src.multimodal_reaction import (BIO_GAP_MEAN, BIO_GAP_SHAPE, BIO_NOISE, IMAGE_GAP_MEAN,
                                               IMAGE_GAP_SHAPE, IMAGE_NOISE, R0, angular_mode,
                                               angular_mode_matrix, invariant_features, renewal_times, truth_frames)

DATA = Path(__file__).resolve().parents[2] / "data" / "multimodal_reaction_test"
PAST, FUTURE = (2.0, 5.0, 10.0, 20.0), 5.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def models():
    return {"ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
            "boosting": HistGradientBoostingRegressor(max_iter=300, random_state=0)}


def load(path: Path, rng: np.random.Generator, n_pairs: int):
    """Per-frame features and targets, and per-pair features and targets, of every trajectory in one split."""
    frames, pairs = [], []
    with h5py.File(path, "r") as f:
        t, t_frames, x = f["t"][:], f["t_frames"][:], f["x"][:]
        stride = int(f.attrs["frame_stride"])
        K, K0 = angular_mode_matrix(x), angular_mode_matrix(x, np.array([R0]))
        dtf = t_frames[1] - t_frames[0]
        for i in range(f["beta"].shape[0]):
            gamma, Gamma, phase = f["gamma"][i], f["Gamma"][i], f["phase"][i]
            omega = np.polyfit(t_frames, f["phase0"][i], 1)[0]
            u = truth_frames(f, i)
            u = u + IMAGE_NOISE * rng.standard_normal(u.shape).astype(np.float32)
            feat = invariant_features(u, K)
            phase_img = -np.angle(angular_mode(u, K0)[:, 0])
            mean_gamma = lambda a, b: (np.interp(b, t, Gamma) - np.interp(a, t, Gamma)) / (b - a)

            s = t_frames
            target = {"gamma(t)": gamma[::stride][:len(s)]}
            for h in PAST:
                target[f"past {h:g}"] = np.where(s >= h, mean_gamma(s - h, s), np.nan)
            target[f"future {FUTURE:g}"] = np.where(s + FUTURE <= t[-1], mean_gamma(s, s + FUTURE), np.nan)
            frames.append((feat, target))

            bio_t = renewal_times(t[-1], BIO_GAP_MEAN, BIO_GAP_SHAPE, rng)
            bio_y = np.interp(bio_t, t, gamma) + BIO_NOISE * rng.standard_normal(len(bio_t))
            steps = np.maximum(np.round(rng.gamma(IMAGE_GAP_SHAPE, IMAGE_GAP_MEAN / IMAGE_GAP_SHAPE, n_pairs)
                                        / dtf).astype(int), 1)
            steps = np.minimum(steps, len(s) - 2)
            k1 = (rng.uniform(size=n_pairs) * (len(s) - 1 - steps)).astype(int)
            k2 = k1 + steps
            bio_mean = np.array([np.interp(np.linspace(a, b, 200), bio_t, bio_y).mean() for a, b in zip(s[k1], s[k2])])
            pairs.append({"gap": s[k2] - s[k1], "omega": np.full(n_pairs, omega), "advance": phase[k2] - phase[k1],
                          "advance_mod": wrap(phase_img[k2] - phase_img[k1]),
                          "img": np.concatenate([feat[k1], feat[k2]], 1), "bio": bio_mean,
                          "true": mean_gamma(s[k1], s[k2])})
    frame_feat = np.concatenate([fe for fe, _ in frames])
    frame_target = {k: np.concatenate([tg[k] for _, tg in frames]) for k in frames[0][1]}
    pair = {k: np.concatenate([p[k] for p in pairs]) for k in pairs[0]}
    return frame_feat, frame_target, pair


def pair_inputs(pair):
    base = np.stack([pair["gap"], pair["omega"]], 1)
    return {"gap length, omega (chance)": base,
            "+ images at both ends (leak)": np.concatenate([base, pair["img"]], 1),
            "+ biomarker in the gap": np.concatenate([base, pair["bio"][:, None]], 1),
            "+ images and biomarker": np.concatenate([base, pair["bio"][:, None], pair["img"]], 1),
            "+ true gamma (upper bound)": np.concatenate([base, pair["true"][:, None]], 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DATA)
    p.add_argument("--pairs", type=int, default=400, help="visit gaps drawn per trajectory")
    p.add_argument("--frame-step", type=int, default=1, help="use every this-th frame for test 1")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    Xf_train, yf_train, pair_train = load(args.dir / "truth_train.h5", rng, args.pairs)
    Xf_test, yf_test, pair_test = load(args.dir / "truth_test.h5", rng, args.pairs)
    lines = [f"leak check, {args.dir.name}: fitted on train, scored on test; image noise {IMAGE_NOISE}, "
             f"biomarker noise {BIO_NOISE}, imaging gaps Gamma-distributed (shape {IMAGE_GAP_SHAPE}, mean "
             f"{IMAGE_GAP_MEAN})", ""]

    lines.append("1. one image -> gamma; R^2 on test (0 = no better than the mean, 1 = exact)")
    lines.append(f"   {'target':<14}{'ridge':>8}{'boosting':>10}")
    sl = slice(None, None, args.frame_step)
    for name in yf_train:
        ok_tr, ok_te = np.isfinite(yf_train[name][sl]), np.isfinite(yf_test[name][sl])
        scores = []
        for model in models().values():
            model.fit(Xf_train[sl][ok_tr], yf_train[name][sl][ok_tr])
            scores.append(r2_score(yf_test[name][sl][ok_te], model.predict(Xf_test[sl][ok_te])))
        lines.append(f"   {name:<14}{scores[0]:>8.3f}{scores[1]:>10.3f}")

    err = np.abs(wrap(pair_test["advance_mod"] - pair_test["advance"]))
    lines += ["", "2. phase advance over a visit gap; the images give it modulo 2 pi "
                  f"(median error {np.median(err):.3f} rad, 99th percentile {np.percentile(err, 99):.3f} rad)",
              f"   test gaps: {len(err)}, true advance {pair_test['advance'].mean() / (2 * np.pi):.2f} turns on "
              f"average (std {pair_test['advance'].std() / (2 * np.pi):.2f})",
              f"   {'predictor':<30}{'resolved, ridge':>17}{'boosting':>10}{'RMSE (turns)':>14}"]
    inputs_train, inputs_test = pair_inputs(pair_train), pair_inputs(pair_test)
    for name in inputs_train:
        resolved, rmse = [], []
        for model in models().values():
            model.fit(inputs_train[name], pair_train["advance"] / pair_train["gap"])
            pred = model.predict(inputs_test[name]) * pair_test["gap"]
            k = np.round((pred - pair_test["advance_mod"]) / (2 * np.pi))
            resolved.append(np.mean(np.abs(pair_test["advance_mod"] + 2 * np.pi * k - pair_test["advance"]) < np.pi))
            rmse.append(np.sqrt(np.mean((pred - pair_test["advance"]) ** 2)) / (2 * np.pi))
        lines.append(f"   {name:<30}{resolved[0]:>17.3f}{resolved[1]:>10.3f}{min(rmse):>14.3f}")

    text = "\n".join(lines)
    print(text)
    (args.dir / "leak_check.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
