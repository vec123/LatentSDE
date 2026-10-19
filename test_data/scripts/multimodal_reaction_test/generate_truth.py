"""
Generate the dense ground truth of the multimodal_reaction_test toy (test_data/src/multimodal_reaction.py):
lambda-omega spirals with the same solver, grid and step as test_data/scripts/reaction_diffusion, a constant
beta per trajectory, and a hidden phase-rate forcing gamma(t). The PDE is solved once without the forcing (w0);
the forced field is exactly w = exp(-i Gamma(t)) w0, so the driver is drawn after the solve, and --driver-only
draws it again in existing files without a new solve.

    python -m test_data.scripts.multimodal_reaction_test.generate_truth [--n-train 20] [--n-val 4] [--n-test 4]
                                                                         [--nt 4000] [--workers 6]
    python -m test_data.scripts.multimodal_reaction_test.generate_truth --driver-only [--tau 5] [--gamma-std 0.5]

Writes <out-dir>/truth_<split>.h5 for each split:
    t (nt,)                          solver snapshot times, dt = 500 / 9999 as in reaction_diffusion
    beta, initial_phase (n_traj,)    the constant coupling (stratified over BETA_RANGE), the initial spiral's rotation
    t_frames (n_frames,)             every --frame-stride-th snapshot time
    u0, v0 (n_traj, n_frames, n, n)  the unforced field at t_frames; the forced u is truth_frames(f, i, k)
    phase0 (n_traj, n_frames)        unwrapped rotation phase of w0, -arg c1(R0)
    gamma, Gamma (n_traj, nt)        the hidden forcing and its integral
    phase (n_traj, n_frames)         unwrapped rotation phase of the forced field, phase0 + Gamma
The constants of a split come from the seed (seed, split), the drivers from streams spawned from (seed, split, 1).
The modalities are drawn from these files by sample_modalities.py.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

from test_data.scripts.reaction_diffusion.generate_reaction_diffusion import solve
from test_data.src.multimodal_reaction import (BETA_RANGE, GAMMA_STD, R0, TAU, angular_mode, angular_mode_matrix,
                                               integral, ou_path, unwrapped_phase)

DATA = Path(__file__).resolve().parents[2] / "data" / "multimodal_reaction_test"
SPLITS = ("train", "val", "test")


def draw_constants(seed: int, split_id: int, m: int):
    """beta (m,) stratified over BETA_RANGE (one in each of m equal bins, shuffled), so every split, even a small
    one, covers the range; and the initial phases (m,), uniform."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, split_id]))
    low, high = BETA_RANGE
    beta = low + (high - low) * (rng.permutation(m) + rng.uniform(size=m)) / m
    return beta, rng.uniform(0.0, 2 * np.pi, m)


def simulate(split_id: int, i: int, beta: float, phase0: float, n: int, L: float, nt: int, dt: float, substeps: int,
             stride: int):
    _, x, _, states, _ = solve(n, L, dt * (nt - 1), nt, beta=beta, substeps=substeps, dtype=np.dtype("float32"),
                               phase=phase0, derivatives=False)
    frames = states[::stride]
    c1 = angular_mode(frames[:, 0] + 1j * frames[:, 1], angular_mode_matrix(x, np.array([R0])))[:, 0]
    return split_id, i, beta, phase0, frames[:, 0].copy(), frames[:, 1].copy(), unwrapped_phase(c1), np.abs(c1).min()


def write_driver(f: h5py.File, seed: int, tau: float, gamma_std: float):
    t, stride = f["t"][:], int(f.attrs["frame_stride"])
    streams = np.random.SeedSequence([seed, int(f.attrs["split_id"]), 1]).spawn(f["beta"].shape[0])
    for i, stream in enumerate(streams):
        gamma = ou_path(t, np.random.default_rng(stream), tau, gamma_std)
        Gamma = integral(t, gamma)
        f["gamma"][i], f["Gamma"][i], f["phase"][i] = gamma, Gamma, f["phase0"][i] + Gamma[::stride]
    f.attrs.update(tau=tau, gamma_std=gamma_std, driver_seed=seed,
                   driver=f"gamma stationary OU, time scale {tau}, std {gamma_std}; Gamma its integral (trapezoid "
                          f"on t); forced w = exp(-i Gamma) w0, i.e. u_t += gamma v, v_t += -gamma u")
    rate = np.diff(f["phase"][:], axis=1) / np.diff(f["t_frames"][:])
    print(f"  {SPLITS[int(f.attrs['split_id'])]}: driver drawn, rotation rate [{rate.min():.2f}, {rate.max():.2f}], "
          f"mean {rate.mean():.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DATA)
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-val", type=int, default=4)
    p.add_argument("--n-test", type=int, default=4)
    p.add_argument("--n", type=int, default=100, help="grid points per dimension")
    p.add_argument("--L", type=float, default=20.0, help="domain side length")
    p.add_argument("--nt", type=int, default=4000, help="snapshots per trajectory (nt - 1 steps)")
    p.add_argument("--dt", type=float, default=500.0 / 9999, help="snapshot spacing, the validated one by default")
    p.add_argument("--substeps", type=int, default=5, help="ETDRK4 steps per snapshot interval")
    p.add_argument("--frame-stride", type=int, default=4, help="store u0, v0 at every this-th snapshot")
    p.add_argument("--tau", type=float, default=TAU, help="time scale of the OU forcing gamma")
    p.add_argument("--gamma-std", type=float, default=GAMMA_STD, help="stationary std of gamma")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6, help="parallel processes, one trajectory each")
    p.add_argument("--driver-only", action="store_true", help="draw gamma again in the existing files, no solve")
    args = p.parse_args()

    if args.driver_only:
        for split in SPLITS:
            with h5py.File(args.out_dir / f"truth_{split}.h5", "r+") as f:
                write_driver(f, args.seed, args.tau, args.gamma_std)
        return

    counts = dict(zip(SPLITS, (args.n_train, args.n_val, args.n_test)))
    t = np.linspace(0.0, args.dt * (args.nt - 1), args.nt)
    t_frames = t[::args.frame_stride]
    x = np.linspace(-args.L / 2, args.L / 2, args.n + 1)[:-1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split_id, split in enumerate(SPLITS):
        f = h5py.File(args.out_dir / f"truth_{split}.h5", "w")
        m = counts[split]
        f.create_dataset("t", data=t)
        f.create_dataset("t_frames", data=t_frames)
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=x)
        for name in ("beta", "initial_phase"):
            f.create_dataset(name, (m,), dtype="f8")
        for name in ("gamma", "Gamma"):
            f.create_dataset(name, (m, args.nt), dtype="f8")
        for name in ("phase0", "phase"):
            f.create_dataset(name, (m, len(t_frames)), dtype="f8")
        for name in ("u0", "v0"):
            f.create_dataset(name, (m, len(t_frames), args.n, args.n), dtype="f4",
                             chunks=(1, min(64, len(t_frames)), args.n, args.n))
        f.attrs.update(
            equations="w = u + i v: w_t = (1 - |w|^2) w - i beta |w|^2 w + d lap(w) - i gamma(t) w, i.e. "
                      "u_t = (1-A^2)u + beta*A^2*v + d*lap(u) + gamma*v; v_t = -beta*A^2*u + (1-A^2)v + d*lap(v) "
                      "- gamma*u; A^2 = u^2+v^2",
            initial_condition="u0 = tanh(r)cos(theta - r + phase), v0 = tanh(r)sin(theta - r + phase), "
                              "phase = initial_phase, uniform in [0, 2 pi)",
            beta_range=BETA_RANGE,
            phase=f"-unwrap(arg c1(R0)), c1(r) = mean over theta of w(r, theta) exp(-i theta), R0 = {R0}",
            layout="u0[i, k, a, b] = u0_i(x[b], y[a], t_frames[k]); gamma[i, k] at t[k]",
            boundary="periodic", solver="pseudo-spectral (rfft2) + ETDRK4", d=0.1, L=args.L, n=args.n,
            substeps=args.substeps, frame_stride=args.frame_stride, seed=args.seed, split_id=split_id,
        )
        files[split_id] = f

    start = time.perf_counter()
    jobs = [(s, i, beta, phase0, args.n, args.L, args.nt, args.dt, args.substeps, args.frame_stride)
            for s, split in enumerate(SPLITS)
            for i, (beta, phase0) in enumerate(zip(*draw_constants(args.seed, s, counts[split])))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(simulate, *job) for job in jobs]):
            s, i, beta, phase0, u0, v0, phase, c1_min = future.result()
            f = files[s]
            f["beta"][i], f["initial_phase"][i], f["u0"][i], f["v0"][i], f["phase0"][i] = beta, phase0, u0, v0, phase
            rate = np.diff(phase) / np.diff(t_frames)
            print(f"  {SPLITS[s]} {i}: beta {beta:.2f}, unforced rate [{rate.min():.2f}, {rate.max():.2f}], "
                  f"min |c1(R0)| {c1_min:.3f}, u0 [{u0.min():.2f}, {u0.max():.2f}] "
                  f"({time.perf_counter() - start:.0f}s)", flush=True)
    for f in files.values():
        write_driver(f, args.seed, args.tau, args.gamma_std)
        f.close()
    print(f"Wrote {', '.join(f'truth_{s}.h5' for s in SPLITS)} to {args.out_dir}")


if __name__ == "__main__":
    main()
