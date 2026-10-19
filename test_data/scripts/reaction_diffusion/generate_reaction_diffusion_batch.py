"""
Generate several short reaction-diffusion trajectories, for training a LatentSDE.

Same lambda-omega system as generate_reaction_diffusion.py, same grid and the same per-step numerics (same dt
and substeps, so the same validated accuracy), but nt snapshots instead of 10000, and n_traj trajectories
instead of one. Each trajectory starts from the same spiral, rotated by a different phase (2 pi i / n_traj +
phase_offset), so the n_traj trajectories differ even with the same physics (d1, d2, beta). phase_offset is 0
by default (the training set); pass a different one (e.g. pi / n_traj, halfway between the training phases) to
generate a val or test set whose trajectories are not reproductions of the training ones.

beta can vary per trajectory too (the reaction coupling: it changes the dynamics, not just the initial
condition, unlike phase), by --beta-min/--beta-max (a linspace over n_traj) or --betas (explicit, one per
trajectory); either way it is saved as a "beta" dataset, (n_traj,), so it can be loaded as mu
(load_reaction_diffusion's param). Default: every trajectory uses the same --beta, and no mu is exercised.

Only u and v are stored (no du_dt, dv_dt: those are for the SINDy baselines, not needed to train a LatentSDE).
"""
import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from .generate_reaction_diffusion import solve

DATA = Path(__file__).resolve().parents[2] / "data" / "reaction_diffusion"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DATA / "reaction_diffusion_batch.h5")
    p.add_argument("--n-traj", type=int, default=10)
    p.add_argument("--n", type=int, default=100, help="grid points per dimension")
    p.add_argument("--L", type=float, default=20.0, help="domain side length")
    p.add_argument("--nt", type=int, default=2000, help="snapshots per trajectory (nt - 1 steps)")
    p.add_argument("--dt", type=float, default=500.0 / 9999, help="snapshot spacing (default: matches the single"
                   " long trajectory, so the ETDRK4 step size, dt / substeps, stays the one already validated)")
    p.add_argument("--d1", type=float, default=0.1)
    p.add_argument("--d2", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=1.0, help="used for every trajectory, unless --beta-min/"
                   "--beta-max or --betas is given")
    p.add_argument("--beta-min", type=float, default=None, help="with --beta-max, beta is linspace(min, max,"
                   " n_traj) instead of the constant --beta")
    p.add_argument("--beta-max", type=float, default=None)
    p.add_argument("--betas", type=float, nargs="+", default=None, help="explicit beta per trajectory (n_traj"
                   " values); overrides --beta and --beta-min/--beta-max")
    p.add_argument("--substeps", type=int, default=5, help="ETDRK4 steps per snapshot interval")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--phase-offset", type=float, default=0.0, help="added to every trajectory's phase (radians);"
                   " use a different value than the training set's (default 0) to get different trajectories")
    args = p.parse_args()

    if args.betas is not None:
        if len(args.betas) != args.n_traj:
            p.error(f"--betas needs {args.n_traj} values (one per trajectory), got {len(args.betas)}")
        betas = np.array(args.betas, dtype=np.float64)
    elif args.beta_min is not None or args.beta_max is not None:
        if args.beta_min is None or args.beta_max is None:
            p.error("give both --beta-min and --beta-max, or neither")
        betas = np.linspace(args.beta_min, args.beta_max, args.n_traj)
    else:
        betas = np.full(args.n_traj, args.beta)

    T = args.dt * (args.nt - 1)
    print(f"Solving {args.n_traj} trajectories on a {args.n}x{args.n} grid, t in [0, {T:.2f}] "
          f"with {args.nt - 1} steps each, beta = {betas} ...")

    u = np.empty((args.n_traj, args.nt, args.n, args.n), dtype=args.dtype)
    v = np.empty((args.n_traj, args.nt, args.n, args.n), dtype=args.dtype)
    t0 = time.perf_counter()
    for i in range(args.n_traj):
        t, x, y, states, _ = solve(args.n, args.L, T, args.nt, args.d1, args.d2, betas[i], args.substeps,
                                    np.dtype(args.dtype), phase=2 * np.pi * i / args.n_traj + args.phase_offset)
        u[i], v[i] = states[:, 0], states[:, 1]
        print(f"  trajectory {i + 1}/{args.n_traj} done, beta={betas[i]:.3f} ({time.perf_counter() - t0:.0f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    chunks = (1, min(64, args.nt), args.n, args.n)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("t", data=t)
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("u", data=u, chunks=chunks)
        f.create_dataset("v", data=v, chunks=chunks)
        f.create_dataset("beta", data=betas)
        f.attrs.update(
            equations="u_t = (1-A^2)u + beta*A^2*v + d1*lap(u); v_t = -beta*A^2*u + (1-A^2)v + d2*lap(v); A^2 = u^2+v^2",
            initial_condition="u0 = tanh(r)cos(theta - r + phase), v0 = tanh(r)sin(theta - r + phase), "
                              "phase = 2 pi i / n_traj + phase_offset",
            boundary="periodic",
            layout="fields are (trajectory, time, y, x): u[i, k, a, b] = u_i(x[b], y[a], t[k]); beta[i] is"
                  " trajectory i's reaction coupling (constant across all trajectories unless --beta-min/"
                  "--beta-max or --betas was given)",
            solver="pseudo-spectral (rfft2) + ETDRK4",
            d1=args.d1, d2=args.d2, L=args.L, n=args.n, substeps=args.substeps, phase_offset=args.phase_offset,
            reference="Champion, Lusch, Kutz & Brunton, PNAS 116(45), 2019",
        )

    print(f"Wrote {args.out}  (u range [{u.min():.3f}, {u.max():.3f}])")


if __name__ == "__main__":
    main()
