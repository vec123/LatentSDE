"""Generate the lambda-omega reaction-diffusion dataset of Champion et al. (2019).

    u_t = (1 - A^2) u + beta A^2 v + d1 (u_xx + u_yy)
    v_t = -beta A^2 u + (1 - A^2) v + d2 (v_xx + v_yy),     A^2 = u^2 + v^2

on [-10, 10]^2 with periodic boundaries, d1 = d2 = 0.1, beta = 1, starting from the
single-armed spiral used in the PDE-FIND / SINDy-autoencoder MATLAB scripts:

    u0 = tanh(r) cos(theta - r),   v0 = tanh(r) sin(theta - r)

Space is discretised pseudo-spectrally on a 100 x 100 grid (x = linspace(-10, 10, 101)[:-1],
as in the MATLAB reference) and time is integrated with ETDRK4 (Kassam & Trefethen 2005),
which treats diffusion exactly. Snapshots are stored on [0, 500] with 9999 steps.

The observed state is u (D = 100 x 100); v and the exact time derivatives are stored too,
because the SINDy baselines need du/dt. Splits: train t <= 400, val (400, 450], test (450, 500].
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import fft

DATA = Path(__file__).resolve().parents[2] / "data" / "reaction_diffusion"


def etdrk4_coefficients(L, h, M=32):
    """ETDRK4 coefficients for a diagonal linear operator L, via contour integrals."""
    E = np.exp(h * L)
    E2 = np.exp(h * L / 2)
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * L[..., None] + r
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=-1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=-1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=-1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=-1))
    return E, E2, Q, f1, f2, f3


def reaction(uv, beta):
    """Reaction terms of the lambda-omega system; uv has shape (2, n, n)."""
    u, v = uv
    A2 = u**2 + v**2
    return np.stack([(1 - A2) * u + beta * A2 * v, -beta * A2 * u + (1 - A2) * v])


def solve(n=100, L=20.0, T=500.0, nt=10000, d1=0.1, d2=0.1, beta=1.0, substeps=5, dtype=np.float64, phase=0.0,
          derivatives=True):
    """beta is a scalar, or a path (nt,) of its value at each snapshot: the step from snapshot i to i + 1 holds
    beta[i] fixed over its substeps. beta only enters the reaction term, so the ETDRK4 coefficients stay fixed.
    derivatives=False skips the time derivatives (returned as None), half the memory."""
    beta_path = np.broadcast_to(np.asarray(beta, dtype=np.float64), (nt,))
    x = np.linspace(-L / 2, L / 2, n + 1)[:-1]
    y = x.copy()
    X, Y = np.meshgrid(x, y)  # array[i, j] = f(x_j, y_i), same as MATLAB meshgrid
    t = np.linspace(0.0, T, nt)

    k = 2 * np.pi / L * fft.fftfreq(n, d=1.0 / n)  # rows (y)
    kr = 2 * np.pi / L * fft.rfftfreq(n, d=1.0 / n)  # cols (x), real FFT half-plane
    K2 = k[:, None] ** 2 + kr[None, :] ** 2
    Lin = -np.stack([d1 * K2, d2 * K2])  # diffusion, diagonal in Fourier space

    h = (t[1] - t[0]) / substeps
    E, E2, Q, f1, f2, f3 = etdrk4_coefficients(Lin, h)

    def to_phys(uvh):
        return fft.irfft2(uvh, s=(n, n), axes=(-2, -1))

    def N(uvh, beta):
        return fft.rfft2(reaction(to_phys(uvh), beta), axes=(-2, -1))

    r = np.sqrt(X**2 + Y**2)
    theta = np.angle(X + 1j * Y) + phase
    uv0 = np.stack([np.tanh(r) * np.cos(theta - r), np.tanh(r) * np.sin(theta - r)])
    uvh = fft.rfft2(uv0, axes=(-2, -1))

    states = np.empty((nt, 2, n, n), dtype=dtype)
    derivs = np.empty((nt, 2, n, n), dtype=dtype) if derivatives else None

    def record(i, uvh):
        uv = to_phys(uvh)
        states[i] = uv
        if derivatives:
            derivs[i] = to_phys(Lin * uvh) + reaction(uv, float(beta_path[i]))

    record(0, uvh)
    t0 = time.perf_counter()
    for i in range(1, nt):
        beta_i = float(beta_path[i - 1])
        for _ in range(substeps):
            Nu = N(uvh, beta_i)
            a = E2 * uvh + Q * Nu
            Na = N(a, beta_i)
            b = E2 * uvh + Q * Na
            Nb = N(b, beta_i)
            c = E2 * a + Q * (2 * Nb - Nu)
            Nc = N(c, beta_i)
            uvh = E * uvh + f1 * Nu + 2 * f2 * (Na + Nb) + f3 * Nc
        record(i, uvh)
        if i % 1000 == 0:
            print(f"  t = {t[i]:7.2f}  ({i}/{nt - 1} steps, {time.perf_counter() - t0:.0f}s)")
    if not np.all(np.isfinite(states)):
        raise RuntimeError("solution blew up; increase --substeps")
    return t, x, y, states, derivs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DATA / "reaction_diffusion.h5")
    p.add_argument("--n", type=int, default=100, help="grid points per dimension")
    p.add_argument("--L", type=float, default=20.0, help="domain side length")
    p.add_argument("--T", type=float, default=500.0, help="final time")
    p.add_argument("--nt", type=int, default=10000, help="number of snapshots (nt - 1 steps)")
    p.add_argument("--d1", type=float, default=0.1)
    p.add_argument("--d2", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--substeps", type=int, default=5, help="ETDRK4 steps per snapshot interval")
    p.add_argument("--train-end", type=float, default=400.0)
    p.add_argument("--val-end", type=float, default=450.0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = p.parse_args()

    print(f"Solving on a {args.n}x{args.n} grid, t in [0, {args.T}] with {args.nt - 1} steps ...")
    t, x, y, states, derivs = solve(
        args.n, args.L, args.T, args.nt, args.d1, args.d2, args.beta, args.substeps, np.dtype(args.dtype)
    )

    split = np.where(t <= args.train_end, 0, np.where(t <= args.val_end, 1, 2)).astype(np.int8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    chunks = (min(64, args.nt), args.n, args.n)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("t", data=t)
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        for name, arr in [("u", states[:, 0]), ("v", states[:, 1]), ("du_dt", derivs[:, 0]), ("dv_dt", derivs[:, 1])]:
            f.create_dataset(name, data=arr, chunks=chunks)
        f.create_dataset("split", data=split)
        f.attrs.update(
            equations="u_t = (1-A^2)u + beta*A^2*v + d1*lap(u); v_t = -beta*A^2*u + (1-A^2)v + d2*lap(v); A^2 = u^2+v^2",
            initial_condition="u0 = tanh(r)cos(theta - r), v0 = tanh(r)sin(theta - r)",
            boundary="periodic",
            layout="fields are (time, y, x): u[k, i, j] = u(x[j], y[i], t[k])",
            split_labels="0 = train (t <= train_end), 1 = val, 2 = test",
            solver="pseudo-spectral (rfft2) + ETDRK4",
            d1=args.d1, d2=args.d2, beta=args.beta, L=args.L, n=args.n,
            substeps=args.substeps, train_end=args.train_end, val_end=args.val_end,
            reference="Champion, Lusch, Kutz & Brunton, PNAS 116(45), 2019",
        )

    counts = np.bincount(split, minlength=3)
    print(f"Wrote {args.out}  (u range [{states[:, 0].min():.3f}, {states[:, 0].max():.3f}])")
    print(f"  snapshots: train {counts[0]}, val {counts[1]}, test {counts[2]}")


if __name__ == "__main__":
    main()
