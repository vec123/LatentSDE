"""
Shared parts of the multimodal_reaction_test toy (test_data/scripts/multimodal_reaction_test): the hidden driver,
the irregular observation times, and features of a spiral frame.

The lambda-omega spiral of test_data/scripts/reaction_diffusion, w = u + i v, with a constant beta per trajectory
and a hidden, spatially uniform phase-rate forcing gamma(t), a stationary OU process (time scale TAU, std
GAMMA_STD):

    w_t = (1 - |w|^2) w - i beta |w|^2 w + d lap(w) - i gamma(t) w,   i.e.  u_t += gamma v,  v_t += -gamma u.

With d1 = d2 the unforced system does not change under w -> exp(i alpha) w, so the forced solution is exactly
w = exp(-i Gamma(t)) w0, Gamma = int gamma dt, w0 the unforced solution: the forcing turns the whole field and
leaves its shape alone. So an image shows the rotation phase (modulo 2 pi) and nothing of gamma or its history:
the leak of the first design (a hidden beta(t), whose history the spiral arms record at the slow group velocity
of the waves, about 0.2 space units per time unit, so that images alone fixed the turns between visits in 85% of
gaps) is exactly zero here. The rotation rate is omega(beta) + gamma(t).

The rotation phase comes from the first angular Fourier mode on a circle of radius r about the centre,
c1(r) = mean over theta of field(r, theta) exp(-i theta). For w = A(r) exp(i(theta - psi(r) - phi)),
c1 = A exp(-i(psi + phi)) from w, and half of that from u alone (the conjugate half of u = Re w goes into the
-1 mode). So phi = -arg c1(R0), up to a constant, from w or from u. Features that must not show the phase use
only |c1(r)|, the relative phase of c1 on neighbouring circles (the local wavenumber), and the power spectrum.
"""
import numpy as np

BETA_RANGE = (0.5, 1.5)  # the range reaction_diffusion_var_params already uses
TAU, GAMMA_STD = 5.0, 0.5

R0 = 2.0  # radius of the circle the phase is read on
RADII = np.arange(1.0, 9.51, 0.5)  # circles for the features, inside the [-10, 10]^2 domain
N_THETA = 64

# Defaults of sample_modalities.py; check_leak.py uses the same. With them the integral of gamma over one imaging
# gap has a std of about one turn, so the gap length alone fixes the turns between two visits in only about 40% of
# gaps, and the biomarker in nearly all.
IMAGE_GAP_MEAN, IMAGE_GAP_SHAPE, IMAGE_NOISE = 20.0, 8.0, 0.05
BIO_GAP_MEAN, BIO_GAP_SHAPE, BIO_NOISE = 2.0, 4.0, 0.05


def ou_path(t: np.ndarray, rng: np.random.Generator, tau: float = TAU, std: float = GAMMA_STD) -> np.ndarray:
    """A stationary OU process on the times t, exact discretization, started from its stationary law."""
    a = np.exp(-np.diff(t) / tau)
    noise = std * np.sqrt(1 - a ** 2) * rng.standard_normal(len(t) - 1)
    x = np.empty(len(t))
    x[0] = std * rng.standard_normal()
    for k in range(len(t) - 1):
        x[k + 1] = a[k] * x[k] + noise[k]
    return x


def integral(t: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Gamma (..., nt) = int_0^t gamma, exact for gamma linear between the times t."""
    steps = 0.5 * (gamma[..., 1:] + gamma[..., :-1]) * np.diff(t)
    return np.concatenate([np.zeros_like(gamma[..., :1]), np.cumsum(steps, -1)], -1)


def rotate(u0: np.ndarray, v0: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    """The forced field u = Re(exp(-i Gamma) (u0 + i v0)) of frames (..., n, n), with Gamma (...) per frame."""
    c, s = np.cos(Gamma)[..., None, None], np.sin(Gamma)[..., None, None]
    return (c * u0 + s * v0).astype(u0.dtype)


def truth_frames(f, i: int, k=slice(None)) -> np.ndarray:
    """The forced u of trajectory i at the frame indices k, from an open truth_<split>.h5."""
    Gamma = f["Gamma"][i, ::int(f.attrs["frame_stride"])][k]
    return rotate(f["u0"][i, k], f["v0"][i, k], Gamma)


def renewal_times(T: float, mean_gap: float, shape: float, rng: np.random.Generator) -> np.ndarray:
    """Times in [0, T] with gamma gaps (mean mean_gap; shape sets the regularity, std = mean / sqrt(shape)),
    the first one uniform in [0, mean_gap)."""
    times = [rng.uniform(0.0, mean_gap)]
    while True:
        nxt = times[-1] + rng.gamma(shape, mean_gap / shape)
        if nxt > T:
            break
        times.append(nxt)
    return np.array([s for s in times if s <= T])


def angular_mode_matrix(x: np.ndarray, radii: np.ndarray = RADII, n_theta: int = N_THETA) -> np.ndarray:
    """K (n_r, n * n) complex, with c1 = frames.reshape(-1, n * n) @ K.T: bilinear interpolation on n_theta points
    per circle (periodic grid x, the same for y), then the mean of exp(-i theta) times the values."""
    n, dx = len(x), x[1] - x[0]
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    K = np.zeros((len(radii), n * n), dtype=complex)
    for j, r in enumerate(radii):
        fx, fy = (r * np.cos(theta) - x[0]) / dx, (r * np.sin(theta) - x[0]) / dx
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        wx, wy = fx - ix, fy - iy
        e = np.exp(-1j * theta) / n_theta
        for dy, dxi, w in [(0, 0, (1 - wy) * (1 - wx)), (0, 1, (1 - wy) * wx), (1, 0, wy * (1 - wx)), (1, 1, wy * wx)]:
            np.add.at(K[j], ((iy + dy) % n) * n + (ix + dxi) % n, w * e)
    return K


def angular_mode(frames: np.ndarray, K: np.ndarray) -> np.ndarray:
    """c1 (..., n_r) of frames (..., n, n), real (u) or complex (w)."""
    lead = frames.shape[:-2]
    return (frames.reshape(-1, K.shape[1]) @ K.T).reshape(*lead, K.shape[0])


def unwrapped_phase(c1_r0: np.ndarray) -> np.ndarray:
    """phi (..., n_frames) from c1 at R0 along the last axis (time): -arg c1, unwrapped."""
    return -np.unwrap(np.angle(c1_r0), axis=-1)


def invariant_features(frames: np.ndarray, K: np.ndarray, n_bins: int = 30) -> np.ndarray:
    """(n_frames, n_features) that do not change when the spiral rotates: |c1(r)|, cos and sin of the relative phase
    of c1 on neighbouring circles, and the log radially averaged power spectrum of the frame."""
    c1 = angular_mode(frames, K)
    rel = c1[:, 1:] * np.conj(c1[:, :-1])
    rel = rel / (np.abs(rel) + 1e-12)
    n = frames.shape[-1]
    power = np.abs(np.fft.rfft2(frames)) ** 2
    k = np.hypot(np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(n)[None, :])
    bins = np.minimum((k / 0.5 * n_bins).astype(int), n_bins - 1).ravel()
    counts = np.bincount(bins, minlength=n_bins)
    spectrum = np.stack([np.bincount(bins, p.ravel(), minlength=n_bins) for p in power]) / np.maximum(counts, 1)
    return np.concatenate([np.abs(c1), rel.real, rel.imag, np.log(spectrum + 1e-8)], 1)
