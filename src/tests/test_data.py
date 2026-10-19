import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from ..data import PCA, load_observations, load_trajectories
from ..data.reaction_diffusion import load_reaction_diffusion, make_batches, window_trajectories
from ._reference import run_tests, use_float64

use_float64()
DUFFING = Path(__file__).resolve().parents[2] / "test_data" / "data" / "duffing" / "duffing_discovery_low_noise_low_data.csv"


def test_load_duffing_observations():
    if not DUFFING.exists():
        raise unittest.SkipTest(f"{DUFFING} is missing")
    data = load_observations(str(DUFFING), trajectory=0, state_columns=["x1", "x2"], time_column="t",
                             truth_columns=["x1_true", "x2_true"], obs_std_columns=["x1_obs_std", "x2_obs_std"])
    assert data["t"].shape == (32,) and data["y"].shape == (32, 2) and data["obs_std"].shape == (2,)
    torch.testing.assert_close(data["x_true"][0], torch.tensor([3.0, 2.0]))
    assert load_trajectories(str(DUFFING), ["x1", "x2"]).shape[1:] == (32, 2)


def test_pca_reconstructs_low_rank_data_and_reloads():
    np.random.seed(0)
    torch.manual_seed(0)
    y = torch.randn(200, 2) @ torch.randn(2, 30) + 0.001 * torch.randn(200, 30)
    pca, z = PCA.create(y, percent_cutoff=0.999, max_evecs=10)
    assert z.shape == (200, 2)
    assert (pca.decode(pca.encode(y)) - y).abs().max() < 0.01
    reloaded = PCA()
    reloaded.load_state_dict(pca.state_dict())
    torch.testing.assert_close(reloaded.encode(y), pca.encode(y))


def test_reaction_diffusion_windows_and_batches():
    t = np.linspace(0, 1, 6)
    u = np.arange(2 * 6 * 2 * 2, dtype=np.float32).reshape(2, 6, 2, 2)  # 2 trajectories, 6 steps, 2x2 grid
    beta = np.array([0.5, 1.5], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "fake_reaction_diffusion.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("t", data=t)
            f.create_dataset("u", data=u)
            f.create_dataset("beta", data=beta)
        t_loaded, u_loaded, mu_none = load_reaction_diffusion(path, field="u")
        t_loaded, u_loaded, mu_loaded = load_reaction_diffusion(path, field="u", param="beta")
    assert t_loaded.shape == (6,) and u_loaded.shape == (2, 6, 2, 2)
    assert mu_none is None
    torch.testing.assert_close(mu_loaded, torch.as_tensor(beta, dtype=torch.get_default_dtype()).unsqueeze(-1))

    t_windows, x_windows, mu_windows = window_trajectories(t_loaded, u_loaded, length=3, mu=mu_loaded)
    assert t_windows.shape == (4, 3) and x_windows.shape == (4, 3, 4) and mu_windows.shape == (4, 1)
    # trajectory 0's windows come first, then trajectory 1's, each pair non-overlapping
    for row, (traj, start) in enumerate([(0, 0), (0, 3), (1, 0), (1, 3)]):
        torch.testing.assert_close(t_windows[row], t_loaded[start:start + 3])
        torch.testing.assert_close(x_windows[row], u_loaded[traj, start:start + 3].flatten(1))
        torch.testing.assert_close(mu_windows[row], mu_loaded[traj])

    for n_groups, size in ((1, 4), (2, 2), (4, 1)):
        batches = make_batches(t_windows, x_windows, n_groups, mu_windows)
        assert len(batches) == n_groups
        assert all(b["t"].shape == (size, 3) and b["x"].shape == (size, 3, 4) and b["mu"].shape == (size, 1)
                  for b in batches)

    # without mu_windows, batches default to the usual empty mu (make_batch's own default)
    assert make_batches(t_windows, x_windows, 1)[0]["mu"].shape == (4, 0)

    raised = False
    try:
        make_batches(t_windows, x_windows, n_groups=3)  # 4 windows do not split evenly into 3 groups
    except ValueError:
        raised = True
    assert raised

    # one window per trajectory: the times must not share memory, or torch.func.jvp in the posteriors fails
    t_whole, _, _ = window_trajectories(t_loaded, u_loaded, length=6)
    assert t_whole.shape == (2, 6) and t_whole.stride(0) != 0
    torch.func.jvp(lambda t: t.exp(), (t_whole,), (torch.ones_like(t_whole),))


def test_load_trajectories_with_times():
    if not DUFFING.exists():
        raise unittest.SkipTest(f"{DUFFING} is missing")
    t, x = load_trajectories(str(DUFFING), ["x1", "x2"], time_column="t")
    assert t.shape == (32,) and x.shape[1:] == (32, 2)
    torch.testing.assert_close(t, torch.linspace(0, 20, 32))


def test_reaction_diffusion_constant_beta_path_matches_scalar():
    """A constant beta path gives the scalar-beta solution bit for bit, so the existing datasets stay reproducible."""
    from test_data.scripts.reaction_diffusion.generate_reaction_diffusion import solve
    _, _, _, a, da = solve(16, 20.0, 1.0, 5, beta=0.9)
    _, _, _, b, db = solve(16, 20.0, 1.0, 5, beta=np.full(5, 0.9))
    _, _, _, c, dc = solve(16, 20.0, 1.0, 5, beta=0.9, derivatives=False)
    assert np.array_equal(a, b) and np.array_equal(da, db) and np.array_equal(a, c) and dc is None


def test_reaction_diffusion_solver_commutes_with_a_phase_rotation():
    """solve(phase=a) = exp(i a) solve(phase=0), state and time derivative: with d1 = d2 the lambda-omega system has
    this symmetry, which makes the phase-rate forcing of multimodal_reaction_test an exact rotation of the unforced
    field, w = exp(-i Gamma(t)) w0."""
    from test_data.scripts.reaction_diffusion.generate_reaction_diffusion import solve
    a = 0.7
    _, _, _, s0, d0 = solve(16, 20.0, 1.0, 5, beta=0.9)
    _, _, _, sa, da = solve(16, 20.0, 1.0, 5, beta=0.9, phase=a)
    for x0, xa in [(s0, sa), (d0, da)]:
        np.testing.assert_allclose(xa[:, 0] + 1j * xa[:, 1], np.exp(1j * a) * (x0[:, 0] + 1j * x0[:, 1]), atol=1e-12)


def test_merge_modalities_puts_each_modality_on_the_union_of_times():
    """Two modalities at their own times: union grid, a shared time in one step, mask only where observed, padding
    with the last time and mask 0."""
    from ..data.multimodal import merge_modalities
    times = [[torch.tensor([0.0, 2.0]), torch.tensor([1.0])],  # modality 0 (2 columns), trajectories 0 and 1
             [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0])]]  # modality 1 (1 column)
    values = [[torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0]])],
              [torch.tensor([[7.0], [8.0], [9.0]]), torch.tensor([[10.0]])]]
    t, x, mask = merge_modalities(times, values)
    torch.testing.assert_close(t, torch.tensor([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]]))
    torch.testing.assert_close(mask[0], torch.tensor([[1.0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1]]))
    torch.testing.assert_close(x[0], torch.tensor([[1.0, 2, 0], [0, 0, 7], [3, 4, 8], [0, 0, 9]]))
    torch.testing.assert_close(mask[1], torch.tensor([[1.0, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    torch.testing.assert_close(x[1, 0], torch.tensor([5.0, 6, 10]))


def test_load_multimodal_with_a_modality_missing_in_some_trajectories():
    """count 0: the trajectory lacks the modality, all its columns stay masked; no mu dataset gives P = 0."""
    from ..data.multimodal import load_multimodal
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.h5"
        with h5py.File(path, "w") as f:
            f["A/t"], f["A/count"] = np.array([[1.0, 3.0], [np.nan, np.nan]]), np.array([2, 0])
            f["A/y"] = np.array([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]])
            f["B/t"], f["B/count"] = np.array([[2.0, np.nan], [0.5, 1.5]]), np.array([1, 2])
            f["B/y"] = np.array([[[5.0], [0.0]], [[6.0], [7.0]]])
        (batch,), columns = load_multimodal(str(path), ["A", "B"])
    assert columns == {"A": slice(0, 2), "B": slice(2, 3)} and batch["mu"].shape == (2, 0)
    torch.testing.assert_close(batch["t"], torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 1.5]]))
    torch.testing.assert_close(batch["x"][0], torch.tensor([[1.0, 2, 0], [0, 0, 5], [3, 4, 0]]))
    torch.testing.assert_close(batch["mask"][1], torch.tensor([[0.0, 0, 1], [0, 0, 1], [0, 0, 0]]))


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
