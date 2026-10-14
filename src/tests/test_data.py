from pathlib import Path
import unittest

import numpy as np
import torch

from ..data import PCA, MultiEvenlySpacedTensors, MultiTemporalSampler, load_observations, load_trajectories
from ._reference import run_tests, use_float64, visde_module

use_float64()
DUFFING = Path(__file__).resolve().parents[2] / "test_data" / "data" / "duffing_discovery_low_noise_low_data.csv"


def test_load_duffing_observations():
    if not DUFFING.exists():
        raise unittest.SkipTest(f"{DUFFING} is missing")
    data = load_observations(str(DUFFING), trajectory=0, state_columns=["x1", "x2"], time_column="t",
                             truth_columns=["x1_true", "x2_true"], obs_std_columns=["x1_obs_std", "x2_obs_std"])
    assert data["t"].shape == (32,) and data["y"].shape == (32, 2) and data["obs_std"].shape == (2,)
    torch.testing.assert_close(data["x_true"][0], torch.tensor([3.0, 2.0]))
    assert load_trajectories(str(DUFFING), ["x1", "x2"]).shape[1:] == (32, 2)


def _window_data():
    t = torch.linspace(0, 1, 11).repeat(3, 1)
    y = torch.randn(3, 11, 4)
    return torch.randn(3, 2), t, y, torch.randn(3, 11, 1)


def test_windows_match_rom_visde():
    mu, t, y, f = _window_data()
    reference = visde_module("data")
    ours = MultiEvenlySpacedTensors(mu, t, y, f, num_window=3)
    theirs = reference.MultiEvenlySpacedTensors(mu, t, y, f, num_window=3)
    assert len(ours) == len(theirs)
    for k in (0, 5, len(ours) - 1):
        for a, b in zip(ours[k], theirs[k]):
            torch.testing.assert_close(a, b)
    batches = list(MultiTemporalSampler(ours, 4, generator=torch.Generator().manual_seed(0)))
    expected = list(reference.MultiTemporalSampler(theirs, 4, generator=torch.Generator().manual_seed(0)))
    assert batches == expected


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


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
