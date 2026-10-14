from pathlib import Path
import unittest

import numpy as np
import torch

from ..data import PCA, load_observations, load_trajectories
from ._reference import run_tests, use_float64

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
