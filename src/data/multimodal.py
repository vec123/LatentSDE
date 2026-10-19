"""
Multimodal observations, each modality at its own irregular times, merged into the one batch dict of
src/latent/model.py.

merge_modalities puts the modalities of a trajectory on the union of their times, one block of columns per
modality, with the mask 1 only where that modality was observed: a step with only a biomarker sample has the image
columns masked, and the other way round. Rows are padded to the longest trajectory with mask 0 and the last time
repeated, as make_batch expects. The posterior then reads the mask per column, and the reconstruction counts only
the observed values.

load_multimodal reads a file with one group per modality, the values as they are (e.g.
test_data/data/simple_multimodal_test, where some trajectories lack a modality: count 0, all its columns masked).
load_multimodal_reaction reads test_data/data/multimodal_reaction_test's observations_<split>.h5 this way, each
modality encoded by its own PCA (test_data/scripts/multimodal_reaction_test/fit_pca.py: <pca_dir>/<modality>/
model.pt), with beta as mu. Encoding the images to their PCA codes also balances the modalities in the loss,
which divides by the number of observed values: 30 values per image, 1 per biomarker sample.
"""
from typing import Dict, List, Sequence, Tuple

import h5py
import torch
from torch import Tensor

from ..latent.batch import make_batch
from .pca import PCA

MODALITIES = ("imaging", "biomarker")


def merge_modalities(times: Sequence[Sequence[Tensor]], values: Sequence[Sequence[Tensor]]
                     ) -> Tuple[Tensor, Tensor, Tensor]:
    """
    times[m][b] (n_mb,) and values[m][b] (n_mb, D_m) of modality m in trajectory b -> t (B, N), x (B, N, D) and
    mask (B, N, D), D = sum of D_m, the columns of modality m after those of the modalities before it.
    """
    dims = [values[m][0].shape[-1] for m in range(len(values))]
    offsets = [sum(dims[:m]) for m in range(len(dims))]
    grids = [torch.cat([times[m][b] for m in range(len(times))]).unique() for b in range(len(times[0]))]
    B, N, D = len(grids), max(len(g) for g in grids), sum(dims)
    ref = values[0][0]
    t = ref.new_zeros(B, N)
    x, mask = ref.new_zeros(B, N, D), ref.new_zeros(B, N, D)
    for b, grid in enumerate(grids):
        t[b, :len(grid)], t[b, len(grid):] = grid, grid[-1]
        for m, (off, d) in enumerate(zip(offsets, dims)):
            k = torch.searchsorted(grid, times[m][b])
            x[b, k, off:off + d] = values[m][b]
            mask[b, k, off:off + d] = 1
    return t, x, mask


def batches(times, values, names: Sequence[str], mu: Tensor = None, n_groups: int = 1
            ) -> Tuple[List[dict], Dict[str, slice]]:
    """
    merge_modalities, then the batch dicts (the trajectories split into n_groups batches of equal size, the last
    one taking the rest) and the columns of each modality, {name: slice}. mu (n_traj, P), or None for P = 0.
    """
    t, x, mask = merge_modalities(times, values)
    dims = [v[0].shape[-1] for v in values]
    columns = {name: slice(sum(dims[:m]), sum(dims[:m + 1])) for m, name in enumerate(names)}
    mu = x.new_zeros(t.shape[0], 0) if mu is None else mu
    size = t.shape[0] // n_groups
    bounds = [(g * size, (g + 1) * size if g < n_groups - 1 else t.shape[0]) for g in range(n_groups)]
    return [make_batch(t[a:b], x[a:b], mask[a:b], mu[a:b]) for a, b in bounds], columns


def load_multimodal(path: str, modalities: Sequence[str], n_groups: int = 1) -> Tuple[List[dict], Dict[str, slice]]:
    """
    The batch dicts and columns (see batches) of an h5 file with one group per modality: <name>/t (n_traj, max_n),
    <name>/y (n_traj, max_n, D) and <name>/count (n_traj,), count 0 where a trajectory lacks the modality (e.g.
    test_data/data/simple_multimodal_test); mu from a dataset "mu" (n_traj, P) if the file has one. No encoding:
    the values go into the batch as they are.
    """
    dtype = torch.get_default_dtype()
    times, values = [], []
    with h5py.File(path, "r") as f:
        for name in modalities:
            count, t, y = f[f"{name}/count"][:], f[f"{name}/t"][:], f[f"{name}/y"][:]
            times.append([torch.as_tensor(t[b, :n], dtype=dtype) for b, n in enumerate(count)])
            values.append([torch.as_tensor(y[b, :n], dtype=dtype) for b, n in enumerate(count)])
        mu = torch.as_tensor(f["mu"][:], dtype=dtype) if "mu" in f else None
    return batches(times, values, modalities, mu, n_groups)


def load_multimodal_reaction(path: str, pca_dir: str, n_groups: int = 1) -> Tuple[List[dict], Dict[str, slice]]:
    """
    The batch dicts and columns (see batches) of one observations_<split>.h5, each modality encoded by its PCA.
    """
    dtype = torch.get_default_dtype()
    times, values = [], []
    with h5py.File(path, "r") as f:
        mu = torch.as_tensor(f["beta"][:], dtype=dtype).unsqueeze(-1)
        for name in MODALITIES:
            pca = PCA()
            pca.load_state_dict(torch.load(f"{pca_dir}/{name}/model.pt", weights_only=True))
            pca = pca.to(dtype)
            count, t = f[f"{name}/count"][:], f[f"{name}/t"]
            raw = f[f"{name}/u"] if name == "imaging" else f[f"{name}/y"]
            times.append([torch.as_tensor(t[b, :n], dtype=dtype) for b, n in enumerate(count)])
            with torch.no_grad():
                values.append([pca.encode(torch.as_tensor(raw[b, :n], dtype=dtype).reshape(n, -1))
                               for b, n in enumerate(count)])
    return batches(times, values, MODALITIES, mu, n_groups)
