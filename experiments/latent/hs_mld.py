"""
The HS data as paths of frozen MLD latents (experiments/mld_ae, Bounoua et al., Multi-modal Latent Diffusion).

Stage 1 is done: one deterministic autoencoder per modality, trained alone (runs/mld_ae/<modality>/model.pt). Here each
visit is encoded by them, z_m = (e_m(x_m) - z_mean_m) / z_std_m, and the latents of the modalities present at the
visit are concatenated, in the order of MODALITIES, into one vector z = [z_exam, z_photo, z_ultrasound, z_serum] (30
values for the latent sizes 8, 10, 4, 8). A block of a modality that was not measured at the visit has mask 0. The
latent SDE then models the path of z over the visits, in place of MLD's diffusion model (the tokens are z itself).

Visits are the dates of (cohort, patient) at which at least one modality was measured; time is in years since the
first of them. The patients are split by runs/mld_ae/split.csv (the split the autoencoders were trained with; use no
other, or the test patients of one stage are train patients of the other). The held-out cohorts are loaded whole
(split "heldout_surgical" or "heldout_community"); note that the autoencoders of stage 1 did see 80% of their
patients' rows (never their labels), because split.csv splits every cohort. The static covariates mu and the
treatment u are those of src/data/hs.py.

decode_block(...) maps a block of z back to the data units with the frozen decoder.
"""
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from experiments.mld_ae import data as mld_data
from experiments.mld_ae import model as mld_model
from src.data import hs
from src.latent.batch import make_batch

MODALITIES = ("exam", "photo", "ultrasound", "serum")
KEYS = ["cohort", "patient_id", "date"]
_CACHE: dict = {}


def _autoencoder(modality: str, mld_dir: str):
    key = ("ae", modality, mld_dir)
    if key not in _CACHE:
        _CACHE[key] = mld_model.load(os.path.join(mld_dir, modality, "model.pt"))
    return _CACHE[key]


def latent_sizes(modalities: Sequence[str] = MODALITIES, mld_dir: str = "runs/mld_ae") -> List[int]:
    return [_autoencoder(m, mld_dir)[0].latent_dim for m in modalities]


def raw_columns(modality: str, width: int) -> Tuple[List[str], List[str]]:
    """The names of the raw value columns and of the below-limit flag columns of a modality in the visit table."""
    return [f"{modality}_x{j}" for j in range(width)], [f"{modality}_b{j}" for j in range(width)]


def _visits(modalities: Sequence[str], mld_dir: str) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """One row per (cohort, patient, date) with the standardized latent of each modality (NaN where not measured)."""
    key = ("visits", tuple(modalities), mld_dir)
    if key in _CACHE:
        return _CACHE[key]
    split = pd.read_csv(os.path.join(mld_dir, "split.csv"))
    frame, blocks = None, {}
    for modality in modalities:
        table, _ = mld_data.load(modality, split)
        model, _ckpt = _autoencoder(modality, mld_dir)
        x, blq = mld_data.arrays(table, model.columns)
        with torch.no_grad():
            z = (model.encode(torch.as_tensor(x), torch.as_tensor(blq)) - model.z_mean) / model.z_std
        names = [f"{modality}_z{j}" for j in range(z.shape[1])]
        part = table[KEYS].copy()
        part["date"] = pd.to_datetime(part["date"])
        part[names] = z.numpy()
        raw_x, raw_b = raw_columns(modality, len(model.columns))
        part[raw_x], part[raw_b] = x, blq.astype(np.float32)  # the rows in the model space of mld_ae (NaN where missing)
        part = part.groupby(KEYS, as_index=False)[names + raw_x + raw_b].mean()  # several sources on one date: their mean
        frame = part if frame is None else frame.merge(part, on=KEYS, how="outer")
        blocks[modality] = names
    _CACHE[key] = (frame.sort_values(KEYS).reset_index(drop=True), blocks)
    return _CACHE[key]


def load_hs_mld(split: str, batch_size: int = 25, mld_dir: str = "runs/mld_ae", seed: int = 0,
                cohorts: Sequence[str] = hs.TRAIN_COHORTS, covariates: str = "all",
                modalities: Sequence[str] = MODALITIES, raw: bool = False) -> Tuple[List[dict], List[pd.DataFrame]]:
    """
    The batch dicts of one split ("train", "val", "test" of the training `cohorts`, or a held-out cohort) and, for each,
    a table (cohort, patient_id, date) of its padded rows (B * N; padding rows have a NaT date). x (B, N, D) holds
    the concatenated standardized latents (0 where a block is missing), mask (B, N, D) is 1 over the blocks that are
    present; mu the static covariates (covariates "all": hs.DIM_MU_ALL values), u the treatment (hs.DIM_U classes).
    With raw=True the batch also has raw_<modality> (B, N, D_m) in the model space of mld_ae, NaN where missing, and
    blq_<modality> (the below-limit flags), for the end-to-end model (experiments/latent/e2e.py).
    """
    frame, blocks = _visits(modalities, mld_dir)
    columns = [c for m in modalities for c in blocks[m]]
    parts = pd.read_csv(os.path.join(mld_dir, "split.csv")).set_index("patient_id")["split"]
    frame = frame[frame["cohort"].isin(set(cohorts) | set(hs.HELDOUT))]

    rows = hs._rows(hs.ROOT)[KEYS + ["current_treatment"]]
    frame = frame.merge(rows, on=KEYS, how="left")
    first = frame.groupby("patient_id")["date"].min()
    if covariates != "all":
        raise ValueError("hs_mld: covariates must be 'all'")
    cov = hs._covariates_all(hs.ROOT, first)
    train_ids = frame.loc[frame["cohort"].isin(cohorts) & (frame["patient_id"].map(parts) == "train"), "patient_id"].unique()
    ref = cov.loc[train_ids, hs.CONTINUOUS]
    cov[hs.CONTINUOUS] = (cov[hs.CONTINUOUS] - ref.mean()) / ref.std()

    treatment = {name: k for k, name in enumerate(hs.TREATMENT_CLASSES)}
    treatment["bimekizumab"] = treatment["secukinumab"]
    wanted = (frame[frame["cohort"] == split] if split in hs.HELDOUT
              else frame[frame["cohort"].isin(cohorts) & (frame["patient_id"].map(parts) == split)])
    dtype = torch.get_default_dtype()
    rng = np.random.default_rng(seed + 1)
    batches, tables = [], []
    for cohort, group in wanted.groupby("cohort"):
        ids = group["patient_id"].unique()
        ids = ids[rng.permutation(len(ids))]
        for start in range(0, len(ids), batch_size):
            chunk = [group[group["patient_id"] == pid] for pid in ids[start:start + batch_size]]
            N, D = max(2, *(len(c) for c in chunk)), len(columns)
            t, x, mask = np.zeros((len(chunk), N)), np.zeros((len(chunk), N, D)), np.zeros((len(chunk), N, D))
            u = np.zeros((len(chunk), N, hs.DIM_U))
            extra = {}
            if raw:
                for m in modalities:
                    xn, bn = raw_columns(m, len(_autoencoder(m, mld_dir)[0].columns))
                    rx, rb = np.full((len(chunk), N, len(xn)), np.nan), np.zeros((len(chunk), N, len(xn)))
                    for b, c in enumerate(chunk):
                        rx[b, :len(c)], rb[b, :len(c)] = c[xn].to_numpy(float), np.nan_to_num(c[bn].to_numpy(float))
                    extra[f"raw_{m}"], extra[f"blq_{m}"] = torch.as_tensor(rx, dtype=dtype), torch.as_tensor(rb, dtype=dtype)
            table = pd.DataFrame({"cohort": cohort, "patient_id": np.repeat([c["patient_id"].iloc[0] for c in chunk], N),
                                  "date": pd.NaT})
            for b, c in enumerate(chunk):
                n = len(c)
                years = (c["date"] - c["date"].iloc[0]).dt.days.to_numpy() / 365.25
                v = c[columns].to_numpy(float)
                t[b, :n], t[b, n:] = years, years[-1]
                x[b, :n], mask[b, :n] = np.nan_to_num(v), np.isfinite(v)
                u[b, np.arange(n), c["current_treatment"].map(treatment).fillna(0).astype(int).to_numpy()] = 1.0
                u[b, n:] = u[b, n - 1]
                table.loc[b * N:b * N + n - 1, "date"] = c["date"].to_numpy()
            a = cov.loc[[c["patient_id"].iloc[0] for c in chunk]]
            mu = np.c_[a.fillna(0.0).to_numpy(), a.notna().to_numpy()].astype(float)
            batches.append({**make_batch(torch.as_tensor(t, dtype=dtype), torch.as_tensor(x, dtype=dtype),
                                         torch.as_tensor(mask, dtype=dtype), torch.as_tensor(mu, dtype=dtype),
                                         torch.as_tensor(u, dtype=dtype)), **extra})
            tables.append(table)
    return batches, tables


def decode_block(modality: str, z: np.ndarray, mld_dir: str = "runs/mld_ae") -> Tuple[np.ndarray, List[str]]:
    """The standardized latents z (n, latent_dim) of a modality -> its values in the data units (n, D) and the names
    of the columns, with the frozen decoder."""
    model, _ = _autoencoder(modality, mld_dir)
    with torch.no_grad():
        raw = torch.as_tensor(z, dtype=torch.float32) * model.z_std + model.z_mean
        pred = model.predict(model.decode(raw)).numpy()
    return mld_data.to_units(pred, model.columns), [c.name for c in model.columns]
