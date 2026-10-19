"""
The tabular modalities of test_data/data/multimodal (hidradenitis suppurativa, the generator is
test_data/src/multimodal) in the one batch dict of src/latent/model.py.

Two training cohorts (specialist_imaging, biomarker_study) and two held-out ones (heldout_surgical: pre-op exam,
ultrasound and serum; heldout_community: one survey with exam, photo and a point-of-care CRP) give four modalities, each a block of columns (BLOCKS):

    exam        specialist visits and biomarker samples (counts, IHS4, Hurley stage, DLQI, pain, 5 sites)
    photo       specialist visits (lesion area, erythema, scarring, 8 image embeddings)
    ultrasound  specialist visits, not site C (fluid, tunnels, collection size, dermal thickness, Doppler, SOS-HS)
    serum       specialist CRP and biomarker panel (CRP, ESR, IL-6, IL-1b, IL-17A, TNF-a, calprotectin, NLR)

A row is one visit or sample date; a value that is not there has mask 0, so a biomarker sample has only the short
exam and the serum block. Counts and concentrations are log(1 + x), the lesion area and the collection size are
log x; every column is then standardized with the mean and std of the training patients. The below-limit flags
(blq) are not used: such a value stays at the limit the generator wrote. Time is in years since the first row
of the patient. The patients of each training cohort are split by seed into train, val and test (70/15/15); the
held-out cohorts are loaded whole (split "heldout_surgical" or "heldout_community"), and never used for the standardization.
The static covariates mu (sex, BMI, smoking status, from each cohort's patient table) go into the batch's mu: the
prior SDE, the decoder, the posterior and the prior of the static latent read them. The forcing u(t) is the treatment of the visit as one of 7 classes (NO_TREATMENT, the first, is "none or unknown").
"""
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from ..latent.batch import make_batch

ROOT = "test_data/data/multimodal"
SITES = ["site_axillary", "site_inframammary", "site_inguinal", "site_gluteal_perianal", "site_atypical"]
# column -> transform ("log1p", "log" or "id")
BLOCKS: Dict[str, Dict[str, str]] = {
    "exam": {**{c: "log1p" for c in ["nodules", "abscesses", "draining_tunnels", "an_count", "ihs4"]},
             **{c: "id" for c in ["hurley_stage", "dlqi", "pain_nrs", *SITES]}},
    "photo": {"lesion_area_cm2": "log", "erythema_index": "id", "scarring_score": "id",
              **{f"img_emb_{i}": "id" for i in range(8)}},
    "ultrasound": {"us_fluid_collections": "log1p", "us_tunnels": "log1p", "us_max_collection_mm": "log",
                   "us_dermal_thickness_mm": "id", "us_doppler_grade": "id", "us_sos_hs_stage": "id"},
    "serum": {"crp_mg_l": "log1p", "esr_mm_h": "log1p", "il6_pg_ml": "log1p", "il1b_pg_ml": "log1p",
              "il17a_pg_ml": "log1p", "tnfa_pg_ml": "log1p", "calprotectin_ug_ml": "log1p", "nlr": "log1p"},
}
COLUMNS = [c for block in BLOCKS.values() for c in block]
BLOCK_SIZES = [len(block) for block in BLOCKS.values()]
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}
TRAIN_COHORTS = ("specialist_imaging", "biomarker_study")
HELDOUT = ("heldout_surgical", "heldout_community")
TREATMENT_CLASSES = ["none", "topical_clindamycin", "tetracycline", "clindamycin_rifampicin", "adalimumab",
                     "secukinumab", "short_antibiotic"]  # bimekizumab joins secukinumab (IL-17 drugs)
DIM_U = len(TREATMENT_CLASSES)
COVARIATES = ["female", "bmi", "smoker_current", "smoker_former", "age", "onset_age", "duration", "family_history",
              "phototype", "hx_t2d", "hx_ibd", "hx_depression", "hx_pcos", "hx_spondyloarthritis", "hx_acne",
              "hx_pilonidal", "prior_systemic_antibiotics", "prior_biologic", "prior_hs_surgery"]
CONTINUOUS = ["bmi", "age", "onset_age", "duration", "phototype"]
DIM_MU_ALL = 2 * len(COVARIATES)  # covariates="all": the values (0 where missing), then 1 where a value is there
DIM_MU = 5  # the static covariates mu: female, standardized BMI (0 if missing), BMI missing, smoker current, former


def _rows(root: str) -> pd.DataFrame:
    """One row per (cohort, patient, date), the columns of COLUMNS (NaN where not measured) and the treatment."""
    visits = pd.read_csv(f"{root}/specialist_imaging/visits.csv").rename(columns={"visit_date": "date"})
    samples = pd.read_csv(f"{root}/biomarker_study/samples.csv").rename(columns={"sample_date": "date"})
    preop = pd.read_csv(f"{root}/heldout_surgical/preop.csv").rename(columns={"preop_date": "date"})
    survey = pd.read_csv(f"{root}/heldout_community/participants.csv").rename(
        columns={"survey_date": "date", "participant_id": "patient_id", "poc_crp_mg_l": "crp_mg_l"})
    visits["cohort"], samples["cohort"] = "specialist_imaging", "biomarker_study"
    preop["cohort"], survey["cohort"] = "heldout_surgical", "heldout_community"
    survey["current_treatment"] = "none"
    rows = pd.concat([visits, samples, preop, survey], ignore_index=True)
    for col in COLUMNS:
        if col not in rows:
            rows[col] = np.nan
    rows["date"] = pd.to_datetime(rows["date"])
    rows = rows[rows[COLUMNS].notna().any(axis=1)]  # a date with no value at all carries no data (31 of 9654 rows)
    return rows[["cohort", "patient_id", "date", "current_treatment", *COLUMNS]].sort_values(
        ["cohort", "patient_id", "date"])


def _covariates(root: str) -> pd.DataFrame:
    """Per patient: female, bmi (NaN if missing), smoking status; of all four cohorts, indexed by patient_id."""
    parts = []
    for file, bmi in (("specialist_imaging/patients.csv", "bmi"), ("biomarker_study/patients.csv", "bmi"),
                      ("heldout_surgical/patients.csv", "bmi"),
                      ("heldout_community/participants.csv", "bmi_self_reported")):
        d = pd.read_csv(f"{root}/{file}").rename(columns={"participant_id": "patient_id", bmi: "bmi"})
        parts.append(d[["patient_id", "sex", "bmi", "smoking_status"]])
    d = pd.concat(parts).set_index("patient_id")
    d["female"] = (d["sex"] == "F").astype(float)
    return d


def _covariates_all(root: str, first: pd.Series) -> pd.DataFrame:
    """
    Every static covariate of the four cohorts, COVARIATES as columns, NaN where a cohort does not have one.
    first: the first date of each patient (the age is that year minus the birth year, where the table has a birth
    year). The onset age and the disease duration are given by the specialist and the biomarker tables only.
    """
    def smoking(d):
        return {"smoker_current": (d["smoking_status"] == "current").astype(float),
                "smoker_former": (d["smoking_status"] == "former").astype(float)}

    sp = pd.read_csv(f"{root}/specialist_imaging/patients.csv").set_index("patient_id")
    age = first.reindex(sp.index).dt.year - sp["birth_year"]
    spec = pd.DataFrame({"female": (sp.sex == "F").astype(float), "bmi": sp.bmi, **smoking(sp), "age": age,
                         "onset_age": sp.age_at_onset_reported, "duration": age - sp.age_at_onset_reported,
                         "family_history": sp.family_history_hs, "phototype": sp.fitzpatrick_phototype,
                         **{c: sp[c] for c in ("hx_t2d", "hx_ibd", "hx_depression", "hx_pcos", "hx_spondyloarthritis",
                                               "hx_acne", "hx_pilonidal")}})
    bm = pd.read_csv(f"{root}/biomarker_study/patients.csv").set_index("patient_id")
    bio = pd.DataFrame({"female": (bm.sex == "F").astype(float), "bmi": bm.bmi, **smoking(bm),
                        "age": bm.age_at_baseline, "onset_age": bm.age_at_baseline - bm.disease_duration_years,
                        "duration": bm.disease_duration_years, "family_history": bm.family_history_hs,
                        **{c: bm[c] for c in ("prior_systemic_antibiotics", "prior_biologic", "prior_hs_surgery")}})
    su = pd.read_csv(f"{root}/heldout_surgical/patients.csv").set_index("patient_id")
    sur = pd.DataFrame({"female": (su.sex == "F").astype(float), "bmi": su.bmi, **smoking(su),
                        "age": first.reindex(su.index).dt.year - su["birth_year"]})
    co = pd.read_csv(f"{root}/heldout_community/participants.csv").set_index("participant_id")
    com = pd.DataFrame({"female": (co.sex == "F").astype(float), "bmi": co.bmi_self_reported, **smoking(co),
                        "age": co.age, "hx_depression": co.hx_depression, "hx_t2d": co.hx_t2d})
    return pd.concat([spec, bio, sur, com]).reindex(columns=COVARIATES).astype(float)


def _transform(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    for j, kind in enumerate(kind for block in BLOCKS.values() for kind in block.values()):
        if kind == "log1p":
            out[:, j] = np.log1p(values[:, j])
        elif kind == "log":
            out[:, j] = np.log(values[:, j])
    return out


def split_patients(rows: pd.DataFrame, seed: int = 0) -> Dict[str, str]:
    """{patient_id: split} of the training cohorts, each cohort split on its own."""
    rng = np.random.default_rng(seed)
    out = {}
    for cohort in TRAIN_COHORTS:
        ids = rows.loc[rows["cohort"] == cohort, "patient_id"].unique()
        ids = ids[rng.permutation(len(ids))]
        n_train, n_val = int(SPLIT_FRACTIONS["train"] * len(ids)), int(SPLIT_FRACTIONS["val"] * len(ids))
        for k, pid in enumerate(ids):
            out[pid] = "train" if k < n_train else "val" if k < n_train + n_val else "test"
    return out


def load_hs(split: str, batch_size: int = 25, root: str = ROOT, seed: int = 0,
            cohorts: Sequence[str] = TRAIN_COHORTS, covariates: str = "basic") -> Tuple[List[dict], List[pd.DataFrame]]:
    """
    The batch dicts of one split ("train", "val", "test" of the training `cohorts`, or a held-out cohort) and, for
    each, a table (cohort, patient_id, date) of its padded rows (B * N, the padding rows marked by a NaT date), to
    join with the truth. Each batch holds the patients of one cohort, 25 at a time, so that the padding stays
    small. The standardization is that of the train patients of `cohorts`. covariates: "basic" (DIM_MU = 5: sex, BMI,
    smoking) or "all" (DIM_MU_ALL: every static covariate of COVARIATES, standardized the same way, each with a flag
    for "given", so that a cohort without one is not confused with a value).
    """
    rows = _rows(root)
    parts = rows["patient_id"].map(split_patients(rows, seed))
    in_cohorts = rows["cohort"].isin(cohorts)
    train = rows[in_cohorts & (parts == "train")]
    values = _transform(train[COLUMNS].to_numpy(float))
    mean, std = np.nanmean(values, 0), np.nanstd(values, 0)
    dtype = torch.get_default_dtype()
    rng = np.random.default_rng(seed + 1)
    all_cov = None
    if covariates == "all":
        all_cov = _covariates_all(root, rows.groupby("patient_id")["date"].min())
        ref = all_cov.loc[train["patient_id"].unique(), CONTINUOUS]
        all_cov[CONTINUOUS] = (all_cov[CONTINUOUS] - ref.mean()) / ref.std()
    cov = _covariates(root)
    bmi_ref = cov.loc[train["patient_id"].unique(), "bmi"]
    bmi_mean, bmi_std = bmi_ref.mean(), bmi_ref.std()
    cov = _covariates(root)
    bmi_ref = cov.loc[train["patient_id"].unique(), "bmi"]
    bmi_mean, bmi_std = bmi_ref.mean(), bmi_ref.std()
    wanted = rows[rows["cohort"] == split] if split in HELDOUT else rows[in_cohorts & (parts == split)]
    treatment = {name: k for k, name in enumerate(TREATMENT_CLASSES)}
    treatment["bimekizumab"] = treatment["secukinumab"]
    batches, tables = [], []
    for cohort, group in wanted.groupby("cohort"):
        ids = group["patient_id"].unique()
        ids = ids[rng.permutation(len(ids))]
        for start in range(0, len(ids), batch_size):
            chunk = [group[group["patient_id"] == pid] for pid in ids[start:start + batch_size]]
            N, D = max(2, *(len(c) for c in chunk)), len(COLUMNS)  # IntegratedPosterior needs 2 steps
            t, x, mask = np.zeros((len(chunk), N)), np.zeros((len(chunk), N, D)), np.zeros((len(chunk), N, D))
            u = np.zeros((len(chunk), N, DIM_U))
            cv = cov.loc[[g["patient_id"].iloc[0] for g in chunk]]
            if all_cov is not None:
                a = all_cov.loc[[g["patient_id"].iloc[0] for g in chunk]]
                mu = np.c_[a.fillna(0.0).to_numpy(), a.notna().to_numpy()].astype(float)
            else:
                mu = np.c_[cv["female"], ((cv["bmi"] - bmi_mean) / bmi_std).fillna(0.0), cv["bmi"].isna(),
                           cv["smoking_status"] == "current", cv["smoking_status"] == "former"].astype(float)
            table = pd.DataFrame({"cohort": cohort, "patient_id": np.repeat([c["patient_id"].iloc[0] for c in chunk], N),
                                  "date": pd.NaT})
            for b, c in enumerate(chunk):
                n = len(c)
                years = (c["date"] - c["date"].iloc[0]).dt.days.to_numpy() / 365.25
                v = (_transform(c[COLUMNS].to_numpy(float)) - mean) / std
                t[b, :n], t[b, n:] = years, years[-1]
                x[b, :n], mask[b, :n] = np.nan_to_num(v), np.isfinite(v)
                u[b, np.arange(n), c["current_treatment"].map(treatment).fillna(0).astype(int).to_numpy()] = 1.0
                u[b, n:] = u[b, n - 1]
                table.loc[b * N:b * N + n - 1, "date"] = c["date"].to_numpy()
            batches.append(make_batch(torch.as_tensor(t, dtype=dtype), torch.as_tensor(x, dtype=dtype),
                                      torch.as_tensor(mask, dtype=dtype), torch.as_tensor(mu, dtype=dtype),
                                      torch.as_tensor(u, dtype=dtype)))
            tables.append(table)
    return batches, tables
