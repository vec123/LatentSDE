"""
Per-modality tables of the HS data set. A modality is a measurement type, not a cohort
(plan/multimodal-encoders-decoders.md, section 3): one table holds the rows of that measurement from all cohorts,
one row per (patient, date, source). The autoencoder of a modality reads one row and gives one latent vector. It
does not use the time; the date stays in the table, so that a later stage can put the latents on a time axis.

| modality    | columns                                                  | sources                                     |
|-------------|----------------------------------------------------------|---------------------------------------------|
| exam        | nodules, abscesses, tunnels, AN, IHS4, Hurley stage,     | specialist visits, surgical pre-op,         |
|             | DLQI, pain, 5 sites                                      | biomarker samples (AN, IHS4, Hurley, DLQI), |
|             |                                                          | community survey (sites, pain, DLQI)        |
| photo       | lesion area, erythema, scarring, img_emb_0..7            | specialist visits, community survey         |
| ultrasound  | fluid collections, tunnels, max collection, dermal       | specialist visits, surgical pre-op          |
|             | thickness, Doppler grade, SOS-HS stage                   |                                             |
| serum       | CRP, ESR, IL-6, IL-1b, IL-17A, TNF-a, calprotectin, NLR  | biomarker samples, specialist visits,       |
|             |                                                          | surgical pre-op, community point-of-care    |
|             |                                                          | CRP, CRP labs (LOINC 1988-5) of the records |
| self_report | flares in the last 6 months, doctor-diagnosed HS         | community survey                            |
| records     | all codes and settings of one date (multi-hot)           | claims events, specialist and surgical      |
|             |                                                          | hospital records                            |

The CRP values of the records go to serum, and stay in records only as the code. The HbA1c values (LOINC 4548-4)
are not used; their code stays in records. truth/ is not read.

Column kinds. The loss and the reconstruction follow the kind (model.py):
- normal: the value (or its log, with log=True), standardized; squared error.
- count: log1p of the count, standardized; squared error.
- censored: the log value, standardized. Below the limit of detection (blq), the data holds the LOD, and only the
  part of the reconstruction above the LOD counts.
- binary: 0 or 1; binary cross entropy.
- categorical: the index of the level; cross entropy (Hurley stage, Doppler grade, SOS-HS stage).

The patients of each cohort are split into train, val and test (seeded). All modalities use the same split, so
that a later multimodal stage can use it again.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

DATA = os.path.join("test_data", "data", "multimodal")
IDS = ("cohort", "patient_id", "date", "source")
MODALITIES = ("exam", "photo", "ultrasound", "serum", "self_report", "records")
PATIENTS = {"claims_ehr": "claims_ehr/patients.csv", "specialist_imaging": "specialist_imaging/patients.csv",
            "biomarker_study": "biomarker_study/patients.csv", "heldout_surgical": "heldout_surgical/patients.csv",
            "heldout_community": "heldout_community/participants.csv"}
RECORDS = {"claims_ehr": ("claims_events", "claims_lab", "claims_ehr/events.csv"),  # (records source, CRP source)
           "specialist_imaging": ("specialist_records", "specialist_lab", "specialist_imaging/hospital_records.csv"),
           "heldout_surgical": ("surgical_records", "surgical_lab", "heldout_surgical/hospital_records.csv")}
SITES = ("site_axillary", "site_inframammary", "site_inguinal", "site_gluteal_perianal", "site_atypical")
MARKERS = ("crp_mg_l", "esr_mm_h", "il6_pg_ml", "il1b_pg_ml", "il17a_pg_ml", "tnfa_pg_ml", "calprotectin_ug_ml")
CRP_LOINC = "1988-5"
SPLITS = ("train", "val", "test")


@dataclass
class Column:
    """One column of a modality: its kind, its transform to the model space and back."""
    name: str
    kind: str              # normal, count, censored, binary or categorical
    log: bool = False      # normal: take the log before the standardization
    levels: tuple = ()     # categorical: the values, in order
    mean: float = 0.0
    std: float = 1.0

    @property
    def width(self) -> int:
        return len(self.levels) if self.kind == "categorical" else 1

    def _scaled(self, v: np.ndarray) -> np.ndarray:
        if self.kind == "count":
            return np.log1p(v)
        return np.log(v) if self.log or self.kind == "censored" else v

    def fit(self, values) -> "Column":
        """Set mean and std from the (train) values. Only for the standardized kinds."""
        if self.kind in ("normal", "count", "censored"):
            v = self._scaled(np.asarray(values, dtype=float))
            v = v[np.isfinite(v)]
            self.mean, self.std = float(v.mean()), float(v.std()) or 1.0
        return self

    def forward(self, values) -> np.ndarray:
        """Data units -> model space. NaN (missing) stays NaN."""
        v = np.asarray(values, dtype=float)
        if self.kind == "binary":
            return v
        if self.kind == "categorical":
            out = np.full(v.shape, np.nan)
            for k, level in enumerate(self.levels):
                out[v == level] = k
            return out
        return (self._scaled(v) - self.mean) / self.std

    def inverse(self, values) -> np.ndarray:
        """Model space -> data units. Binary: the probability stays; categorical: the index -> its level."""
        v = np.asarray(values, dtype=float)
        if self.kind == "binary":
            return v
        if self.kind == "categorical":
            index = np.clip(np.rint(np.nan_to_num(v)), 0, len(self.levels) - 1).astype(int)
            return np.where(np.isfinite(v), np.asarray(self.levels, dtype=float)[index], np.nan)
        v = v * self.std + self.mean
        if self.kind == "count":
            return np.expm1(v)
        return np.exp(v) if self.log or self.kind == "censored" else v


def columns(modality: str, tokens: Sequence[str] = ()) -> List[Column]:
    """The columns of a modality, not fitted. records: one binary column per token (code or setting)."""
    if modality == "exam":
        return ([Column(c, "count") for c in ("nodules", "abscesses", "draining_tunnels", "an_count", "ihs4")]
                + [Column("hurley_stage", "categorical", levels=(1, 2, 3)), Column("dlqi", "normal"),
                   Column("pain_nrs", "normal")] + [Column(s, "binary") for s in SITES])
    if modality == "photo":
        return ([Column("lesion_area_cm2", "normal", log=True), Column("erythema_index", "normal"),
                 Column("scarring_score", "normal")] + [Column(f"img_emb_{j}", "normal") for j in range(8)])
    if modality == "ultrasound":
        return [Column("us_fluid_collections", "count"), Column("us_tunnels", "count"),
                Column("us_max_collection_mm", "normal", log=True), Column("us_dermal_thickness_mm", "normal"),
                Column("us_doppler_grade", "categorical", levels=(0, 1, 2, 3)),
                Column("us_sos_hs_stage", "categorical", levels=(1, 2, 3))]
    if modality == "serum":
        return [Column(m, "censored") for m in MARKERS] + [Column("nlr", "normal", log=True)]
    if modality == "self_report":
        return [Column("flares_last_6_months", "count"), Column("doctor_diagnosed_hs", "binary")]
    if modality == "records":
        return [Column(t, "binary") for t in tokens]
    raise ValueError(f"unknown modality {modality}")


# ------------------------------------------------------------------ tables
def _read(path: str) -> pd.DataFrame:
    frame = pd.read_csv(os.path.join(DATA, path), dtype={"code": str})
    return frame.rename(columns={"participant_id": "patient_id"})


def _block(frame: pd.DataFrame, cohort: str, source: str, date: str, cols: Sequence[Column],
           rename: Dict[str, str] = None, blq: Dict[str, str] = None) -> pd.DataFrame:
    """
    The rows of frame with at least one value in cols, as table rows: the ids, one column per Column (NaN where
    frame has no such column), and <name>_blq for each censored column. rename and blq map a Column name to the
    column of frame that holds its value or its BLQ flag (0 where blq gives none).
    """
    rename, blq = rename or {}, blq or {}
    out = pd.DataFrame({"cohort": cohort, "patient_id": frame["patient_id"].to_numpy(),
                        "date": frame[date].to_numpy(), "source": source})
    for c in cols:
        src = rename.get(c.name, c.name)
        out[c.name] = frame[src].to_numpy(dtype=float) if src in frame else np.nan
    for c in cols:
        if c.kind == "censored":
            out[f"{c.name}_blq"] = frame[blq[c.name]].fillna(0).to_numpy(dtype=float) if c.name in blq else 0.0
    return out[out[[c.name for c in cols]].notna().any(axis=1).to_numpy()]


def _tabular(modality: str) -> pd.DataFrame:
    cols = columns(modality)
    if modality in ("exam", "photo", "ultrasound"):
        visits = _read("specialist_imaging/visits.csv")
        blocks = [_block(visits, "specialist_imaging", "specialist_visit", "visit_date", cols)]
        if modality in ("exam", "ultrasound"):
            blocks.append(_block(_read("heldout_surgical/preop.csv"), "heldout_surgical", "surgical_preop",
                                 "preop_date", cols))
        if modality == "exam":
            blocks.append(_block(_read("biomarker_study/samples.csv"), "biomarker_study", "biomarker_sample",
                                 "sample_date", cols))
        if modality in ("exam", "photo"):
            blocks.append(_block(_read("heldout_community/participants.csv"), "heldout_community",
                                 "community_survey", "survey_date", cols))
        return pd.concat(blocks, ignore_index=True)
    if modality == "serum":
        crp = {"crp_mg_l": "crp_blq"}
        blocks = [_block(_read("biomarker_study/samples.csv"), "biomarker_study", "biomarker_sample", "sample_date",
                         cols, blq={m: f"{m}_blq" for m in MARKERS}),
                  _block(_read("specialist_imaging/visits.csv"), "specialist_imaging", "specialist_visit",
                         "visit_date", cols, blq=crp),
                  _block(_read("heldout_surgical/preop.csv"), "heldout_surgical", "surgical_preop", "preop_date",
                         cols, blq=crp),
                  _block(_read("heldout_community/participants.csv"), "heldout_community", "community_poc",
                         "survey_date", cols, rename={"crp_mg_l": "poc_crp_mg_l"}, blq={"crp_mg_l": "poc_crp_blq"})]
        for cohort, (_, source, path) in RECORDS.items():
            labs = _read(path)
            labs = labs[labs["code"] == CRP_LOINC]
            labs = labs.assign(crp_mg_l=labs["value"], crp_blq=(labs["flag"] == "BLQ").astype(float))
            blocks.append(_block(labs, cohort, source, "date", cols, blq=crp))
        return pd.concat(blocks, ignore_index=True)
    if modality == "self_report":
        return _block(_read("heldout_community/participants.csv"), "heldout_community", "community_survey",
                      "survey_date", cols)
    raise ValueError(f"unknown modality {modality}")


def _records() -> Tuple[pd.DataFrame, List[str]]:
    """One row per (patient, date) of the records: 1 for each code and each setting of that date, else 0."""
    events = []
    for cohort, (source, _, path) in RECORDS.items():
        e = _read(path)
        key = pd.DataFrame({"cohort": cohort, "patient_id": e["patient_id"].to_numpy(), "date": e["date"].to_numpy(),
                            "source": source})
        events.append(key.assign(token=(e["code_system"] + ":" + e["code"]).to_numpy()))
        events.append(key.assign(token=("setting:" + e["setting"]).to_numpy()))
    events = pd.concat(events, ignore_index=True)
    grouped = events.groupby(list(IDS), sort=False)
    rows, keys = grouped.ngroup().to_numpy(), grouped.size().index.to_frame(index=False)
    tokens = sorted(events["token"].unique(), key=lambda t: (not t.startswith("setting:"), t))
    col = pd.Index(tokens).get_indexer(events["token"])
    hot = np.zeros((len(keys), len(tokens)), dtype=np.float32)
    hot[rows, col] = 1.0
    table = pd.concat([keys, pd.DataFrame(hot, columns=tokens)], axis=1)
    return table, tokens


def load(modality: str, split: pd.DataFrame) -> Tuple[pd.DataFrame, List[Column]]:
    """The table of a modality, with the column "split" (from split_patients), and its columns (not fitted)."""
    if modality == "records":
        table, tokens = _records()
        cols = columns("records", tokens)
    else:
        table, cols = _tabular(modality), columns(modality)
    table = table.merge(split, on=["cohort", "patient_id"], how="left", validate="many_to_one")
    if table["split"].isna().any():
        raise ValueError(f"{modality}: rows of patients that the split does not have")
    return table, cols


def split_patients(seed: int = 0, fractions: Tuple[float, float] = (0.8, 0.1)) -> pd.DataFrame:
    """(cohort, patient_id, split): the patients of each cohort in random order, the first fractions[0] train, the
    next fractions[1] val, the rest test."""
    rng = np.random.default_rng(seed)
    frames = []
    for cohort, path in PATIENTS.items():
        ids = _read(path)["patient_id"].to_numpy()
        n_train, n_val = round(fractions[0] * len(ids)), round(fractions[1] * len(ids))
        split = np.array(["test"] * len(ids), dtype=object)
        order = rng.permutation(len(ids))
        split[order[:n_train]], split[order[n_train:n_train + n_val]] = "train", "val"
        frames.append(pd.DataFrame({"cohort": cohort, "patient_id": ids, "split": split}))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ arrays
def fit(cols: Sequence[Column], table: pd.DataFrame) -> List[Column]:
    """Fit the transform of each column on the train rows of table."""
    train = table[table["split"] == "train"]
    return [c.fit(train[c.name]) for c in cols]


def arrays(table: pd.DataFrame, cols: Sequence[Column]) -> Tuple[np.ndarray, np.ndarray]:
    """x (n, D) in the model space, NaN where missing, and blq (n, D) bool, True where a censored value is below
    its LOD."""
    x = np.stack([c.forward(table[c.name].to_numpy()) for c in cols], axis=1).astype(np.float32)
    blq = np.stack([table[f"{c.name}_blq"].to_numpy() > 0 if c.kind == "censored" else np.zeros(len(table), bool)
                    for c in cols], axis=1)
    return x, blq


def to_units(values: np.ndarray, cols: Sequence[Column]) -> np.ndarray:
    """(n, D) in the model space -> data units, column by column."""
    return np.stack([c.inverse(values[:, j]) for j, c in enumerate(cols)], axis=1)
