"""
Checks on the generated cohorts: realism statistics, and an information map showing what each modality
alone reveals about the latent state (out-of-sample Spearman for progression s and activity a, accuracy
for the subtype) next to all modalities of a cohort combined.
"""
import numpy as np
import pandas as pd

from .latent import BIOLOGICS, COMORBIDITIES, REGIONS, SUBTYPES, TREATMENTS, WPY, stage_of, week_of, weeks_of
from .modalities import FLARE_CODES

CODE_GROUPS = {f"code_{r}": codes for r, codes in FLARE_CODES.items()}
CODE_GROUPS["code_hs"] = ["L73.2"]
CODE_GROUPS.update({f"code_{name}": [icd] for name, (icd, _) in COMORBIDITIES.items()})
ATC_GROUPS = {"fill_antibiotic": [c for k in (2, 3, 7) for course in TREATMENTS[k]["atc"] for c in course],
              "fill_biologic": [c for k in BIOLOGICS for course in TREATMENTS[k]["atc"] for c in course]}


def population_summary(sim):
    pop = sim.pop
    wk = week_of("2024-12-30")
    prev = pop.onset_week <= wk
    dx = (sim.diag_week <= wk) & prev
    recent = dx & (sim.diag_week >= week_of("2012-01-02"))
    delay = (sim.diag_week - pop.onset_week)[recent] / WPY
    stage = stage_of(sim.S[wk - sim.w0].astype(float))
    on_bio = np.isin(sim.TX[wk - sim.w0], BIOLOGICS)
    return pd.Series({
        "prevalent patients (2024-12)": prev.sum(),
        "female": pop.female[prev].mean(),
        "mean age at onset": ((pop.onset_week - pop.birth_week) / WPY)[prev].mean(),
        "diagnosed": dx.sum() / prev.sum(),
        "diagnostic delay, mean years (diagnosed 2012+)": delay.mean(),
        "diagnostic delay, median years": np.median(delay),
        "Hurley I": (stage[prev] == 1).mean(), "Hurley II": (stage[prev] == 2).mean(),
        "Hurley III": (stage[prev] == 3).mean(),
        "on a biologic, among diagnosed": on_bio[dx].mean(),
    }).round(3)


def cohort_summary(tables):
    truth = tables["truth/observations"]
    patients = tables["truth/patients"]
    rows = []
    for cohort, obs in truth.groupby("cohort", sort=False):
        weeks = weeks_of(obs.date)
        span = pd.Series(weeks).groupby(obs.patient_id.to_numpy()).agg(["min", "max", "size"])
        span = span[span["max"] > span["min"]]          # longitudinal patients only
        years = (span["max"] - span["min"] + 1) / WPY
        last = obs.groupby("patient_id").stage.last()
        sub = patients[patients.cohort == cohort].subtype_id
        rows.append({"cohort": cohort, "patients": len(last), "observation dates": len(obs),
                     "median follow-up (y)": np.median(years) if len(span) else np.nan,
                     "dates per patient-year": span["size"].sum() / years.sum() if len(span) else np.nan,
                     **{f"Hurley {s} (last)": (last == s).mean() for s in (1, 2, 3)},
                     **{t["name"]: (sub == k).mean() for k, t in enumerate(SUBTYPES)}})
    return pd.DataFrame(rows).set_index("cohort").round(3)


def flare_coverage(sim, tables, window=2):
    """Share of flares during each patient's follow-up seen by an observation within `window` weeks."""
    out = {}
    for cohort, obs in tables["truth/observations"].groupby("cohort", sort=False):
        seen = total = 0
        for pid, w in pd.Series(weeks_of(obs.date)).groupby(obs.patient_id.to_numpy()):
            w = np.unique(w.to_numpy())
            if len(w) < 2:
                continue
            flares = np.flatnonzero(sim.NEW[w[0] - sim.w0: w[-1] - sim.w0 + 1, sim.patient_index[pid]]) + w[0]
            pos = np.searchsorted(w, flares)
            seen += ((pos < len(w)) & (w[np.minimum(pos, len(w) - 1)] - flares <= window)).sum()
            total += len(flares)
        out[cohort] = seen / total if total else np.nan
    return pd.Series(out, name=f"flares observed within {window} weeks").round(3)


def _log1p(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def _scores(X, groups, y_s, y_a, y_c, seed=0):
    """Ridge fit on a random half of the patients, evaluated on the other half."""
    X = _log1p(X)
    miss = np.isnan(X)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
    Z = np.column_stack([np.where(miss, 0, (X - mu) / sd), miss[:, miss.any(0) & ~miss.all(0)], np.ones(len(X))])
    uniq = np.unique(groups)
    train = np.isin(groups, np.random.default_rng(seed).choice(uniq, len(uniq) // 2, replace=False))
    G = Z[train].T @ Z[train] + np.eye(Z.shape[1])

    def fit(Y):
        return Z[~train] @ np.linalg.solve(G, Z[train].T @ Y[train])

    spearman = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")
    onehot = np.eye(len(SUBTYPES))[y_c]
    return {"spearman progression": spearman(fit(y_s), y_s[~train]),
            "spearman activity": spearman(fit(y_a), y_a[~train]),
            "subtype accuracy": (fit(onehot).argmax(1) == y_c[~train]).mean()}


def _records_window(records, keys, date_col, weeks=26):
    """Counts of hospital-record events in the `weeks` before each observation (keys: patient_id, date)."""
    r = records.assign(code_str=records.code.astype(str))
    m = keys[["patient_id", date_col]].reset_index().merge(r, on="patient_id")
    m = m[(m.date <= m[date_col]) & (m.date > m[date_col] - pd.Timedelta(weeks=weeks))]
    feats = pd.DataFrame({
        "ed": (m.setting == "emergency") & (m.code_system == "ICD10CM"),
        "derm": (m.setting == "dermatology") & (m.code_system == "ICD10CM"),
        "drainage": m.code_str.isin(["10060", "10061"]),
        "surgery": m.code_str.str.startswith("114"),
        "biologic": m.code_str.isin(ATC_GROUPS["fill_biologic"]),
    }).astype(float).assign(index=m["index"].to_numpy())
    out = feats.groupby("index").sum().reindex(keys.index, fill_value=0.0)
    out["crp_lab"] = m[m.code_system == "LOINC"].groupby("index").value.max().reindex(keys.index)
    return out


def _claims_features(events, patients):
    years = ((patients.enrolment_end - patients.enrolment_start).dt.days / 365.25).to_numpy()
    e = events.assign(code_str=events.code.astype(str))
    feats = {}
    hs = e[e.code_system == "ICD10CM"]
    for setting in ("primary_care", "emergency", "dermatology"):
        feats[f"visits_{setting}"] = hs[hs.setting == setting].groupby("patient_id").date.nunique()
    feats.update({g: e[e.code_str.isin(codes)].groupby("patient_id").size() for g, codes in CODE_GROUPS.items()})
    feats.update({g: e[e.code_str.isin(codes)].groupby("patient_id").size() for g, codes in ATC_GROUPS.items()})
    feats["procedures"] = e[e.code_system == "CPT"].groupby("patient_id").size()
    feats["crp_lab"] = e[e.code_str == "1988-5"].groupby("patient_id").value.mean()
    X = pd.DataFrame(feats).reindex(patients.patient_id)
    rates = [c for c in X.columns if c != "crp_lab"]
    X[rates] = X[rates].fillna(0).to_numpy() / years[:, None]
    groups = {"encounter timing": [c for c in X.columns if c.startswith("visits_")],
              "diagnosis codes": list(CODE_GROUPS), "dispensings + procedures": list(ATC_GROUPS) + ["procedures"],
              "CRP labs": ["crp_lab"]}
    return X, groups


def information_map(tables):
    truth = tables["truth/observations"].assign(date=lambda d: pd.to_datetime(d.date))
    subtype = tables["truth/patients"].set_index("patient_id").subtype_id
    rows = []

    def add(cohort, X, groups, key_ids, key_dates=None):
        if key_dates is None:          # patient-level: latent state at the last observation
            t = truth[truth.cohort == cohort].groupby("patient_id").last().loc[key_ids]
        else:
            t = truth.set_index(["patient_id", "date"]).loc[list(zip(key_ids, key_dates))]
        y_s, y_a = t.progression.to_numpy(), t.activity.to_numpy()
        y_c = subtype.loc[key_ids].to_numpy()
        for name, cols in {**groups, "all combined": [c for cols in groups.values() for c in cols]}.items():
            rows.append({"cohort": cohort, "modality": name, **_scores(X[cols].to_numpy(), key_ids, y_s, y_a, y_c)})

    X, groups = _claims_features(tables["claims_ehr/events"], tables["claims_ehr/patients"])
    add("claims_ehr", X, groups, X.index.to_numpy())

    v = tables["specialist_imaging/visits"]
    rec = _records_window(tables["specialist_imaging/hospital_records"], v, "visit_date")
    X = pd.concat([v, rec], axis=1)
    groups = {"exam": [c for c in v.columns if c.startswith("site_")] + ["nodules", "abscesses", "draining_tunnels",
                                                                          "hurley_stage", "dlqi", "pain_nrs"],
              "photo": [c for c in v.columns if c.startswith(("img_", "lesion_", "erythema", "scarring"))],
              "ultrasound": [c for c in v.columns if c.startswith("us_")], "CRP": ["crp_mg_l"],
              "hospital records (26 wk)": list(rec.columns)}
    add("specialist_imaging", X, groups, v.patient_id.to_numpy(), v.visit_date)

    s = tables["biomarker_study/samples"]
    groups = {"cytokines": ["il6_pg_ml", "il1b_pg_ml", "il17a_pg_ml", "tnfa_pg_ml", "calprotectin_ug_ml"],
              "clinical labs": ["crp_mg_l", "esr_mm_h", "nlr"], "exam": ["an_count", "ihs4", "hurley_stage", "dlqi"]}
    add("biomarker_study", s, groups, s.patient_id.to_numpy(), s.sample_date)

    c = tables["heldout_community/participants"]
    groups = {"self-report": [f"site_{r}" for r in REGIONS] + ["flares_last_6_months", "pain_nrs", "dlqi"],
              "smartphone photo": [x for x in c.columns if x.startswith(("img_", "lesion_", "erythema", "scarring"))],
              "point-of-care CRP": ["poc_crp_mg_l"]}
    add("heldout_community", c, groups, c.participant_id.to_numpy(), c.survey_date)
    return pd.DataFrame(rows).set_index(["cohort", "modality"]).round(3)


def report_all(sim, tables, info=None):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    lines = ["== population ==", population_summary(sim).to_string(), "", "== cohorts ==",
             cohort_summary(tables).T.to_string(), "", "== flare coverage ==", flare_coverage(sim, tables).to_string()]
    claims = tables["claims_ehr/events"]
    years = ((tables["claims_ehr/patients"].enrolment_end - tables["claims_ehr/patients"].enrolment_start).dt.days
             / 365.25).sum()
    enc = claims[claims.code_system == "ICD10CM"].drop_duplicates(["patient_id", "date", "setting"])
    lines += ["", "== claims_ehr: encounters with a diagnosis code per patient-year ==",
              (enc.setting.value_counts() / years).round(2).to_string(),
              f"events per patient-year: {len(claims) / years:.1f}",
              f"patients with an HS code (L73.2): {claims[claims.code == 'L73.2'].patient_id.nunique() / len(tables['claims_ehr/patients']):.2f}",
              f"patients with a biologic fill: {claims[claims.code.isin(ATC_GROUPS['fill_biologic'])].patient_id.nunique() / len(tables['claims_ehr/patients']):.2f}"]
    v = tables["specialist_imaging/visits"]
    t = tables["truth/observations"].set_index(["patient_id", "date"])
    v_stage = t.loc[list(zip(v.patient_id, v.visit_date))].stage.to_numpy()
    lines += ["", "== specialist_imaging: visits ==",
              f"available: photo {v.lesion_area_cm2.notna().mean():.2f}, ultrasound {v.us_tunnels.notna().mean():.2f}, "
              f"CRP {v.crp_mg_l.notna().mean():.2f}, DLQI {v.dlqi.notna().mean():.2f}",
              f"mean DLQI {v.dlqi.mean():.1f}; median IHS4 by true Hurley: "
              + ", ".join(f"{s}: {np.nanmedian(v.ihs4[v_stage == s]):.0f}" for s in (1, 2, 3))]
    s = tables["biomarker_study/samples"]
    lines += ["", "== biomarker_study: samples ==", s.visit.value_counts().to_string(),
              f"median CRP {s.crp_mg_l.median():.1f} mg/L; below LOD: "
              + ", ".join(f"{c[:-4]} {s[c].mean():.2f}" for c in s.columns if c.endswith("_blq"))]
    c = tables["heldout_community/participants"]
    lines += ["", f"== heldout_community: doctor-diagnosed {c.doctor_diagnosed_hs.mean():.2f} ==",
              "", "== information map (out-of-sample) ==",
              (information_map(tables) if info is None else info).to_string()]
    return "\n".join(lines)
