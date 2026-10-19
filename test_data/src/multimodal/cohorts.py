"""
Synthetic multi-cohort hidradenitis suppurativa data mimicking the Almirall hackathon challenge.

One latent model (latent.py) drives every cohort; cohorts select patients from disjoint slices of the
population and record different modalities (modalities.py) at their own frequency and completeness:

    claims_ehr/          large; clinical records only: patients.csv, events.csv (ICD-10-CM diagnoses, ATC
                         dispensings, CPT procedures, LOINC labs). Encounters are flare-triggered, HS is
                         coded only after the ~7-year diagnostic delay; window 2012-2024, ~4.5 years each
    specialist_imaging/  dermatology registry, enriched for Hurley II-III; patients.csv, visits.csv (exam,
                         photo features, ultrasound, CRP every 3-6 months) and hospital_records.csv for the
                         same patients. Site C has no ultrasound (truth/withheld_ultrasound.csv)
    biomarker_study/     small; inclusion Hurley II-III and AN count >= 3; samples.csv with a serum panel at
                         weeks 0/16/52 (LOD censoring, assay plates, dropout) plus a minimal exam
    heldout_surgical/    selected by HS surgery, a mechanism absent from training: preop.csv (exam,
                         ultrasound, labs) and hospital_records.csv
    heldout_community/   population survey, mostly undiagnosed and mild: participants.csv (self-report,
                         smartphone photo features, point-of-care CRP)
    truth/               patients.csv (subtype, onset, diagnosis), observations.csv (stage, progression s,
                         activity a at every observation date), withheld_ultrasound.csv

No cohort identifies the dynamics alone: records see flare timing over years but not severity, imaging
sees progression but misses flares between visits, biomarkers see activity and endotype at three time
points. CRP, the lesion exam, lesion sites and treatment history are shared anchors across cohorts.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .latent import (BIOLOGICS, COMORBIDITIES, DEROOFING, DERMATOLOGY, EXCISION, NEVER, REGIONS, SIM_END,
                     SUBTYPES, TREATMENT_NAMES, WPY, encounters_within, sample_population, simulate, state,
                     take, to_dates, week_of, weeks_of)
from .modalities import (MARKERS, add_scores, background_events, biomarkers, clinical_events, clinical_exam,
                         dispensings, involved_regions, photo, report, ultrasound)

POOLS = {"claims_ehr": 0.36, "specialist_imaging": 0.21, "biomarker_study": 0.15,
         "heldout_surgical": 0.16, "heldout_community": 0.12}
SIZES = {"claims_ehr": 4000, "specialist_imaging": 700, "biomarker_study": 220,
         "heldout_surgical": 300, "heldout_community": 800}
SITES = {"site_A": dict(share=0.40, photo=0.65, us=0.60),
         "site_B": dict(share=0.35, photo=0.85, us=0.45),
         "site_C": dict(share=0.25, photo=0.60, us=0.0)}
RESEARCH_PANEL = ["il6_pg_ml", "il1b_pg_ml", "il17a_pg_ml", "tnfa_pg_ml", "calprotectin_ug_ml"]
STUDY_LOD = {"crp_mg_l": 0.3, "esr_mm_h": 2.0, "nlr": 0.0, "il6_pg_ml": 1.5, "il1b_pg_ml": 1.0, "il17a_pg_ml": 1.2,
             "tnfa_pg_ml": 0.7, "calprotectin_ug_ml": 0.1}
SELF_REPORTED = ["t2d", "ibd", "depression", "pcos", "spondyloarthritis", "acne", "pilonidal"]
EXAM_COLUMNS = ["nodules", "abscesses", "draining_tunnels"]


def _sex(pop, p):
    return np.where(pop.female[p], "F", "M")


def _smoking(pop, p, rng):
    u = rng.random(len(p))
    return np.where(pop.smoker[p], np.where(u < 0.7, "current", "former"), np.where(u < 0.93, "never", "former"))


def _blank(df, mask, columns=None):
    """Set columns (default: all) to missing on rows where mask is true."""
    df.loc[np.asarray(mask), list(df.columns) if columns is None else columns] = np.nan
    return df


def _exam_with_gaps(exam, rng, item=0.03, hurley=0.07, dlqi=0.2, pain=0.15):
    for col in EXAM_COLUMNS:
        _blank(exam, rng.random(len(exam)) < item, [col])
    _blank(exam, rng.random(len(exam)) < hurley, ["hurley_stage"])
    _blank(exam, rng.random(len(exam)) < dlqi, ["dlqi"])
    _blank(exam, rng.random(len(exam)) < pain, ["pain_nrs"])
    return add_scores(exam)


def _self_report(pop, p, w, rng, conditions, sensitivity=0.85, false_positive=0.02):
    return {f"hx_{name}": np.where(pop.comorb[name][p] <= w, rng.random(len(p)) < sensitivity,
                                   rng.random(len(p)) < false_positive).astype(int) for name in conditions}


def _hospital_records(sim, p, start, end, rng):
    return pd.concat([clinical_events(sim, p, start, end, rng, hospital=True),
                      dispensings(sim, p, start, end, rng, codes=BIOLOGICS, setting="hospital_pharmacy")],
                     ignore_index=True)


def _finish_events(events, ids):
    events = events.assign(patient_id=ids.loc[events.p].to_numpy())
    events = events.sort_values(["patient_id", "date"], kind="stable")
    return events[["patient_id", "date", "setting", "code_system", "code", "value", "flag", "p"]]


def _ids(prefix, p):
    return pd.Series([f"{prefix}{i:05d}" for i in range(1, len(p) + 1)], index=p)


def build_claims(sim, pool, n, rng):
    pop = sim.pop
    lo, hi, stop = week_of("2012-01-02"), week_of("2023-07-03"), week_of("2024-12-30")
    start = rng.integers(lo, hi, len(pool))
    end = np.minimum(start + np.round(rng.exponential(4.5, len(pool)).clip(0.5) * WPY).astype(int), stop)
    e = encounters_within(sim, pool, start, end)
    hs_related = e.groupby("p").size()
    derm_dx = e[(e.setting == DERMATOLOGY) & e.dx].p.unique()
    eligible = np.union1d(hs_related.index[hs_related >= 2], derm_dx)   # "probable HS" case definition
    p = np.sort(rng.choice(eligible, min(n, len(eligible)), replace=False))
    j = np.searchsorted(pool, p)
    start, end = start[j], end[j]
    ids = _ids("CL", p)
    events = _finish_events(pd.concat([clinical_events(sim, p, start, end, rng), dispensings(sim, p, start, end, rng),
                                       background_events(sim, p, start, end, rng)], ignore_index=True), ids)
    patients = pd.DataFrame({"patient_id": ids.to_numpy(), "sex": _sex(pop, p), "birth_year": pop.birth_year[p],
                             "enrolment_start": to_dates(start, rng, 1),
                             "enrolment_end": to_dates(end, rng, 1) + pd.Timedelta(days=6)})
    return SimpleNamespace(ids=ids, obs=events[["p", "date"]],
                           tables={"claims_ehr/patients": patients, "claims_ehr/events": events.drop(columns="p")})


def build_specialist(sim, pool, n, rng):
    pop = sim.pop
    lo, hi, stop = week_of("2014-01-06"), week_of("2023-06-26"), week_of(SIM_END)
    enc = sim.enc
    derm = enc[(enc.setting == DERMATOLOGY) & enc.dx & enc.p.isin(pool)]
    first = derm[(derm.w >= lo) & (derm.w <= hi)].groupby("p").w.min()
    stage = state(sim, first.index, first.to_numpy()).stage
    weight = 0.25 + 0.75 * (stage >= 2) + 0.5 * (stage == 3)          # tertiary centres see the severe end
    p = np.sort(rng.choice(first.index, min(n, len(first)), replace=False, p=weight / weight.sum()))
    enrol = first.loc[p].to_numpy()
    leave = np.minimum(enrol + np.round(rng.exponential(8 * WPY, len(p))).astype(int), stop)  # consent withdrawal
    v = derm.merge(pd.DataFrame({"p": p, "enrol": enrol, "leave": leave}), on="p")
    v = v[(v.w >= v.enrol) & (v.w <= v.leave)].reset_index(drop=True)
    site = pd.Series(rng.choice(list(SITES), len(p), p=[s["share"] for s in SITES.values()]), index=p)
    vsite = site.loc[v.p].to_numpy()
    ids = _ids("SP", p)

    st = state(sim, v.p, v.w)
    k = len(v)
    exam, inv = clinical_exam(pop, st, rng)
    missed_exam = rng.random(k) < 0.03
    exam = _exam_with_gaps(_blank(exam, missed_exam), rng)
    pics = _blank(photo(pop, st, inv, rng, vsite), rng.random(k) >= np.array([SITES[s]["photo"] for s in vsite]))
    us = ultrasound(pop, st, inv, rng, vsite)
    us_rate = np.array([SITES[s]["us"] for s in vsite]) * np.where(exam.hurley_stage >= 2, 1.1, 0.6)  # scanned if severe
    withheld = us[vsite == "site_C"].assign(patient_id=ids.loc[v.p[vsite == "site_C"]].to_numpy())
    us = _blank(us.copy(), rng.random(k) >= us_rate)
    crp, blq = report(biomarkers(pop, st, rng, ["crp_mg_l"]).crp_mg_l, 0.5, 1)
    crp_drawn = rng.random(k) < np.where(np.isin(st.tx, BIOLOGICS), 0.6, 0.3)

    date = to_dates(v.w, rng)
    visits = pd.concat([pd.DataFrame({
        "patient_id": ids.loc[v.p].to_numpy(), "visit_date": date,
        "visit_number": v.groupby("p").cumcount().to_numpy() + 1,
        "current_treatment": TREATMENT_NAMES[st.tx]}), exam, pics, us, pd.DataFrame({
            "crp_mg_l": np.where(crp_drawn, crp, np.nan), "crp_blq": np.where(crp_drawn, blq, np.nan)})], axis=1)
    withheld = withheld.assign(visit_date=date[vsite == "site_C"])[["patient_id", "visit_date"] + list(us.columns)]

    onset_age = (pop.onset_week[p] - pop.birth_week[p]) / WPY
    patients = pd.DataFrame({
        "patient_id": ids.to_numpy(), "site": site.to_numpy(), "sex": _sex(pop, p), "birth_year": pop.birth_year[p],
        "enrolment_date": to_dates(enrol, rng),
        "age_at_onset_reported": np.where(rng.random(len(p)) < 0.85, np.round(onset_age + rng.normal(0, 2, len(p))), np.nan),
        "bmi": np.where(rng.random(len(p)) < 0.9, np.round(pop.bmi[p] + rng.normal(0, 0.5, len(p)), 1), np.nan),
        "smoking_status": _smoking(pop, p, rng),
        "family_history_hs": np.where(rng.random(len(p)) < 0.7, pop.family[p].astype(float), np.nan),
        "fitzpatrick_phototype": np.where(rng.random(len(p)) < 0.8, pop.phototype[p], np.nan),
        **_self_report(pop, p, enrol, rng, SELF_REPORTED)})

    hist_start = np.maximum(enrol - 104, sim.w0)
    records_ = _finish_events(_hospital_records(sim, p, hist_start, leave, rng), ids)
    obs = pd.concat([pd.DataFrame({"p": v.p.to_numpy(), "date": date}), records_[["p", "date"]]], ignore_index=True)
    return SimpleNamespace(ids=ids, obs=obs, tables={
        "specialist_imaging/patients": patients, "specialist_imaging/visits": visits,
        "specialist_imaging/hospital_records": records_.drop(columns="p"), "truth/withheld_ultrasound": withheld})


def build_biomarker(sim, pool, n, rng):
    pop = sim.pop
    lo, hi, stop = week_of("2019-01-07"), week_of("2023-06-26"), week_of(SIM_END)
    enc = sim.enc
    cand = enc[(enc.setting == DERMATOLOGY) & enc.dx & enc.p.isin(pool) & (enc.w >= lo) & (enc.w <= hi)]
    cand = cand.reset_index(drop=True)
    st = state(sim, cand.p, cand.w)
    age = (cand.w.to_numpy() - pop.birth_week[cand.p]) / WPY
    screen, _ = clinical_exam(pop, st, rng)
    ok = ((screen.hurley_stage >= 2) & (screen.nodules + screen.abscesses >= 3) & (age >= 18) & (age <= 70)
          & ~np.isin(st.tx, BIOLOGICS)).to_numpy()
    first = cand[ok].groupby("p").head(1)
    rows = np.sort(rng.choice(first.index, min(n, len(first)), replace=False))
    p, base = cand.p.to_numpy()[rows], cand.w.to_numpy()[rows]
    ids = _ids("BM", p)

    w16 = base + 16 + rng.integers(-2, 3, len(p))
    w52 = base + 52 + rng.integers(-4, 5, len(p))
    at16 = (rng.random(len(p)) < 0.88) & (w16 <= stop)
    at52 = at16 & (rng.random(len(p)) < 0.85) & (w52 <= stop)
    sp = np.concatenate([p, p[at16], p[at52]])
    sw = np.concatenate([base, w16[at16], w52[at52]])
    visit = np.repeat(["week_0", "week_16", "week_52"], [len(p), at16.sum(), at52.sum()])
    order = np.lexsort((sw, np.searchsorted(p, sp)))
    sp, sw, visit = sp[order], sw[order], visit[order]
    k = len(sp)
    st = state(sim, sp, sw)

    exam, _ = clinical_exam(pop, st, rng)
    is_base = visit == "week_0"
    exam.loc[is_base] = screen.loc[rows].to_numpy()          # baseline is the screening exam
    exam = add_scores(exam)[["an_count", "ihs4", "hurley_stage", "dlqi"]]

    plate = np.arange(k) // 40                                # samples run in patient order, 40 per plate
    shifts = {name: rng.normal(0, 0.15, plate.max() + 1)[plate] for name in RESEARCH_PANEL}
    raw = biomarkers(pop, st, rng, list(MARKERS), plate_shift=shifts)
    qc_fail = rng.random(k) < 0.04
    panel = {}
    for name in MARKERS:
        lod = STUDY_LOD[name]
        value, blq = report(raw[name], lod, MARKERS[name]["digits"])
        lost = qc_fail if name in RESEARCH_PANEL else rng.random(k) < 0.02
        panel[name] = np.where(lost, np.nan, value)
        if lod > 0:
            panel[f"{name}_blq"] = np.where(lost, np.nan, blq)
    date = to_dates(sw, rng)
    samples = pd.concat([pd.DataFrame({"patient_id": ids.loc[sp].to_numpy(), "visit": visit, "sample_date": date,
                                       "assay_plate": plate + 1, "current_treatment": TREATMENT_NAMES[st.tx]}),
                         exam, pd.DataFrame(panel)], axis=1)

    prior = [sim.TX[: b - sim.w0, pj] for pj, b in zip(p, base)]
    patients = pd.DataFrame({
        "patient_id": ids.to_numpy(), "sex": _sex(pop, p),
        "age_at_baseline": np.floor((base - pop.birth_week[p]) / WPY).astype(int),
        "bmi": np.round(pop.bmi[p] + rng.normal(0, 0.3, len(p)), 1), "smoking_status": _smoking(pop, p, rng),
        "disease_duration_years": np.round(np.maximum((base - pop.onset_week[p]) / WPY + rng.normal(0, 1.5, len(p)), 0), 1),
        "family_history_hs": pop.family[p].astype(int),
        "prior_systemic_antibiotics": [int(np.isin(t, (2, 3)).any()) for t in prior],
        "prior_biologic": [int(np.isin(t, BIOLOGICS).any()) for t in prior],
        "prior_hs_surgery": (state(sim, p, base).surgeries > 0).astype(int)})
    return SimpleNamespace(ids=ids, obs=pd.DataFrame({"p": sp, "date": date}),
                           tables={"biomarker_study/patients": patients, "biomarker_study/samples": samples})


def build_surgical(sim, pool, n, rng):
    pop = sim.pop
    lo, hi, stop = week_of("2018-01-01"), week_of("2024-12-30"), week_of(SIM_END)
    enc = sim.enc
    cand = enc[(enc.proc >= DEROOFING) & enc.p.isin(pool) & (enc.w >= lo) & (enc.w <= hi)].groupby("p").head(1)
    cand = cand.loc[np.sort(rng.choice(cand.index, min(n, len(cand)), replace=False))]
    p, index_w, proc = cand.p.to_numpy(), cand.w.to_numpy(), cand.proc.to_numpy()
    ids = _ids("SU", p)
    pre = index_w - rng.integers(1, 5, len(p))
    st = state(sim, p, pre)
    k = len(p)
    exam, inv = clinical_exam(pop, st, rng)
    exam = _exam_with_gaps(exam, rng, item=0.02, hurley=0.03, dlqi=0.5, pain=0.3)
    us = _blank(ultrasound(pop, st, inv, rng, "surgical"), rng.random(k) >= 0.95)
    labs = biomarkers(pop, st, rng, ["crp_mg_l", "esr_mm_h"])
    crp, crp_blq = report(labs.crp_mg_l, 0.5, 1)
    esr, _ = report(labs.esr_mm_h, 2.0, 0)
    drawn, esr_drawn = rng.random(k) < 0.8, rng.random(k) < 0.6
    date = to_dates(pre, rng)
    preop = pd.concat([pd.DataFrame({
        "patient_id": ids.to_numpy(), "preop_date": date,
        "procedure": np.where(proc == EXCISION, "wide_excision", "deroofing"),
        "current_treatment": TREATMENT_NAMES[st.tx]}), exam, us, pd.DataFrame({
            "crp_mg_l": np.where(drawn, crp, np.nan), "crp_blq": np.where(drawn, crp_blq, np.nan),
            "esr_mm_h": np.where(esr_drawn, esr, np.nan)})], axis=1)
    patients = pd.DataFrame({
        "patient_id": ids.to_numpy(), "sex": _sex(pop, p), "birth_year": pop.birth_year[p],
        "bmi": np.where(rng.random(k) < 0.8, np.round(pop.bmi[p], 1), np.nan),
        "smoking_status": np.where(rng.random(k) < 0.7, _smoking(pop, p, rng), None)})
    records_ = _finish_events(_hospital_records(sim, p, index_w - 104, np.minimum(index_w + 104, stop), rng), ids)
    obs = pd.concat([pd.DataFrame({"p": p, "date": date}), records_[["p", "date"]]], ignore_index=True)
    return SimpleNamespace(ids=ids, obs=obs, tables={
        "heldout_surgical/patients": patients, "heldout_surgical/preop": preop,
        "heldout_surgical/hospital_records": records_.drop(columns="p")})


def build_community(sim, pool, n, rng):
    pop = sim.pop
    w = rng.integers(week_of("2023-03-06"), week_of("2024-10-28"), len(pool))
    age = (w - pop.birth_week[pool]) / WPY
    st = state(sim, pool, w)
    p_respond = 0.22 * np.where(pop.female[pool], 1.3, 1.0) * (1 + 0.8 * (st.a > 0.8))  # symptomatic people reply
    respond = (pop.onset_week[pool] <= w) & (age >= 18) & (age <= 75) & (rng.random(len(pool)) < p_respond)
    j = np.sort(rng.choice(np.flatnonzero(respond), min(n, respond.sum()), replace=False))
    p, w, st = pool[j], w[j], take(st, j)
    k = len(p)
    ids = _ids("CO", p)

    exam, inv = clinical_exam(pop, st, rng)
    sites = np.where(inv, rng.random(inv.shape) < 0.9, rng.random(inv.shape) < 0.05).astype(int)
    i = w - sim.w0
    flares = sim.NEW[i[:, None] - np.arange(26), p[:, None]].sum(1).astype(int)
    pics = _blank(photo(pop, st, inv, rng, "smartphone", noise=1.6), rng.random(k) >= 0.55)
    poc, poc_blq = report(biomarkers(pop, st, rng, ["crp_mg_l"]).crp_mg_l, 5.0, 0)
    poc_done = rng.random(k) < 0.4
    date = to_dates(w, rng, 7)
    participants = pd.concat([pd.DataFrame({
        "participant_id": ids.to_numpy(), "survey_date": date, "age": np.floor(age[j]).astype(int),
        "sex": _sex(pop, p), "bmi_self_reported": np.where(rng.random(k) < 0.95, np.round(
            pop.bmi[p] - 0.8 + rng.normal(0, 1.2, k), 1), np.nan),
        "smoking_status": _smoking(pop, p, rng),
        "doctor_diagnosed_hs": (sim.diag_week[p] <= w).astype(int),
        **{f"site_{r}": sites[:, m] for m, r in enumerate(REGIONS)},
        "flares_last_6_months": rng.binomial(flares, 0.8) + rng.poisson(0.3, k),
        "pain_nrs": exam.pain_nrs.to_numpy(),
        "dlqi": np.where(rng.random(k) < 0.9, exam.dlqi, np.nan),
        **_self_report(pop, p, w, rng, ["depression", "t2d"], sensitivity=0.75)}), pics, pd.DataFrame({
            "poc_crp_mg_l": np.where(poc_done, poc, np.nan), "poc_crp_blq": np.where(poc_done, poc_blq, np.nan)})],
        axis=1)
    return SimpleNamespace(ids=ids, obs=pd.DataFrame({"p": p, "date": date}),
                           tables={"heldout_community/participants": participants})


BUILDERS = {"claims_ehr": build_claims, "specialist_imaging": build_specialist, "biomarker_study": build_biomarker,
            "heldout_surgical": build_surgical, "heldout_community": build_community}


def generate(seed=0, n_pop=40000, sizes=None):
    """Simulate the population and build every cohort; returns ({relative path: table}, sim)."""
    rng = np.random.default_rng(seed)
    sizes = {**SIZES, **(sizes or {})}
    pop = sample_population(n_pop, rng)
    sim = simulate(pop, rng)
    pool_of = rng.choice(len(POOLS), n_pop, p=list(POOLS.values()))
    tables, truth_patients, truth_obs = {}, [], []
    sim.patient_index = {}                       # cohort patient id -> population index
    for i, name in enumerate(POOLS):
        out = BUILDERS[name](sim, np.flatnonzero(pool_of == i), sizes[name], rng)
        tables.update(out.tables)
        sim.patient_index.update(zip(out.ids.to_numpy(), out.ids.index))
        obs = out.obs.drop_duplicates()
        st = state(sim, obs.p, weeks_of(obs.date))
        truth = pd.DataFrame({"cohort": name, "patient_id": out.ids.loc[obs.p].to_numpy(), "date": obs.date.to_numpy(),
                              "stage": st.stage, "progression": np.round(st.s, 3), "activity": np.round(st.a, 3)})
        truth = truth.sort_values(["patient_id", "date"], kind="stable")
        truth_obs.append(truth)
        p = out.ids.index.to_numpy()
        ends = truth.groupby("patient_id").stage.agg(["first", "last"]).loc[out.ids.to_numpy()]
        dx = sim.diag_week[p]
        truth_patients.append(pd.DataFrame({
            "cohort": name, "patient_id": out.ids.to_numpy(), "subtype_id": pop.c[p],
            "subtype": np.array([t["name"] for t in SUBTYPES])[pop.c[p]],
            "onset_date": to_dates(pop.onset_week[p], rng, 1),
            "diagnosis_date": to_dates(np.where(dx < NEVER, dx, 0), rng, 1).where(dx < NEVER),
            "stage_first_observation": ends["first"].to_numpy(), "stage_last_observation": ends["last"].to_numpy()}))
    tables["truth/patients"] = pd.concat(truth_patients, ignore_index=True)
    tables["truth/observations"] = pd.concat(truth_obs, ignore_index=True)
    return tables, sim
