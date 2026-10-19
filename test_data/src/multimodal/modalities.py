"""
Emission models: how each modality sees the latent state (subtype c, damage s, activity a, flare pulse f).

Each modality is a partial view on purpose, so the latent dynamics are only identifiable jointly:

    clinical records  long-running, flare-triggered encounters (timing and frequency of flares), site-specific
                      misdiagnosis codes and comorbidities (weak on c), escalation and surgery (coarse,
                      censored on s); no severity scores
    imaging           exam, photo and ultrasound: strong on s, moderate on a, involved sites inform c;
                      visits every 3-6 months, so most 1-3 week flares fall between them
    biomarkers        serum panel: strong on a, cytokine endotype informs c, weak on s; few time points

CRP, the lesion exam (counts, Hurley, sites) and treatment history recur across cohorts as anchors that
tie the views together.
"""
import numpy as np
import pandas as pd

from .latent import (BIOLOGICS, COMORBIDITIES, DEROOFING, DERMATOLOGY, EMERGENCY, EXCISION,
                     INCISION_DRAINAGE, PRIMARY, REGIONS, SETTING_NAMES, STAGE_CUTS, SUBTYPES, TREATMENTS,
                     WPY, encounters_within, state, take, to_dates)

HURLEY_CONFUSION = np.array([[0.88, 0.11, 0.01], [0.12, 0.78, 0.10], [0.01, 0.17, 0.82]])

MARKERS = {  # log-mean: log(base) + activity*log1p(a) + damage*s + bmi*(BMI-27) + female*[F] + subtype offset
    "crp_mg_l": dict(base=1.0, activity=1.25, damage=0.12, bmi=0.03, female=0.0, sd=0.50, digits=1),
    "esr_mm_h": dict(base=9.0, activity=0.55, damage=0.10, bmi=0.02, female=0.20, sd=0.35, digits=0),
    "il6_pg_ml": dict(base=1.8, activity=0.90, damage=0.10, bmi=0.02, female=0.0, sd=0.60, digits=2),
    "il1b_pg_ml": dict(base=0.45, activity=0.80, damage=0.05, bmi=0.0, female=0.0, sd=0.70, digits=2),
    "il17a_pg_ml": dict(base=0.8, activity=0.90, damage=0.05, bmi=0.0, female=0.0, sd=0.70, digits=2),
    "tnfa_pg_ml": dict(base=1.6, activity=0.60, damage=0.05, bmi=0.0, female=0.0, sd=0.50, digits=2),
    "calprotectin_ug_ml": dict(base=0.9, activity=0.75, damage=0.10, bmi=0.0, female=0.0, sd=0.50, digits=2),
    "nlr": dict(base=1.9, activity=0.45, damage=0.05, bmi=0.0, female=0.0, sd=0.30, digits=2),
}
MARKER_OFFSETS = {name: np.array([t["markers"].get(name, 0.0) for t in SUBTYPES]) for name in MARKERS}
DRUG_INTERFERENCE = {("tnfa_pg_ml", 4): 0.8, ("il17a_pg_ml", 5): 0.9, ("il17a_pg_ml", 6): 0.9}  # drug-bound cytokine
LOINC = {"crp_mg_l": "1988-5", "esr_mm_h": "4537-7", "hba1c": "4548-4"}

# Fixed "world" properties shared by every run: the image feature extractor and device/operator offsets.
_world = np.random.default_rng(20260929)
IMAGE_W1 = _world.normal(0, 0.9, (9, 16))
IMAGE_B1 = _world.normal(0, 0.3, 16)
IMAGE_W2 = _world.normal(0, 0.5, (16, 8))
DEVICES = {name: (_world.normal(0, 0.04), _world.normal(0, 0.25, 8))
           for name in ("site_A", "site_B", "site_C", "surgical", "smartphone")}
US_OPERATORS = {name: _world.normal(0, 0.2) for name in ("site_A", "site_B", "site_C", "surgical")}

FLARE_CODES = {  # what an HS flare gets coded as when HS is not recognised, by lesion site
    "axillary": ["L02.411", "L02.412", "L02.91", "L02.92"],
    "inframammary": ["N61.1", "L02.213", "L02.91"],
    "inguinal": ["L02.214", "L02.91", "L02.92"],
    "gluteal_perianal": ["L02.31", "L05.01", "K61.0"],
    "atypical": ["L73.9", "L02.11", "L70.0", "L08.9"],
}
SURGERY_CPT = {  # (deroofing / limited, wide excision) by site
    "axillary": ("11450", "11451"), "inframammary": ("11404", "11406"), "inguinal": ("11462", "11463"),
    "gluteal_perianal": ("11470", "11471"), "atypical": ("11404", "11406"),
}
SURGICAL_SITE_WEIGHT = np.array([1.0, 0.4, 0.9, 1.3, 0.3])
NOISE_CODES = ["J06.9", "M54.50", "Z00.00", "R51.9", "J02.9", "K21.9", "N39.0", "Z23", "R10.9", "L30.9"]
CODING_SENSITIVITY = {"obesity": 0.2, "tobacco": 0.25}   # per background visit; 0.35 for other conditions


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def soft(x, scale=3.0):
    """~x for small x, logarithmic beyond `scale`: lesion counts saturate in extreme flares."""
    return scale * np.log1p(np.asarray(x) / scale)


def negbin(rng, mean, k):
    return rng.poisson(rng.gamma(k, np.maximum(mean, 1e-9) / k))


def report(values, lod, digits):
    """Round to the assay's precision; values below the limit of detection are reported at the LOD."""
    values = np.asarray(values, dtype=float)
    blq = values < lod
    return np.round(np.where(blq, lod, values), digits), blq


def involved_regions(pop, p, s):
    """Lesion sites involved at damage s: sites accumulate with progression in a patient-specific order."""
    prob = pop.region_p[p] * (0.45 + 0.55 * (1 - np.exp(-0.8 * s)))[:, None]
    u = pop.region_u[p]
    inv = u < prob
    inv[np.arange(len(p)), np.argmin(u / pop.region_p[p], axis=1)] = True
    return inv


# --- imaging / clinical exam -------------------------------------------------------------------------------

def clinical_exam(pop, st, rng):
    """Lesion counts, assessed Hurley stage, involved sites and patient-reported outcomes."""
    k = len(st.p)
    inv = involved_regions(pop, st.p, st.s)
    n_reg = inv.sum(1)
    chronic, flare = soft(np.maximum(st.a - st.f, 0)), soft(st.f)
    tunnel_rate = pop.tunnel_mult[st.p] * np.maximum(0.6 ** st.surgeries, 0.25) * np.maximum(st.s - 0.7, 0) ** 1.1
    depressed = pop.comorb["depression"][st.p] <= st.w
    cum = HURLEY_CONFUSION[st.stage - 1].cumsum(1)
    u = rng.random(k)
    out = pd.DataFrame({
        "nodules": negbin(rng, n_reg * (0.5 + 0.9 * chronic + 1.2 * flare), 2.0),
        "abscesses": negbin(rng, 0.1 + 0.9 * flare * np.sqrt(n_reg), 1.5),
        "draining_tunnels": negbin(rng, 0.02 + tunnel_rate, 3.0),
        "hurley_stage": 1 + (u > cum[:, 0]) + (u > cum[:, 1]),
        "dlqi": np.clip(np.round(1.5 + 3.0 * st.a + 1.2 * st.s + 3.0 * depressed + rng.normal(0, 3.5, k)), 0, 30),
        "pain_nrs": np.clip(np.round(0.8 + 1.8 * st.a + 0.4 * st.s + rng.normal(0, 1.5, k)), 0, 10),
    })
    for j, region in enumerate(REGIONS):
        out[f"site_{region}"] = inv[:, j].astype(int)
    return out, inv


def add_scores(exam):
    """AN count and IHS4 (nodules + 2 abscesses + 4 draining tunnels); missing if any count is missing."""
    exam.insert(3, "an_count", exam.nodules + exam.abscesses)
    exam.insert(4, "ihs4", exam.nodules + 2 * exam.abscesses + 4 * exam.draining_tunnels)
    return exam


def photo(pop, st, inv, rng, device, noise=1.0):
    """Features extracted from clinical photographs: lesion area, erythema, scarring, embedding."""
    k = len(st.p)
    device = np.broadcast_to(np.asarray(device), (k,))
    ery_shift = np.array([DEVICES[d][0] for d in device])
    emb_shift = np.array([DEVICES[d][1] for d in device]).reshape(k, 8)
    phototype = pop.phototype[st.p]
    x = np.column_stack([np.log1p(st.s), st.a, st.f, inv, phototype / 6])
    emb = np.tanh(x @ IMAGE_W1 + IMAGE_B1) @ IMAGE_W2 + emb_shift + rng.normal(0, 0.3 * noise, (k, 8))
    out = pd.DataFrame({
        "lesion_area_cm2": np.round(rng.lognormal(np.log(1 + 8 * st.s + 5 * soft(st.f) + 2 * inv.sum(1)), 0.35 * noise), 1),
        # erythema is under-detected on darker skin
        "erythema_index": np.round(np.maximum((0.15 + 0.5 * (1 - np.exp(-0.35 * st.a))) * (1 - 0.08 * (phototype - 1))
                                              + ery_shift + rng.normal(0, 0.05 * noise, k), 0), 3),
        "scarring_score": np.round(np.clip(sigmoid(1.6 * (st.s - 1.5)) + rng.normal(0, 0.07 * noise, k), 0, 1), 3),
    })
    for j in range(8):
        out[f"img_emb_{j}"] = np.round(emb[:, j], 4)
    return out


def ultrasound(pop, st, inv, rng, operator):
    """Doppler ultrasound: sees subclinical tunnels (earlier than the exam) and fluid collections."""
    k = len(st.p)
    operator = np.broadcast_to(np.asarray(operator), (k,))
    bias = np.array([US_OPERATORS[o] for o in operator])
    fluid = rng.poisson(0.2 + 1.2 * soft(st.f) * np.sqrt(inv.sum(1)) + 0.3 * st.s)
    tunnel_rate = (1.4 * pop.tunnel_mult[st.p] * np.maximum(0.75 ** st.surgeries, 0.4)
                   * np.maximum(st.s - 0.4, 0) ** 1.1)
    return pd.DataFrame({
        "us_fluid_collections": fluid,
        "us_tunnels": negbin(rng, 0.05 + tunnel_rate, 3.0),
        "us_max_collection_mm": np.where(fluid > 0, np.round(rng.lognormal(np.log(7 + 5 * soft(st.f) + 2 * st.s), 0.3), 1),
                                         np.nan),
        "us_dermal_thickness_mm": np.round(2.1 + 0.3 * st.s + 0.4 * np.log1p(st.a) + bias + rng.normal(0, 0.35, k), 2),
        "us_doppler_grade": np.digitize(1.2 * np.log1p(st.a) + 0.2 * st.s + rng.normal(0, 0.3, k), [0.6, 1.2, 1.9]),
        "us_sos_hs_stage": 1 + ((st.s + rng.normal(0, 0.35, k))[:, None] >= np.array(STAGE_CUTS)).sum(1),
    })


# --- biomarkers --------------------------------------------------------------------------------------------

def biomarkers(pop, st, rng, names, plate_shift=None):
    """Unrounded serum concentrations; plate_shift maps marker -> per-sample log offset (assay batch)."""
    out = {}
    for name in names:
        m = MARKERS[name]
        mu = (np.log(m["base"]) + m["activity"] * np.log1p(st.a) + m["damage"] * st.s + m["bmi"] * (pop.bmi[st.p] - 27)
              + m["female"] * pop.female[st.p] + MARKER_OFFSETS[name][pop.c[st.p]])
        for (marker, code), shift in DRUG_INTERFERENCE.items():
            if marker == name:
                mu = mu + shift * (st.tx == code)
        if plate_shift is not None and name in plate_shift:
            mu = mu + plate_shift[name]
        out[name] = np.exp(mu + rng.normal(0, m["sd"], len(st.p)))
    return pd.DataFrame(out)


# --- clinical records --------------------------------------------------------------------------------------

def records(p, date, setting, system, code, value=np.nan, flag=""):
    return pd.DataFrame({"p": np.asarray(p), "date": date, "setting": setting, "code_system": system,
                         "code": code, "value": value, "flag": flag})


def clinical_events(sim, p, start, end, rng, hospital=False):
    """Diagnoses, procedures, CRP labs and analgesics at the simulated HS-related encounters of patients p
    within [start, end]; hospital=True keeps emergency and dermatology encounters only."""
    pop = sim.pop
    e = encounters_within(sim, p, start, end)
    if hospital:
        e = e[e.setting != PRIMARY].reset_index(drop=True)
    ep, ew, es, eproc, edx = (e[col].to_numpy() for col in ("p", "w", "setting", "proc", "dx"))
    k = len(e)
    st = state(sim, ep, ew)
    inv = involved_regions(pop, ep, st.s)
    date = to_dates(ew, rng, 7)
    setting = np.array([SETTING_NAMES[x] for x in es], dtype=object)

    # HS is coded only once recognised, and not always; otherwise the flare site's abscess/folliculitis code
    region = np.argmax(inv * rng.random(inv.shape), axis=1)
    site_code = np.array([rng.choice(FLARE_CODES[REGIONS[r]]) for r in region], dtype=object)
    p_hs = np.select([es == PRIMARY, es == EMERGENCY], [0.65, 0.45], 0.95)
    frames = [records(ep, date, setting, "ICD10CM", np.where(edx & (rng.random(k) < p_hs), "L73.2", site_code))]

    drain = eproc == INCISION_DRAINAGE
    frames.append(records(ep[drain], date[drain], setting[drain], "CPT",
                          rng.choice(["10060", "10061"], drain.sum(), p=[0.6, 0.4])))
    surg = eproc >= DEROOFING
    site = np.argmax(inv[surg] * SURGICAL_SITE_WEIGHT * rng.random((surg.sum(), len(REGIONS))), axis=1)
    excision = eproc[surg] == EXCISION
    cpt = [SURGERY_CPT[REGIONS[r]][int(x)] for r, x in zip(site, excision)]
    lag = np.where(excision, np.minimum(rng.integers(2, 8, surg.sum()), e.end.to_numpy()[surg] - ew[surg]), 0)
    frames.append(records(ep[surg], to_dates(ew[surg] + lag, rng), np.where(excision, "outpatient_surgery",
                                                                           "dermatology"), "CPT", cpt))

    lab = rng.random(k) < np.select([es == PRIMARY, es == EMERGENCY], [0.05, 0.35], 0.15)
    crp, blq = report(biomarkers(pop, take(st, lab), rng, ["crp_mg_l"]).crp_mg_l, 0.5, 1)
    frames.append(records(ep[lab], date[lab], setting[lab], "LOINC", LOINC["crp_mg_l"], crp, np.where(blq, "BLQ", "")))

    if not hospital:
        pain = (es != DERMATOLOGY) & (rng.random(k) < 0.3)
        frames.append(records(ep[pain], date[pain], "pharmacy", "ATC", np.where(es[pain] == EMERGENCY, "N02AX02",
                                                                                "M01AE01")))
    return pd.concat(frames, ignore_index=True)


def dispensings(sim, p, start, end, rng, codes=None, setting="pharmacy"):
    """Fills implied by the weekly treatment state: one per refill interval of each course, ~10% of refills
    missed. codes restricts to some treatments (e.g. hospital-dispensed biologics)."""
    rows = []
    for pj, a, b in zip(p, start, end):
        tx = sim.TX[a - sim.w0: b - sim.w0 + 1, pj]
        cut = np.flatnonzero(np.r_[True, tx[1:] != tx[:-1]])
        for s0, s1 in zip(cut, np.r_[cut[1:], len(tx)]):
            code = int(tx[s0])
            if code == 0 or (codes is not None and code not in codes):
                continue
            t = TREATMENTS[code]
            fills = np.arange(s0, s1, t["refill"])
            fills = a + fills[np.r_[True, rng.random(len(fills) - 1) < 0.9]]
            for atc in t["atc"][rng.integers(len(t["atc"]))]:
                rows.append((np.full(len(fills), pj), fills, np.full(len(fills), atc, dtype=object)))
    if not rows:
        return records([], [], setting, "ATC", [])
    pf, wf, atc = (np.concatenate(x) for x in zip(*rows))
    return records(pf, to_dates(wf, rng, 7), setting, "ATC", atc)


def background_events(sim, p, start, end, rng):
    """Primary care unrelated to HS: comorbidity and everyday codes, routine labs, chronic medication."""
    pop = sim.pop
    mid = (start + end) // 2
    n_comorb = sum((pop.comorb[name][p] <= mid).astype(int) for name in COMORBIDITIES)
    age = (mid - pop.birth_week[p]) / WPY
    rate = np.sqrt(pop.psi[p] / pop.psi.mean()) * (2.0 + 0.04 * np.maximum(age - 30, 0) + 0.5 * n_comorb)
    k = rng.poisson(rate * (end - start + 1) / WPY)
    pv = np.repeat(p, k)
    wv = np.repeat(start, k) + (rng.random(k.sum()) * np.repeat(end - start + 1, k)).astype(int)
    date = to_dates(wv, rng)
    frames = []
    for name, (icd, _) in COMORBIDITIES.items():
        m = (pop.comorb[name][pv] <= wv) & (rng.random(len(pv)) < CODING_SENSITIVITY.get(name, 0.35))
        frames.append(records(pv[m], date[m], "primary_care", "ICD10CM", icd))
    m = rng.random(len(pv)) < 0.6
    frames.append(records(pv[m], date[m], "primary_care", "ICD10CM", rng.choice(NOISE_CODES, m.sum())))

    t2d = pop.comorb["t2d"][pv] <= wv
    m = rng.random(len(pv)) < np.where(t2d, 0.4, 0.04)
    hba1c = np.round(np.where(t2d[m], rng.normal(7.2, 1.1, m.sum()), rng.normal(5.4, 0.35, m.sum())), 1)
    frames.append(records(pv[m], date[m], "primary_care", "LOINC", LOINC["hba1c"], hba1c))
    m = rng.random(len(pv)) < 0.03
    crp, blq = report(biomarkers(pop, state(sim, pv[m], wv[m]), rng, ["crp_mg_l"]).crp_mg_l, 0.5, 1)
    frames.append(records(pv[m], date[m], "primary_care", "LOINC", LOINC["crp_mg_l"], crp, np.where(blq, "BLQ", "")))

    # 90-day fills of chronic medication from condition onset, for the ~70% who are treated
    for name, (_, atc) in COMORBIDITIES.items():
        if atc is None:
            continue
        first = np.maximum(pop.comorb[name][p], start)
        n_fill = np.where((first <= end) & (rng.random(len(p)) < 0.7), (end - first) // 12 + 1, 0)
        pf = np.repeat(p, n_fill)
        wf = np.repeat(first, n_fill) + 12 * (np.arange(n_fill.sum()) - np.repeat(np.cumsum(n_fill) - n_fill, n_fill))
        keep = rng.random(len(pf)) < 0.85
        frames.append(records(pf[keep], to_dates(wf[keep], rng, 7), "pharmacy", "ATC", atc))
    return pd.concat(frames, ignore_index=True)
