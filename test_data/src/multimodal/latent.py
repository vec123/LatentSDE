"""
Shared latent model behind the synthetic hidradenitis suppurativa (HS) cohorts.

Every patient has a static subpopulation c and a disease course simulated on a weekly calendar grid:

    flares        Poisson arrivals; rate set by subtype, smoking, BMI, age, past surgery and treatment
    activity a(t) chronic inflammation growing with damage, plus flare pulses decaying over ~1-3 weeks
    damage s(t)   monotone and saturating, accumulated through flares plus a slow drift; the true progression stage
                  is 1 + [s >= 1] + [s >= 3] (Hurley I-III)

Subpopulations follow the latent classes of Canoui-Poitrine et al. (2013) plus a rarer severe
inflammatory class. Care seeking, misdiagnosis, referral, diagnosis, treatment escalation (antibiotics,
then biologics once approved for HS) and surgery are simulated alongside and feed back into the course,
so treatment is confounded by indication as in real-world data. Weekly states are kept from
STORE_START on for the emission models in modalities.py.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

WPY = 52.18                                 # weeks per year
SIM_START = pd.Timestamp("1970-01-05")      # Monday of grid week 0
STORE_START = pd.Timestamp("2012-01-02")    # weekly latent states are kept from here on
SIM_END = pd.Timestamp("2025-12-29")
NEVER = 10 ** 6
STAGE_CUTS = (1.0, 3.0)
S_MAX = 8.0                                 # damage saturates below this
FLARE_DECAY = np.exp(-1 / 1.5)              # flare pulses decay with a 1.5-week time constant
UNCONTROLLED = 0.9                          # activity above which a dermatologist escalates treatment

PRIMARY, EMERGENCY, DERMATOLOGY = 1, 2, 3
SETTING_NAMES = {PRIMARY: "primary_care", EMERGENCY: "emergency", DERMATOLOGY: "dermatology"}
INCISION_DRAINAGE, DEROOFING, EXCISION = 1, 2, 3
PC_SEEK = 0.25                              # primary care visit hazard per unit flare size, times care propensity
ED_SEEK = 0.035                             # emergency visit hazard per squared flare size
P_DIAGNOSIS = {PRIMARY: 0.02, EMERGENCY: 0.01, DERMATOLOGY: 0.85}

REGIONS = ["axillary", "inframammary", "inguinal", "gluteal_perianal", "atypical"]

SUBTYPES = [
    dict(name="axillary_mammary", prevalence=0.42, female=0.85, onset_age=(23, 7), bmi=31.0, smoker=0.65,
         family=0.30, flare_rate=6.0, flare_size=0.0, chronic=0.30, damage=0.004, drift=0.004, tunnels=1.0,
         regions=[0.90, 0.60, 0.60, 0.15, 0.10], markers={}),
    dict(name="follicular", prevalence=0.24, female=0.45, onset_age=(18, 5), bmi=26.5, smoker=0.75,
         family=0.50, flare_rate=8.0, flare_size=-0.3, chronic=0.40, damage=0.0025, drift=0.002, tunnels=0.7,
         regions=[0.60, 0.15, 0.50, 0.35, 0.50],
         markers=dict(crp_mg_l=-0.2, il6_pg_ml=-0.1, il17a_pg_ml=0.5)),
    dict(name="gluteal", prevalence=0.22, female=0.55, onset_age=(24, 8), bmi=24.5, smoker=0.80,
         family=0.25, flare_rate=4.0, flare_size=-0.1, chronic=0.20, damage=0.007, drift=0.006, tunnels=1.4,
         regions=[0.25, 0.05, 0.40, 0.90, 0.10],
         markers=dict(crp_mg_l=-0.4, il6_pg_ml=-0.3, il1b_pg_ml=-0.2, il17a_pg_ml=-0.2)),
    dict(name="inflammatory", prevalence=0.12, female=0.40, onset_age=(25, 8), bmi=32.0, smoker=0.70,
         family=0.30, flare_rate=12.0, flare_size=0.35, chronic=0.80, damage=0.015, drift=0.020, tunnels=1.3,
         regions=[0.85, 0.45, 0.80, 0.60, 0.35],
         markers=dict(crp_mg_l=0.6, il6_pg_ml=0.6, il1b_pg_ml=0.8, il17a_pg_ml=0.9, tnfa_pg_ml=0.3)),
]

TOPICAL, TETRACYCLINE, CLINDA_RIF, SHORT_ABX = 1, 2, 3, 7
BIOLOGICS = (4, 5, 6)
TREATMENTS = {  # rate/size/chronic multiply flare rate, flare size and chronic activity while on treatment
    1: dict(name="topical_clindamycin", rate=0.9, size=1.0, chronic=1.0, weeks=12, refill=8, atc=[("D10AF01",)]),
    2: dict(name="tetracycline", rate=0.7, size=0.9, chronic=0.9, weeks=12, refill=4,
            atc=[("J01AA02",), ("J01AA04",), ("J01AA08",)]),
    3: dict(name="clindamycin_rifampicin", rate=0.5, size=0.8, chronic=0.8, weeks=10, refill=4,
            atc=[("J01FF01", "J04AB02")]),
    4: dict(name="adalimumab", approved="2015-07-28", response=0.50, refill=4, atc=[("L04AB04",)]),
    5: dict(name="secukinumab", approved="2023-06-01", response=0.45, refill=4, atc=[("L04AC10",)]),
    6: dict(name="bimekizumab", approved="2024-04-22", response=0.55, refill=4, atc=[("L04AC21",)]),
    7: dict(name="short_antibiotic", rate=0.8, size=0.8, chronic=1.0, weeks=2, refill=99,
            atc=[("J01CR02",), ("J01CF05",), ("J01FF01",), ("J01DB01",)]),
}
RESPONDER = (0.35, 0.75, 0.55)              # biologic multipliers for responders and non-responders
NON_RESPONDER = (0.85, 1.0, 0.95)
TREATMENT_NAMES = np.array(["none"] + [TREATMENTS[k]["name"] for k in range(1, 8)])

COMORBIDITIES = {  # name: (ICD-10-CM code, chronic medication ATC or None)
    "obesity": ("E66.9", None),
    "tobacco": ("F17.210", None),
    "t2d": ("E11.9", "A10BA02"),
    "dyslipidemia": ("E78.5", "C10AA05"),
    "hypertension": ("I10", "C09AA05"),
    "depression": ("F32.9", "N06AB06"),
    "anxiety": ("F41.9", None),
    "ibd": ("K50.90", "A07EC02"),
    "spondyloarthritis": ("M45.9", "M01AE01"),
    "pcos": ("E28.2", "G03HB01"),
    "acne": ("L70.0", "D10AD03"),
    "pilonidal": ("L05.91", None),
}


def week_of(date):
    return (pd.Timestamp(date) - SIM_START).days // 7


def weeks_of(dates):
    return ((pd.DatetimeIndex(dates) - SIM_START).days // 7).to_numpy()


def to_dates(weeks, rng, days=5):
    """Calendar dates inside the given grid weeks, on one of the first `days` weekdays."""
    weeks = np.asarray(weeks, dtype=int)
    return SIM_START + pd.to_timedelta(7 * weeks + rng.integers(0, days, weeks.shape), unit="D")


def stage_of(s):
    return 1 + (np.asarray(s)[..., None] >= np.array(STAGE_CUTS)).sum(-1)


def subtype_param(key):
    return np.array([t[key] for t in SUBTYPES])


def sample_population(n, rng):
    """Static patient attributes for n patients with disease onset between 1970 and mid-2025."""
    draw = 2 * n
    c = rng.choice(len(SUBTYPES), draw, p=subtype_param("prevalence"))
    birth_year = rng.normal(1984, 13, draw).clip(1948, 2008)
    mu, sd = np.array([t["onset_age"] for t in SUBTYPES], dtype=float).T
    onset_age = (mu[c] + sd[c] * rng.standard_normal(draw)).clip(11, 60)
    birth_week = np.round((birth_year - 1970) * WPY).astype(int)
    onset_week = birth_week + np.round(onset_age * WPY).astype(int)
    keep = np.flatnonzero((onset_week >= 0) & (onset_week <= week_of("2025-06-30")))[:n]
    c, birth_year, birth_week, onset_week = c[keep], birth_year[keep], birth_week[keep], onset_week[keep]

    female = rng.random(n) < subtype_param("female")[c]
    bmi = (subtype_param("bmi")[c] * np.exp(0.15 * rng.standard_normal(n))).clip(17, 55)
    smoker = rng.random(n) < subtype_param("smoker")[c]
    susceptibility = rng.lognormal(0, 0.5, n)
    region_p = np.array([t["regions"] for t in SUBTYPES])[c]
    region_p[~female, REGIONS.index("inframammary")] *= 0.2

    def onset(present, age_mu, age_sd, min_age=0):
        age = np.maximum(rng.normal(age_mu, age_sd, n), min_age)
        return np.where(present, birth_week + np.round(age * WPY).astype(int), NEVER)

    def after_hs_onset(present, mean_years):
        return np.where(present, onset_week + np.round(rng.exponential(mean_years, n) * WPY).astype(int), NEVER)

    def draw_(p):
        return rng.random(n) < p

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    inflammatory = c == 3
    comorb = dict(
        obesity=onset(bmi >= 30, 18, 0),
        tobacco=onset(smoker, 17, 2),
        t2d=onset(draw_(sigmoid(-3.0 + 0.15 * (bmi - 30) + 0.5 * inflammatory)), 46, 9, 25),
        dyslipidemia=onset(draw_(sigmoid(-1.9 + 0.08 * (bmi - 30))), 42, 9, 25),
        hypertension=onset(draw_(sigmoid(-1.7 + 0.08 * (bmi - 30) + 0.3 * ~female)), 45, 9, 25),
        depression=after_hs_onset(draw_(0.20 + 0.12 * inflammatory), 5),
        anxiety=after_hs_onset(draw_(0.15), 6),
        ibd=onset(draw_(0.015 + 0.06 * inflammatory), 27, 8, 12),
        spondyloarthritis=onset(draw_(0.02 + 0.07 * inflammatory), 33, 8, 16),
        pcos=onset(draw_(female * (0.07 + 0.06 * (c == 0))), 21, 4, 14),
        acne=onset(draw_(0.2 + 0.3 * (c == 1)), 15, 2, 11),
        pilonidal=onset(draw_(0.05 + 0.2 * (c == 1) + 0.25 * (c == 2)), 21, 4, 13),
    )
    return SimpleNamespace(
        c=c, female=female, birth_year=np.floor(birth_year).astype(int), birth_week=birth_week,
        onset_week=onset_week, bmi=bmi, smoker=smoker,
        psi=rng.lognormal(-0.5 + 0.25 * female, 0.8) * np.where(rng.random(n) < 0.6, 0.03, 1.0),  # care seeking; most rarely consult
        flare_rate=subtype_param("flare_rate")[c] * rng.lognormal(0, 0.4, n) * np.where(smoker, 1.25, 1.0)
        * (1 + 0.025 * np.maximum(bmi - 25, 0)),
        flare_mu=subtype_param("flare_size")[c] - 0.125,     # log-mean of a flare's size; E[size] = exp(flare_size)
        chronic=subtype_param("chronic")[c] * rng.lognormal(0, 0.3, n),
        damage=subtype_param("damage")[c] * susceptibility,
        drift=subtype_param("drift")[c] * susceptibility,
        tunnel_mult=subtype_param("tunnels")[c],
        phototype=rng.choice(6, n, p=[0.06, 0.24, 0.34, 0.20, 0.10, 0.06]) + 1,
        region_p=region_p, region_u=rng.random((n, len(REGIONS))),
        family=rng.random(n) < subtype_param("family")[c],
        comorb=comorb,
    )


def simulate(pop, rng):
    """Weekly disease course, care pathway and treatment for every patient; see the module docstring."""
    n = len(pop.c)
    n_weeks = week_of(SIM_END) + 1
    w0 = week_of(STORE_START)
    S = np.zeros((n_weeks - w0, n), np.float16)
    A, F = np.zeros_like(S), np.zeros_like(S)
    NEW = np.zeros(S.shape, np.uint8)
    TX = np.zeros(S.shape, np.int8)

    s, f = np.zeros(n), np.zeros(n)
    tx, tx_start, tx_end = np.zeros(n, np.int8), np.zeros(n, int), np.full(n, NEVER)
    rate_m, size_m, chronic_m = np.ones(n), np.ones(n), np.ones(n)
    responder, ever_systemic = np.zeros(n, bool), np.zeros(n, bool)
    n_bio, hs_visits, surgeries = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    diagnosed, diag_week = np.zeros(n, bool), np.full(n, NEVER)
    in_derm, next_derm = np.zeros(n, bool), np.full(n, NEVER)
    approved = {code: week_of(TREATMENTS[code]["approved"]) for code in BIOLOGICS}
    response = np.zeros(8)
    response[list(BIOLOGICS)] = [TREATMENTS[code]["response"] for code in BIOLOGICS]
    log = []

    def start(mask, code, w):
        t = TREATMENTS[code]
        tx[mask], tx_start[mask], tx_end[mask] = code, w, w + t["weeks"]
        rate_m[mask], size_m[mask], chronic_m[mask] = t["rate"], t["size"], t["chronic"]
        responder[mask] = False
        if code in (TETRACYCLINE, CLINDA_RIF):
            ever_systemic[mask] = True

    def start_biologic(idx, codes, w):
        resp = rng.random(len(idx)) < response[codes] * np.where(n_bio[idx] > 0, 0.8, 1.0)
        tx[idx], tx_start[idx] = codes, w
        tx_end[idx] = w + np.where(resp, 16 + rng.exponential(3 * WPY, len(idx)), 52).astype(int)
        rate_m[idx], size_m[idx], chronic_m[idx] = np.where(resp[:, None], RESPONDER, NON_RESPONDER).T
        responder[idx] = resp
        n_bio[idx] += 1

    for w in range(max(pop.onset_week.min(), 0), n_weeks):
        active = pop.onset_week <= w
        age = (w - pop.birth_week) / WPY
        ended = (tx > 0) & (tx_end <= w)
        tx[ended], responder[ended] = 0, False
        rate_m[ended] = size_m[ended] = chronic_m[ended] = 1.0

        # disease course
        rate = pop.flare_rate * rate_m * np.exp(-np.maximum(age - 45, 0) / 15) * 0.9 ** surgeries * active
        new = rng.poisson(rate / WPY)
        size = new * np.exp(pop.flare_mu + 0.5 * rng.standard_normal(n)) * size_m
        f = f * FLARE_DECAY + size
        s = s + (pop.damage * size * (1 + 0.3 * np.minimum(s, 3)) + pop.drift / WPY * active) * (1 - s / S_MAX)
        a = active * pop.chronic * (1 + 0.3 * s) * chronic_m + f
        stage = stage_of(s)

        # encounters: flare-triggered primary care / emergency, scheduled or urgent dermatology
        u = rng.random((7, n))
        flare = new > 0
        pc = flare & (u[0] < 1 - np.exp(-PC_SEEK * pop.psi * size * np.where(in_derm, 0.5, 1.0)))
        ed = flare & ~pc & (u[1] < 1 - np.exp(-ED_SEEK * pop.psi * size ** 2))
        due = next_derm <= w
        attend = due & (u[2] < 0.85)
        missed = due & ~attend
        next_derm[missed] = np.where(in_derm[missed] | (u[3][missed] < 0.5), w + 8, NEVER)
        urgent = in_derm & ~due & flare & (size > 1.5) & (u[4] < 0.25)
        derm = attend | urgent

        # diagnosis and referral
        p_dx = np.select([derm, pc, ed], [P_DIAGNOSIS[DERMATOLOGY], P_DIAGNOSIS[PRIMARY], P_DIAGNOSIS[EMERGENCY]], 0.0)
        dx_now = ~diagnosed & (u[5] < p_dx)
        p_ref = np.where(diagnosed, 0.15, np.minimum(0.03 + 0.02 * hs_visits, 0.25) * np.where(stage >= 2, 1.5, 1.0))
        refer = (pc | ed) & ~in_derm & ~derm & (next_derm == NEVER) & (u[6] < p_ref)
        next_derm[refer] = w + 3 + rng.poisson(9, refer.sum())
        hs_visits += pc | ed
        diagnosed |= dx_now
        diag_week[dx_now] = w
        in_derm |= derm

        # treatment: antibiotics for flares outside dermatology, step-up escalation within it
        v = rng.random((7, n))
        start((pc | ed) & (tx == 0) & (v[0] < np.where(ed, 0.7, 0.55)), SHORT_ABX, w)
        drained = ed & (v[1] < 0.45)
        dd = derm & diagnosed
        on_bio = np.isin(tx, BIOLOGICS)
        failing = dd & on_bio & ~responder & (w - tx_start >= 16)
        escalate = (dd & ~on_bio & (a > UNCONTROLLED)) | failing
        avail = np.array([code for code in BIOLOGICS if approved[code] <= w], dtype=np.int8)
        if len(avail):
            idx = np.flatnonzero(escalate & (stage >= 2) & (ever_systemic | on_bio) & (v[2] < 0.25))
            k = rng.integers(len(avail), size=len(idx))
            codes = avail[k]
            same = codes == tx[idx]
            codes[same] = avail[(k[same] + 1) % len(avail)]
            switch = codes != tx[idx]
            start_biologic(idx[switch], codes[switch], w)
            escalate[idx[switch]] = False
        cr = escalate & (stage >= 2) & (v[3] < 0.6)
        tet = escalate & ~cr & ((stage >= 2) | (v[3] < 0.5))
        start(cr, CLINDA_RIF, w)
        start(tet, TETRACYCLINE, w)
        start(escalate & ~cr & ~tet, TOPICAL, w)
        surg = dd & (stage >= 2) & (v[4] < np.where(stage == 3, 0.10, 0.04))
        surg_type = np.where((stage == 3) & (v[5] < 0.6), EXCISION, DEROOFING)
        surgeries += surg

        interval = np.where(np.isin(tx, (TETRACYCLINE, CLINDA_RIF) + BIOLOGICS), 13, 26) + rng.integers(-2, 5, n)
        next_derm[derm] = w + interval[derm]
        lost = attend & (v[6] < 0.07)
        in_derm[lost], next_derm[lost] = False, NEVER

        for setting, mask, proc in ((PRIMARY, pc, np.zeros(n, int)),
                                    (EMERGENCY, ed, np.where(drained, INCISION_DRAINAGE, 0)),
                                    (DERMATOLOGY, derm, np.where(surg, surg_type, 0))):
            idx = np.flatnonzero(mask)
            if len(idx):
                log.append((idx, np.full(len(idx), w), np.full(len(idx), setting), proc[idx], diagnosed[idx]))

        if w >= w0:
            S[w - w0], A[w - w0], F[w - w0], NEW[w - w0], TX[w - w0] = s, a, f, np.minimum(new, 255), tx

    cols = [np.concatenate(col) for col in zip(*log)]
    enc = pd.DataFrame(dict(zip(["p", "w", "setting", "proc", "dx"], cols)))
    enc = enc.sort_values(["p", "w", "setting"], kind="stable").reset_index(drop=True)
    surgery = enc[enc.proc >= DEROOFING]
    return SimpleNamespace(pop=pop, S=S, A=A, F=F, NEW=NEW, TX=TX, w0=w0, enc=enc, diag_week=diag_week,
                           surgery_keys=np.sort(surgery.p.to_numpy() * 100000 + surgery.w.to_numpy()))


def state(sim, p, w):
    """Latent state of patients p at grid weeks w (both arrays), plus surgeries before w."""
    p, w = np.asarray(p, dtype=int), np.asarray(w, dtype=int)
    i = w - sim.w0
    if len(i) and (i.min() < 0 or i.max() >= len(sim.S)):
        raise ValueError("week outside the stored window")
    s = sim.S[i, p].astype(float)
    keys = sim.surgery_keys
    surgeries = np.searchsorted(keys, p * 100000 + w) - np.searchsorted(keys, p * 100000)
    return SimpleNamespace(p=p, w=w, s=s, a=sim.A[i, p].astype(float), f=sim.F[i, p].astype(float),
                           tx=sim.TX[i, p].astype(int), stage=stage_of(s), surgeries=surgeries)


def take(st, mask):
    return SimpleNamespace(**{k: v[mask] for k, v in vars(st).items()})


def encounters_within(sim, p, start, end):
    """Simulated HS-related encounters of patients p inside their [start, end] week windows."""
    win = pd.DataFrame({"p": p, "start": start, "end": end})
    e = sim.enc.merge(win, on="p")
    return e[(e.w >= e.start) & (e.w <= e.end)].reset_index(drop=True)
