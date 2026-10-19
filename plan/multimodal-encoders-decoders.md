# Encoders and decoders for each modality of the HS data

Status: plan only. No code is changed yet.

## 1. Goal

Each modality of `test_data/data/multimodal` gets an encoder in `src/encoders` and a decoder in
`src/decoders`. All encoders write into one shared latent space. The latent SDE learns the evolution in that
space. Each decoder reads the same latent state and gives the likelihood of its own modality.

The plain paths must not change. `discovery/` (SVISE) is not touched. `LatentSDE` with one posterior, one
optional frame encoder and one `Decoder` must give the same loss as now.

## 2. What the data gives

The generator is `test_data/src/multimodal/` (`latent.py`, `modalities.py`, `cohorts.py`).

The true latent state of a patient:

| part | kind | time scale | seen by |
|---|---|---|---|
| damage `s(t)` | dynamic, monotone, saturates at 8 | years | exam, photo, ultrasound, surgery codes |
| flare pulse `f(t)` | dynamic, Poisson jumps | decays in 1.5 weeks | encounter timing, ED visits, abscess counts, self-reported flares |
| activity `a = chronic * (1 + 0.3 s) * tx + f` | dynamic | weeks | serum panel, CRP, pain, DLQI, Doppler |
| subtype `c`, flare rate, susceptibility | static | - | sites, cytokine offsets, tunnels, flare rate |
| care-seeking `psi` (60 % rarely consult) | static | - | encounter rate only |

The cohorts are partial views:

| cohort | patients | dates / patient | modalities |
|---|---|---|---|
| claims_ehr | 4000 | median 28, max 320 | records (ICD, CPT, ATC, LOINC CRP) |
| specialist_imaging | 700 | median 23 (9 visits) | exam, photo, ultrasound (not site C), CRP, hospital records |
| biomarker_study | 220 | 3 (weeks 0/16/52) | serum panel with LOD, short exam (AN, IHS4, Hurley, DLQI) |
| heldout_surgical | 300 | median ~29 | pre-op exam, ultrasound, CRP, ESR, hospital records |
| heldout_community | 800 | 1 | self-report, smartphone photo, point-of-care CRP |

`truth/` gives `s`, `a` and stage at each observation date, the subtype, and the withheld ultrasound of
site C. `diagnostics.txt` gives the "information map": the Spearman values that each modality alone gets. These
values are the baselines for the evaluation (section 9).

## 3. Design rules

1. **A modality is a measurement type, not a cohort.** CRP from claims labs, registry visits, the biomarker
   study and the point-of-care test all go to one `serum` modality. The same decoder head then reads all
   CRP values. These shared heads are the anchors that put all cohorts into the same latent space.
2. **The encoder can read all data. The decoder models only the data that the latent state drives.**
   Dispensings, background codes and scheduled visit dates go into the encoder, but not into the likelihood.
3. **Observed covariates are conditions, not modalities.** Static covariates go into `mu`. Treatment and
   age go into the forcing `u(t)`. The cohort id goes only into the prior of `z(t0)`, because cohort
   selection acts on the state at entry, not on the dynamics.
4. **The loader does all transforms.** Logs, standardization, routing of codes, BLQ flags and context ids
   are done in `src/data/hs.py`. The encoders and decoders get a column spec. They do not know HS column
   names, so they stay usable for other data sets.
5. **Pure ELBO.** No weights per modality in the loss. Each modality term is logged in `terms` for
   diagnostics.

## 4. The shared latent space

- `z(t)`, dimension 4 at first: the dynamic state. The SDE (`DriftNet`, diffusion) acts on `z`.
- `w`, dimension 4 to 8: a static latent per patient, with `q(w) = N(m_w, v_w)` and the prior `N(0, I)`.
  It must carry the subtype, the site set, the flare rate and the care-seeking propensity. Without `w`, the
  drift must put these into `z`, and the records cannot separate "many flares" from "seeks care often".
- `mu`: observed static covariates and their masks.

The drift, the diffusion, the `z0` prior and all decoders read `[z, w, mu]`. The posterior of `z(t)` reads the
tokens of the encoders (section 6). No horseshoe and no sparse library here (encoder mode, see the project
notes); the drift is a neural net.

## 5. Interfaces

New protocols, next to the old ones. The old protocols stay.

`src/encoders/base.py`:

```python
class ModalityEncoder(Protocol):
    def forward(self, x, mask, blq, ctx) -> Tensor:   # (B, N, d_m) -> (B, N, d_token)
        ...
```

`src/decoders/base.py`:

```python
class ModalityDecoder(Protocol):
    def log_prob(self, z, cond, x, mask, blq, ctx) -> Tensor:   # (S, B, N) summed over the columns
        ...
```

`cond` is `[w, mu, u(t)]` at the same times as `z`.

Two containers hold the modalities:

- `encoders/multimodal.py`: `MultimodalEncoder(encoders: dict, columns: dict, d_token, p_drop)`. It cuts
  `x` into the column blocks, calls each encoder, and fuses the tokens at each step:
  `h_k = sum over present m of (e_m(t_k) + E_m)`, with `E_m` a learned modality embedding. It gives
  `feat = [h_k, present_k]` (width `d_token + M`) and `feat_mask`. In training, it drops each modality of a
  patient with probability `p_drop` from the encoder input only (modality dropout). The decoders still see
  that modality, so the posterior must predict it from the other modalities.
- `decoders/multimodal.py`: `MultimodalDecoder(decoders: dict, columns: dict)`. Its
  `log_likelihood(z_obs, z_quad, t_quad, cond_obs, cond_quad, batch)` adds the terms of all modalities and
  gives `(S, B)`, the number of observed values, and one term per modality. It also has
  `extra_times(batch) -> (B, K)` for decoders that need `z` at more times (section 7, self-report).

Batch keys that the HS loader adds (all optional for old data):

| key | shape | content |
|---|---|---|
| `mask` | (B, N, D) | now per value: 0 for NaN inside an observed block |
| `blq` | (B, N, D) bool | value is below the limit of detection; `x` then holds log(LOD) |
| `ctx` | (B, N, K) long | one id per modality: photo device, ultrasound operator, assay plate, CRP source, record setting |
| `window` | (B, 2) | span the records cover (enrolment), for the encounter likelihood |
| `cohort` | (B,) long | only for the `z0` prior |

## 6. Encoders

All tabular modalities use one class, `encoders/tabular.py: TabularEncoder(spec, d_token)`:
input `[x * mask, mask, blq, embed(ctx)]` of the block, then a 2-layer MLP to a token. The records use
`encoders/records.py: RecordsEncoder`.

| modality | encoder | input per date |
|---|---|---|
| `exam` | `TabularEncoder` | nodules, abscesses, draining_tunnels, an_count, ihs4, hurley_stage, dlqi, pain_nrs, 5 sites; ctx: source (clinician or self-report) |
| `photo` | `TabularEncoder` | log lesion_area, erythema, scarring, img_emb_0..7; ctx: device (site_A/B/C, smartphone) |
| `ultrasound` | `TabularEncoder` | fluid collections, tunnels, log max collection, dermal thickness, Doppler grade, SOS-HS stage; ctx: operator |
| `serum` | `TabularEncoder` | log CRP, ESR, IL-6, IL-1b, IL-17A, TNF-a, calprotectin, NLR, with BLQ flags; ctx: plate or CRP source |
| `self_report` | `TabularEncoder` | flares_last_6_months, doctor_diagnosed_hs |
| `records` | `RecordsEncoder` | all codes of one date (ICD, CPT, ATC, LOINC ids), setting, log gap to the last record date |

`RecordsEncoder`: an `EmbeddingBag` (sum) over the raw code ids of a date, plus a setting embedding and the
gap, then an MLP to a token. It reads all codes, also background and drug codes, with no hand mapping.

The temporal posterior stays as it is: `IntegratedPosterior` (best for sparse, irregular data and a fast
state) or `SequencePosterior`. It reads `feat` from `MultimodalEncoder`. `KernelPosterior` cannot be used, as
it needs even time steps.

`encoders/static.py: StaticPosterior` gives `q(w)`: the masked mean of `feat` over all steps of a patient,
together with `mu`, goes into an MLP that gives `(m_w, log v_w)`.

## 7. Decoders

All tabular modalities use one class, `decoders/tabular.py: TabularDecoder(spec, dim_cond)`: a trunk MLP of
`[z, cond, embed(ctx)]`, then one parameter head per column. The column kind sets the likelihood.
`decoders/likelihoods.py` holds the helpers that torch does not have: the ordinal (cumulative logit) and the
left-censored normal.

Column kinds: `normal`, `censored_normal`, `poisson`, `negbin`, `binomial(n)`, `ordinal(k)`, `bernoulli`.

| modality | column | kind | note |
|---|---|---|---|
| `exam` | nodules, abscesses, draining_tunnels | `negbin` | dispersion learned per column |
| | an_count | `negbin`, mean = mean(nodules) + mean(abscesses) | only where the raw counts are missing (biomarker study) |
| | ihs4 | not decoded | a fixed function of the counts; the encoder reads it |
| | hurley_stage | `ordinal(3)` | the head learns the rater confusion |
| | dlqi, pain_nrs | `binomial(30)`, `binomial(10)` | depression in `mu` |
| | 5 sites | `bernoulli` | ctx "self-report" adds a learned offset (community: sensitivity 0.9, false positive 0.05) |
| `photo` | log lesion_area, erythema, scarring | `normal` | phototype in `mu` |
| | img_emb_0..7 | `normal` | offset and noise scale per device (smartphone noise is 1.6 times larger) |
| `ultrasound` | fluid collections | `poisson` | |
| | tunnels | `negbin` | |
| | log max collection | `normal` | only observed when fluid > 0; the loader masks it |
| | dermal thickness | `normal` | offset per operator |
| | Doppler grade, SOS-HS stage | `ordinal(4)`, `ordinal(3)` | |
| `serum` | each log marker | `censored_normal` | offset per assay plate; treatment in `u` (TNF-a under adalimumab, IL-17A under IL-17 drugs); LOD differs by source (0.3, 0.5, 5.0) and comes from `x` |
| `self_report` | flares_last_6_months | `poisson`, mean = `eps + k * int_{t-0.5}^{t} nu(z(s), w) ds` | needs `z` on the last 6 months: `extra_times` gives 8 points there |
| | doctor_diagnosed_hs | `bernoulli` | |

`decoders/records.py: RecordsDecoder` is not tabular. It models the records as a marked point process:

- **Encounter times** for primary care and emergency (flare driven): an inhomogeneous Poisson process per
  setting, `log lambda_s(t) = log nu(z, w, mu, u) + c_s(w, mu)`. The log-likelihood is
  `sum over encounters of log lambda_s(t_e) - int over window of lambda_s dt`. The integral uses the
  quadrature times that the drift residual uses already. `nu` is the same flare head as in `self_report`,
  so "flare" has one meaning in both. `c_s` is the care-seeking part.
- **Dermatology visit times** are not modelled. They follow a schedule (13 or 26 weeks), not the latent
  state. The model is conditioned on them.
- **Marks of each HS-related encounter**: a categorical over the HS-related diagnosis codes (L73.2 or a
  site-specific abscess code; informs sites and diagnosis status) and a categorical over procedures (none,
  incision and drainage, deroofing or excision by site; informs `s`).
- **Routed out**: CRP values go to `serum`. Dispensings go to `u(t)`. Comorbidity and background codes go to
  `mu` as "ever coded" flags.

A date counts as an HS-related encounter if it has a code from a list in the loader config (L73.2, L02.*,
L05.*, K61.0, N61.1, L73.9, L08.9, CPT 10060/10061 and the surgery codes).

## 8. Changes outside src/encoders and src/decoders

These are needed, but they are small and additive.

`src/latent/model.py` (each hook is duck-typed, like `gauge` now):

1. If the encoder has `encode(x, mask, batch)`, use its `(feat, feat_mask)`. Else use the old path.
2. If the decoder has `log_likelihood`, use it, and give it `z` at the quadrature times. Else use
   `decoder(z, mu).log_prob(x)` as now.
3. Add `decoder.extra_times(batch)` to the one posterior call.
4. If `batch["window"]` exists, widen `[t0, t1]` to it. The residual and the encounter integral then use the
   same span.
5. Optional `static` module: sample `w`, add the term `kl_w`, and give `[mu, w]` to the drift, the diffusion
   and the decoders.
6. Optional conditional `z0` prior `N(MLP([mu, w, cohort]))` in place of the fixed `z0_mean`, `z0_logvar`.

`src/data/multimodal.py`: `merge_modalities` sets the mask per value (`isfinite`), not per block. For finite
values the result does not change.

New `src/data/hs.py`: reads the CSV files, makes one row per (patient, date), routes the columns to the
modalities, makes `mu`, `u(t)`, `ctx`, `blq` and `window`, and splits the patients into train, val and test.
Time unit: years since 2012-01-02. `u(t)`: treatment one-hot (none, topical, tetracycline,
clindamycin-rifampicin, adalimumab, IL-17 drug, short antibiotic, unknown), age at `t`, and "L73.2 coded
before `t`". The treatment comes from `current_treatment` where it exists, else from the ATC fills (fill
date plus the refill interval). Batches hold one cohort each and are sorted by length, so that the padding
stays small.

Record windows (assumptions, in the loader config): claims `[enrolment_start, enrolment_end]`; specialist
hospital records `[max(enrolment - 2 y, 2012-01-02), last observation]`; surgical
`[preop - 2 y, min(preop + 2 y, 2025-12-29)]`.

## 9. Steps

Each step is a set of small commits. All tests must pass after each step.

1. **Interfaces.** Protocols, `MultimodalEncoder`, `MultimodalDecoder`, `decoders/likelihoods.py`, and the
   hooks 1 to 4 in `model.py`. No HS data yet. Tests: a parity test (one Gaussian block in
   `MultimodalDecoder` gives the same loss as `GaussianDecoder`, with the same seed), likelihood tests (ordinal
   probabilities add to 1; the censored normal equals `Normal.cdf`; the NB mean), and all old tests.
2. **Tabular modalities.** `TabularEncoder`, `TabularDecoder`, `src/data/hs.py` for specialist_imaging and
   biomarker_study (exam, photo, ultrasound, serum). Shape tests with random masks and rows with no data. A
   short fit on these 920 patients.
3. **Records.** `RecordsEncoder`, `RecordsDecoder`, record windows. Add claims_ehr and the hospital records.
4. **Static latent and held-out cohorts.** `StaticPosterior`, the `z0` prior with cohort, the `self_report`
   decoder, and the evaluation script.

Before each run that can take more than 3 minutes, I ask for permission (project rule).

## 10. Evaluation

Use `truth/` only for evaluation, never for training.

- **Progression and activity**: fit a linear probe from `m(t)` to `s` and to `a` on train patients. Give the
  Spearman value on test patients, per cohort. Compare it with the "all combined" baselines in
  `diagnostics.txt`: claims 0.53 / 0.46, specialist 0.91 / 0.88, biomarker 0.80 / 0.89, community 0.67 / 0.77.
- **Subtype**: a logistic probe on `m_w`. Baselines: claims 0.64, specialist 0.75, biomarker 0.65.
- **Withheld ultrasound**: decode the ultrasound at the site C visits from the other modalities. Compare with
  `truth/withheld_ultrasound.csv`.
- **Cross-modal check**: remove one modality at test time. Predict it from the others. Compare with the same
  model when it sees that modality.
- **Island check**: encode a specialist patient from its hospital records only (a claims-like view), and from
  all its data. The two `m(t)` must be near each other.
- **Held-out cohorts**: stage at pre-op (surgical), and stage from one survey (community).
- **Prior simulation**: start from the posterior mean at the true first date of the patient (project rule).

## 11. Risks

1. **Two time scales.** Flares decay in 1.5 weeks, damage grows over years. 32 quadrature points over 4 years
   miss most flares, so the residual and the encounter integral get a high variance. Mitigation: set the
   quadrature points per year, use `IntegratedPosterior`, and use `NetDiagonalDiffusion` for the jumps.
2. **Islands.** Cohorts with no shared modality can use different parts of the latent space. Mitigation:
   the shared heads (CRP, exam, records), the shared drift, and modality dropout. The island check
   (section 10) measures it.
3. **Informative missing data.** Ultrasound is done more often when the disease is severe. Urgent dermatology
   visits follow flares. Community responders have more symptoms. Only BLQ (censoring) and the encounter
   process are modelled at first.
4. **Confounding by indication.** Treatment starts when activity is high. With treatment in `u`, the drift can
   learn a wrong treatment effect. This is not a goal now.
5. **Care seeking.** The records alone cannot separate the flare rate from the care-seeking propensity. Only
   `w`, the self-report and the imaging cohorts can separate them.
6. **Unclear codes.** L70.0 (acne) is a site code and also a comorbidity code. J01FF01 is a short antibiotic
   and also part of clindamycin-rifampicin. The loader lists decide; some noise stays.
7. **Cohort size.** Claims has 18 times more patients than the biomarker study. Sample the cohorts in a
   balanced way per batch, and keep the ELBO within each batch.

## 12. Open decisions

1. **Generic classes or one class per modality.** Recommended: `TabularEncoder` and `TabularDecoder` with one
   spec per modality, plus the records classes. Other option: one file per modality (`encoders/exam.py`,
   ...), each a thin wrapper of the generic class.
2. **Treatment.** Recommended: a forcing `u(t)`. Other option: a modality with its own decoder, which also
   models the escalation and so the confounding.
3. **Static latent `w`.** Recommended: in step 4, with the records. Other option: from step 2, already
   for the subtype probe on the specialist data.
4. **Token fusion.** Recommended: the sum of tokens plus a presence vector. Other option: a product of
   Gaussian experts, one per modality (MVAE). It handles missing modalities well, but it needs Gaussian nodes
   at each time and does not fit the GRU posteriors.
