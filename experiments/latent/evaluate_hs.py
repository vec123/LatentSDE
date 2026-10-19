"""
Evaluation of runs on test_data/data/multimodal, with the scores of the Almirall challenge (reference/
ALMIRALL_CHALLENGE_2026_WEB.docx.pdf). The truth (test_data/data/multimodal/truth) is used here only, never in
training. Run from the repository root:

    python -m experiments.latent.evaluate_hs runs/<run_dir> [more run dirs ...]

For each run (writes hs_evaluation.json in its folder, and prints one table for all runs):

1. Subtypes: adjusted Rand index (ARI) of a 4-component Gaussian mixture on the patient embedding (the static
   parameters the drift sees: the covariates mu and the posterior mean of w; or the mean of m(t) over the visits
   without w; the raw baseline is the same on the column means and mu), fitted on the train patients, against the
   true subtype; on the test patients of each training cohort and on the two held-out cohorts.
2. Stage: a ridge probe from [m(t), w] to the true progression s, fitted on the train visits; Spearman value on the
   test visits of each cohort and on the held-out cohorts (heldout_surgical: other selection, heldout_community:
   one survey). Also against the Hurley-like stage of truth/observations.csv.
3. Withheld modality: the specialist site C visits have no ultrasound; decoded from the other modalities, against
   truth/withheld_ultrasound.csv (Spearman per column, and the mean).
4. Forecast (always run; the only score that uses the learned drift): from the posterior of the first k visits, the
   drift is rolled out to the next 1 and 2 visits and compared with persistence (no dynamics) and the training mean, on
   the decoded values and on the progression the stage probe reads; see forecast().
The transfer lift is the difference of two runs on the same cohort (e.g. hs_static_biomarker against
hs_static_biomarker_alone on biomarker_study), read from the printed table.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from experiments.latent.run import load_config
from experiments.latent import hs_mld
from src.data.hs import COLUMNS, HELDOUT, ROOT, TRAIN_COHORTS, load_hs

TARGETS = ("progression", "stage")


def _prep(model, batch: dict) -> dict:
    """An end-to-end model encodes the raw rows with its own, trained encoders: the batch with x = those latents."""
    return model.with_encoded(batch) if hasattr(model, "with_encoded") else batch


def _load(split: str, load_kw: dict):
    """The batches and tables of load_hs, or of hs_mld.load_hs_mld (the frozen MLD latents) when load_kw says so."""
    kw = dict(load_kw)
    source = kw.pop("source", "hs")
    return (hs_mld.load_hs_mld if source == "hs_mld" else load_hs)(split, **kw)


def mu_of(model, batch: dict, feat: torch.Tensor, covered: torch.Tensor) -> torch.Tensor:
    """(B, P + dim_w): the static parameters and the posterior mean of w."""
    mu = batch["mu"]
    if model.static is not None:
        mu = torch.cat([mu, model.static(feat, covered, mu)[0]], -1)
    return mu


def encode(model, batch: dict):
    """The posterior mean m (B, N, dim_z) at the steps, the mu with w (B, P + dim_w), and the features."""
    if hasattr(model, "conditioned"):  # a switching model: the batch with its most likely mode appended to mu
        batch = model.conditioned(batch)
    c = model.assign(batch) if getattr(model, "modes", None) is not None else None  # before the posterior is read
    mask = batch["mask"]
    x = torch.where(mask > 0, batch["x"], torch.zeros_like(batch["x"]))
    feat = model.encoder(x, mask) if getattr(model.encoder, "uses_mask", False) else x
    feat_mask = mask.any(-1, keepdim=True).expand_as(feat).to(feat.dtype)
    covered = model.posterior.condition({**batch, "x": feat, "mask": feat_mask})
    # a patient with no observed value the model can read (e.g. columns a model trained on one cohort never saw) has
    # no covered step and a NaN posterior: no information, so the zero state
    post = model.posterior
    # a Kalman posterior is trained as a smoother; the state that is scored is its forward filter (no later visit)
    m = torch.nan_to_num((post.filtered if hasattr(post, "filtered") else post)(batch["t"])[0])
    mu = mu_of(model, batch, feat, covered)
    if getattr(model, "modes", None) is not None:  # mu for drift and decoder: the covariates, then the mode one-hot
        mu = torch.cat([mu, torch.nn.functional.one_hot(c, model.modes.n_modes).to(mu.dtype)], -1)
    return m, mu, covered


def collect(model, split: str, cohorts):
    """
    (visits, patients), joined with the truth. visits: a row per visit with the latent state m0.. (and the posterior
    mean w0.. of the static latent, if the model has one). patients: a row per patient with e0.. (the latent
    embedding: w, then the mean of m over the visits), c0.. (the covariates mu) and r0.. (the mean of each raw column
    over the visits, no model), so that each can be scored alone.
    """
    truth = pd.read_csv(f"{ROOT}/truth/observations.csv", parse_dates=["date"])
    subtype = pd.read_csv(f"{ROOT}/truth/patients.csv")[["cohort", "patient_id", "subtype_id"]]
    visits, patients = [], []
    with torch.no_grad():
        for batch, table in zip(*_load(split, cohorts)):
            batch = _prep(model, batch)
            m, mu, covered = encode(model, batch)
            B, N, d = m.shape
            P = batch["mu"].shape[1]
            w = mu[:, P:]  # the posterior mean of the static latent (B, dim_w), or nothing
            keep = torch.as_tensor(table["date"].notna().to_numpy())
            frame = table[keep.numpy()].copy()
            state = torch.cat([m.reshape(-1, d)[keep], w.repeat_interleave(N, 0)[keep]], -1).numpy()
            frame[[f"m{k}" for k in range(d)] + [f"w{k}" for k in range(w.shape[1])]] = state
            visits.append(frame)
            m_mean = (m * covered.unsqueeze(-1)).sum(1) / covered.sum(1, keepdim=True).clamp_min(1)
            e = torch.cat([w, m_mean], -1)
            r = (batch["x"] * batch["mask"]).sum(1) / batch["mask"].sum(1).clamp_min(1.0)
            pat = table.iloc[::N][["cohort", "patient_id"]].reset_index(drop=True)
            for prefix, value in (("e", e), ("c", batch["mu"]), ("r", r)):
                pat[[f"{prefix}{k}" for k in range(value.shape[1])]] = value.numpy()
            if hasattr(model, "assign"):  # the dynamical mode of each patient
                pat["component"] = model.assign(batch).numpy()
            elif model.static is not None and getattr(model.static, "n_components", 1) > 1:
                pat["component"] = model.static.assign(w).numpy()
            patients.append(pat)
    visits = pd.concat(visits, ignore_index=True).merge(truth, on=["cohort", "patient_id", "date"], how="left")
    patients = pd.concat(patients, ignore_index=True).merge(subtype, on=["cohort", "patient_id"], how="left")
    return visits, patients


def ridge(X: np.ndarray, y: np.ndarray, lam: float = 1e-2):
    A = np.c_[X, np.ones(len(X))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return lambda Z: np.c_[Z, np.ones(len(Z))] @ w


def columns(frame: pd.DataFrame, prefix: str) -> list:
    return [c for c in frame.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]


def score(train: dict, evals: dict) -> dict:
    """
    Per evaluation set and cohort: the stage Spearman values, and for the subtypes
    ari_<x>: adjusted Rand index of a 4-component diagonal Gaussian mixture on x, fitted on the train patients;
    acc_<x>: accuracy of a logistic regression on x, fitted on the train patients with the true subtype;
    where x is latent (the model's patient embedding), cov (sex, BMI, smoking), raw (column means over the visits,
    no model), latent_cov and raw_cov (both together). majority is the accuracy of always naming the most common
    subtype of the set. train and evals map a name to (visits, patients).
    """
    tv, tp = train["train"]
    feats = columns(tv, "m") + columns(tv, "w")
    tv = tv.dropna(subset=["progression"])
    fits = {t: ridge(tv[feats].to_numpy(), tv[t].to_numpy()) for t in TARGETS}
    groups = {"latent": ["e"], "cov": ["c"], "raw": ["r"], "latent_cov": ["e", "c"], "raw_cov": ["r", "c"]}
    cols = {g: sum((columns(tp, p) for p in ps), []) for g, ps in groups.items()}
    stats = {g: (tp[c].to_numpy().mean(0), tp[c].to_numpy().std(0) + 1e-8) for g, c in cols.items()}

    def z(g, frame):
        return (frame[cols[g]].to_numpy() - stats[g][0]) / stats[g][1]

    labelled = tp.dropna(subset=["subtype_id"])
    gmms = {g: GaussianMixture(4, covariance_type="diag", n_init=5, random_state=0, reg_covar=1e-3).fit(z(g, tp))
            for g in ("latent", "cov", "raw")}
    probes = {g: LogisticRegression(max_iter=3000).fit(z(g, labelled), labelled["subtype_id"]) for g in groups}
    out = {}
    for name, (v, p) in evals.items():
        v = v.dropna(subset=["progression"])
        for cohort in sorted(set(v["cohort"]) | set(p["cohort"])):
            vc, pc = v[v["cohort"] == cohort], p[p["cohort"] == cohort]
            row = {f"ari_{g}": float(adjusted_rand_score(pc["subtype_id"], gmm.predict(z(g, pc))))
                   for g, gmm in gmms.items()}
            row.update({f"acc_{g}": float(probe.score(z(g, pc), pc["subtype_id"])) for g, probe in probes.items()})
            row["majority"] = float(pc["subtype_id"].value_counts(normalize=True).iloc[0])
            if "component" in pc:  # the most likely component of the mixture prior of w
                row["ari_component"] = float(adjusted_rand_score(pc["subtype_id"], pc["component"]))
            for t in TARGETS:
                row[f"spearman_{t}"] = float(spearmanr(fits[t](vc[feats].to_numpy()), vc[t]).statistic)
            out[f"{name}:{cohort}"] = row
    return out


def withheld_ultrasound(model, cohorts) -> dict:
    withheld = pd.read_csv(f"{ROOT}/truth/withheld_ultrasound.csv", parse_dates=["visit_date"])
    us_cols = [c for c in withheld.columns if c.startswith("us_")]
    index = [COLUMNS.index(c) for c in us_cols]
    rows = []
    with torch.no_grad():
        for batch, table in zip(*_load("test", cohorts)):
            m, mu, _ = encode(model, batch)
            decoded = model.decoder(m, mu[:, None].expand(-1, m.shape[1], -1)).mean[..., index]
            keep = table["date"].notna().to_numpy()
            frame = table[keep].copy()
            frame[us_cols] = decoded.reshape(-1, len(index))[torch.as_tensor(keep)].numpy()
            rows.append(frame)
    if not rows:
        return {}
    pred = pd.concat(rows).merge(withheld, left_on=["patient_id", "date"], right_on=["patient_id", "visit_date"],
                                 suffixes=("_pred", ""))
    if pred.empty:
        return {}
    out = {c: float(spearmanr(pred[c + "_pred"], pred[c], nan_policy="omit").statistic) for c in us_cols}
    return {"n": len(pred), "mean": float(np.mean(list(out.values()))), **out}


def withheld_ultrasound_mld(model, load_kw: dict) -> dict:
    """
    The withheld ultrasound for a model of frozen MLD latents: the forward-filtered latent of the ultrasound block at the
    test visits of the specialist site C (which have no ultrasound), decoded by the frozen ultrasound decoder, against
    truth/withheld_ultrasound.csv (Spearman per column).
    """
    withheld = pd.read_csv(f"{ROOT}/truth/withheld_ultrasound.csv", parse_dates=["visit_date"])
    modalities = list(load_kw.get("modalities", hs_mld.MODALITIES))
    sizes = hs_mld.latent_sizes(modalities, load_kw.get("mld_dir", "runs/mld_ae"))
    start = sum(sizes[:modalities.index("ultrasound")])
    block = slice(start, start + sizes[modalities.index("ultrasound")])
    rows = []
    with torch.no_grad():
        for batch, table in zip(*_load("test", load_kw)):
            batch = _prep(model, batch)
            m, _, _ = encode(model, batch)
            keep = table["date"].notna().to_numpy()
            frame = table[keep].copy()
            z = m[..., block].reshape(-1, block.stop - block.start)[torch.as_tensor(keep)].numpy()
            units, names = (model.bank.decode_units("ultrasound", z) if hasattr(model, "bank")
                            else hs_mld.decode_block("ultrasound", z, load_kw.get("mld_dir", "runs/mld_ae")))
            frame[names] = units
            rows.append(frame)
    us_cols = [c for c in withheld.columns if c.startswith("us_")]
    pred = pd.concat(rows).merge(withheld, left_on=["patient_id", "date"], right_on=["patient_id", "visit_date"],
                                 suffixes=("_pred", ""))
    if pred.empty:
        return {}
    out = {c: float(spearmanr(pred[c + "_pred"], pred[c], nan_policy="omit").statistic) for c in us_cols}
    return {"n": len(pred), "mean": float(np.mean(list(out.values()))), **out}


def forecast(model, fits: dict, feats: list, load_kw: dict, horizons=(1, 2), substeps: int = 20,
             max_prefixes: int = 4) -> dict:
    """
    The test of the learned SDE, kept in every evaluation. For each test patient and up to max_prefixes prefixes of k >= 2
    visits: the posterior of the prefix gives the latent state at its last visit (a filter, if the posterior is a forward posterior);
    the learned drift (no noise, the treatment of the last visit held fixed) carries it to each of the next `horizons`
    visits. At that visit it is compared with persistence (the state of the last visit, no dynamics) and the training
    mean (0): the decoded values on the observed columns (standardized, mean squared error), and the progression the stage
    probe reads from the latent against the truth. drift_gain = 1 - mse_forecast / mse_persistence: positive when the
    drift beats standing still. The same is reported for the posterior's own extrapolation to the target time (the
    network's state at that time given the prefix only): the forecast of the encoder side, without the SDE.
    """
    truth = pd.read_csv(f"{ROOT}/truth/observations.csv", parse_dates=["date"]).set_index(
        ["cohort", "patient_id", "date"])["progression"]
    out = {}
    acc = {h: {"f": [], "p": [], "e": [], "0": [], "prog_f": [], "prog_p": [], "prog_e": []} for h in horizons}
    seq = ("t", "x", "mask", "u")
    with torch.no_grad():
        for batch, table in zip(*_load("test", load_kw)):
            batch = _prep(model, batch)
            N, P = batch["t"].shape[1], batch["mu"].shape[1]
            for b in range(batch["t"].shape[0]):
                n = int(table["date"].iloc[b * N:(b + 1) * N].notna().sum())
                prefixes = list(range(2, n))
                if len(prefixes) > max_prefixes:
                    prefixes = [prefixes[i] for i in np.linspace(0, len(prefixes) - 1, max_prefixes).round().astype(int)]
                for k in prefixes:
                    sub = {key: (v[b:b + 1, :k] if key in seq or key.startswith(("raw_", "blq_")) else v[b:b + 1])
                           for key, v in batch.items()}  # the visit tensors cut to the prefix, the static ones kept
                    m, mu, _ = encode(model, sub)
                    state = m[:, -1]
                    extra = mu[:, P:]
                    for h in horizons:
                        j = k - 1 + h
                        if j > n - 1:
                            continue
                        x, mask = batch["x"][b, j], batch["mask"][b, j] > 0
                        if not mask.any():
                            continue
                        t0, t1 = sub["t"][0, -1], batch["t"][b, j]
                        z, u = state.clone(), sub["u"][:, -1]
                        for _ in range(substeps):
                            z = z + model.drift(z, mu, u) * (t1 - t0) / substeps
                        # the posterior's own extrapolation to the target time: with a forward posterior, the state the
                        # network gives at t1 from the prefix alone (it has the GRU memory; the drift has none)
                        post = model.posterior
                        extrap = torch.nan_to_num((post.filtered if hasattr(post, "filtered") else post)(t1.view(1, 1))[0][:, 0])
                        for name, st in (("f", z), ("p", state), ("e", extrap)):
                            acc[h][name].append(((model.decoder(st, mu).mean[0] - x)[mask] ** 2).mean().item())
                        acc[h]["0"].append((x[mask] ** 2).mean().item())
                        key = (table["cohort"].iloc[b * N], table["patient_id"].iloc[b * N],
                               table["date"].iloc[b * N + j])
                        if key in truth.index:
                            y = float(truth.loc[key])
                            for name, st in (("prog_f", z), ("prog_p", state), ("prog_e", extrap)):
                                acc[h][name].append(float(fits["progression"](torch.cat([st, extra], -1).numpy())[0]) - y)
    for h in horizons:
        a = acc[h]
        if not a["f"]:  # no patient has enough visits for this horizon (e.g. 3 visits each: no 2-step forecast)
            continue
        mse = {k: float(np.mean(a[k])) for k in ("f", "p", "e", "0")}
        rmse = {k: float(np.sqrt(np.mean(np.square(a[k])))) for k in ("prog_f", "prog_p", "prog_e") if a[k]}
        out[f"h{h}"] = {"n_pairs": len(a["f"]), "mse_forecast": mse["f"], "mse_persistence": mse["p"], "mse_mean": mse["0"],
                        "skill_vs_mean": 1 - mse["f"] / mse["0"], "drift_gain": 1 - mse["f"] / mse["p"],
                        "share_forecast_beats_persistence": float(np.mean(np.array(a["f"]) < np.array(a["p"]))),
                        "mse_posterior_extrapolation": mse["e"], "extrapolation_gain": 1 - mse["e"] / mse["p"],
                        "share_extrapolation_beats_persistence": float(np.mean(np.array(a["e"]) < np.array(a["p"]))),
                        "progression_rmse_forecast": rmse.get("prog_f"), "progression_rmse_persistence": rmse.get("prog_p"),
                        "progression_rmse_extrapolation": rmse.get("prog_e")}
    return out


def evaluate_run(run_dir: str) -> dict:
    torch.set_num_threads(1)
    config = load_config(os.path.join(run_dir, "config.yaml"))
    d = config["data"]["train"]  # the loader settings of the training data, used for every split
    cohorts = {"cohorts": d.get("cohorts", list(TRAIN_COHORTS)), "covariates": d.get("covariates", "basic")}
    if d.get("source") == "hs_mld":
        cohorts.update(source="hs_mld", mld_dir=d.get("mld_dir", "runs/mld_ae"),
                       modalities=d.get("modalities", list(hs_mld.MODALITIES)), raw=d.get("raw", False))
    model = torch.load(os.path.join(run_dir, "model.pt"), weights_only=False, map_location="cpu")
    model.eval()
    train = {"train": collect(model, "train", cohorts)}
    tv = train["train"][0].dropna(subset=["progression"])
    feats = columns(tv, "m") + columns(tv, "w")
    fits = {t: ridge(tv[feats].to_numpy(), tv[t].to_numpy()) for t in TARGETS}
    evals = {"test": collect(model, "test", cohorts)}
    val = collect(model, "val", cohorts)
    # val and test pooled: neither was used for training, and twice the patients of test alone (the transfer lift)
    evals["valtest"] = tuple(pd.concat([a, b], ignore_index=True) for a, b in zip(val, evals["test"]))
    for name in HELDOUT:
        evals[name] = collect(model, name, cohorts)
    withheld = withheld_ultrasound_mld if cohorts.get("source") == "hs_mld" else withheld_ultrasound
    result = {"scores": score(train, evals), "withheld_ultrasound_spearman": withheld(model, cohorts),
              "forecast": forecast(model, fits, feats, cohorts)}
    json.dump(result, open(os.path.join(run_dir, "hs_evaluation.json"), "w"), indent=1)
    return result


def table(results: dict) -> pd.DataFrame:
    rows = {}
    for run, r in results.items():
        flat = {f"{k} {m}": v for k, row in r["scores"].items() for m, v in row.items()}
        flat["withheld_us mean"] = r["withheld_ultrasound_spearman"].get("mean", np.nan)
        for h, f in r["forecast"].items():
            flat[f"forecast {h}: skill vs mean"] = f["skill_vs_mean"]
            flat[f"forecast {h}: drift gain vs persistence"] = f["drift_gain"]
            flat[f"forecast {h}: share drift beats persistence"] = f["share_forecast_beats_persistence"]
            flat[f"forecast {h}: posterior extrapolation gain vs persistence"] = f["extrapolation_gain"]
            flat[f"forecast {h}: progression rmse drift / persistence / extrapolation"] = (
                f"{f['progression_rmse_forecast']:.3f} / {f['progression_rmse_persistence']:.3f} / "
                f"{f['progression_rmse_extrapolation']:.3f}")
        rows[os.path.basename(run)] = flat
    return pd.DataFrame(rows).round(3)


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(table({run: evaluate_run(run) for run in sys.argv[1:]}).to_string())
