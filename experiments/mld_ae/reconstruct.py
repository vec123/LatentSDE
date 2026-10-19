"""
Encodes and decodes each modality with its trained MLD autoencoder (train.py), and measures the quality of the
reconstruction.

    python -m experiments.mld_ae.reconstruct [--out runs/mld_ae] [--split test] [--modalities exam ...]

For each modality with <out>/<modality>/model.pt:
- encode: all rows of the modality -> z (not standardized; z_mean and z_std are in model.pt).
  Writes <out>/<modality>/latents.csv: the ids, the split and z_0, z_1, ...
- decode: z of the rows of --split -> the reconstruction in data units. Writes
  <out>/<modality>/reconstruction_<split>.csv (tabular modalities): <column> (data) and <column>_hat. For a binary
  column, _hat is the probability; for a censored column, <column>_blq is 1 where the data holds the LOD.
- metrics of --split, per column and per source: <out>/metrics_<split>.csv, and for the records per token:
  <out>/records/tokens_<split>.csv.
- plots: <out>/plots/reconstruction_<modality>_<split>.png, skill_<split>.png, training.png,
  records_tokens_<split>.png, records_examples_<split>.png.

Metrics. skill is the part of the error of a constant prediction that the autoencoder removes: 1 is perfect, 0 is
not better than the constant.
- normal, count, censored: r2 (= skill) in the model space (the standardized value, log or log1p), on the values
  above the LOD; mae in data units. Below the LOD: below_lod, the part of the reconstructions at or below the LOD.
- binary, categorical: accuracy, baseline (the accuracy of the most frequent value), skill =
  1 - (1 - accuracy) / (1 - baseline).
- records: per token precision, recall, F1 (threshold 0.5) and AUROC; per date the exact match of the token set
  and the Jaccard index; micro_f1 over all tokens.
"""
import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from . import data, plots
from .model import ModalityAE, load

CHUNK = 8192


def encode(model: ModalityAE, table: pd.DataFrame) -> np.ndarray:
    """The rows of a modality table -> z (n, latent_dim)."""
    x, blq = data.arrays(table, model.columns)
    x, blq = torch.as_tensor(x), torch.as_tensor(blq)
    with torch.no_grad():
        return torch.cat([model.encode(x[i:i + CHUNK], blq[i:i + CHUNK]) for i in range(0, len(x), CHUNK)]).numpy()


def decode(model: ModalityAE, z: np.ndarray) -> np.ndarray:
    """z (n, latent_dim) -> the reconstruction (n, D) in the model space (see ModalityAE.predict); data.to_units
    gives the data units."""
    z = torch.as_tensor(z)
    with torch.no_grad():
        return torch.cat([model.predict(model.decode(z[i:i + CHUNK])) for i in range(0, len(z), CHUNK)]).numpy()


def _groups(sources: np.ndarray):
    yield "all", np.ones(len(sources), bool)
    for s in sorted(set(sources)):
        yield s, sources == s


def column_metrics(modality: str, cols, x: np.ndarray, blq: np.ndarray, pred: np.ndarray,
                   sources: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, col in enumerate(cols):
        for source, group in _groups(sources):
            obs = group & np.isfinite(x[:, j])
            if not obs.any():
                continue
            row = dict(modality=modality, column=col.name, kind=col.kind, source=source, n=int(obs.sum()))
            if col.kind in ("binary", "categorical"):
                true = x[obs, j]
                hat = (pred[obs, j] > 0.5).astype(float) if col.kind == "binary" else pred[obs, j]
                accuracy = float((hat == true).mean())
                baseline = float(np.bincount(true.astype(int)).max() / len(true))
                skill = 1 - (1 - accuracy) / (1 - baseline) if baseline < 1 else np.nan
                row.update(accuracy=accuracy, baseline=baseline, skill=skill)
            else:
                ok, below = obs & ~blq[:, j], obs & blq[:, j]
                true, hat = x[ok, j], pred[ok, j]
                spread = ((true - true.mean()) ** 2).sum()
                r2 = 1 - ((hat - true) ** 2).sum() / spread if ok.sum() > 1 and spread > 0 else np.nan
                mae = float(np.abs(col.inverse(hat) - col.inverse(true)).mean()) if ok.any() else np.nan
                row.update(r2=r2, skill=r2, mae=mae, n_below=int(below.sum()),
                           below_lod=float((pred[below, j] <= x[below, j] + 1e-6).mean()) if below.any() else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def records_metrics(truth: np.ndarray, prob: np.ndarray, tokens: List[str],
                    sources: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per token (all sources) and per source (all tokens)."""
    hat = prob > 0.5
    true = truth > 0.5
    tp, fp, fn = (hat & true).sum(0), (hat & ~true).sum(0), (~hat & true).sum(0)
    auroc = [roc_auc_score(true[:, k], prob[:, k]) if 0 < true[:, k].sum() < len(true) else np.nan
             for k in range(len(tokens))]
    with np.errstate(invalid="ignore", divide="ignore"):
        per_token = pd.DataFrame(dict(token=tokens, support=true.sum(0), predicted=hat.sum(0),
                                      precision=tp / (tp + fp), recall=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn),
                                      auroc=auroc))
    rows = []
    for source, group in _groups(sources):
        h, t = hat[group], true[group]
        both, union = (h & t).sum(1), (h | t).sum(1)
        tp_all, fp_all, fn_all = (h & t).sum(), (h & ~t).sum(), (~h & t).sum()
        rows.append(dict(modality="records", column="(all tokens)", kind="multi-hot", source=source,
                         n=int(group.sum()), exact=float((h == t).all(1).mean()),
                         jaccard=float(np.where(union > 0, both / np.maximum(union, 1), 1.0).mean()),
                         micro_f1=float(2 * tp_all / (2 * tp_all + fp_all + fn_all))))
    return per_token, pd.DataFrame(rows)


def reconstruction_frame(table: pd.DataFrame, cols, x: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    out = table[list(data.IDS)].reset_index(drop=True)
    hat = data.to_units(pred, cols)
    for j, col in enumerate(cols):
        out[col.name] = table[col.name].to_numpy()  # the raw value: the float32 round trip makes 0 -> 1e-8
        out[f"{col.name}_hat"] = pred[:, j] if col.kind == "binary" else hat[:, j]
        if col.kind == "censored":
            out[f"{col.name}_blq"] = table[f"{col.name}_blq"].to_numpy()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join("runs", "mld_ae"))
    parser.add_argument("--split", default="test", choices=data.SPLITS)
    parser.add_argument("--modalities", nargs="+", default=list(data.MODALITIES), choices=data.MODALITIES)
    args = parser.parse_args()
    split = pd.read_csv(os.path.join(args.out, "split.csv"))
    figures = os.path.join(args.out, "plots")
    os.makedirs(figures, exist_ok=True)
    metrics, histories, best = [], {}, {}
    for modality in args.modalities:
        folder = os.path.join(args.out, modality)
        if not os.path.exists(os.path.join(folder, "model.pt")):
            print(f"{modality}: no model.pt, skipped")
            continue
        model, ckpt = load(os.path.join(folder, "model.pt"))
        histories[modality], best[modality] = pd.read_csv(os.path.join(folder, "history.csv")), ckpt["best_step"]
        table, _ = data.load(modality, split)
        cols = model.columns
        # encode all rows, decode the rows of the split
        z = encode(model, table)
        latents = table[list(data.IDS) + ["split"]].reset_index(drop=True)
        latents = pd.concat([latents, pd.DataFrame(z, columns=[f"z_{k}" for k in range(z.shape[1])])], axis=1)
        latents.to_csv(os.path.join(folder, "latents.csv"), index=False)
        rows = (table["split"] == args.split).to_numpy()
        part = table[rows]
        x, blq = data.arrays(part, cols)
        pred = decode(model, z[rows])
        sources = part["source"].to_numpy()
        if modality == "records":
            tokens, m = records_metrics(x, pred, [c.name for c in cols], sources)
            tokens.to_csv(os.path.join(folder, f"tokens_{args.split}.csv"), index=False)
            plots.records_tokens(tokens, m.iloc[0], args.split,
                                 os.path.join(figures, f"records_tokens_{args.split}.png"))
            rng = np.random.default_rng(0)  # the same number of dates from each source
            pick = np.concatenate([rng.choice(np.flatnonzero(sources == s), size=min(16, (sources == s).sum()),
                                              replace=False) for s in sorted(set(sources))])
            plots.records_examples(x[pick], pred[pick], [c.name for c in cols], sources[pick],
                                   os.path.join(figures, f"records_examples_{args.split}.png"))
            print(f"{modality:12s} {len(x):6d} rows: micro F1 {m.iloc[0]['micro_f1']:.3f}, exact set "
                  f"{m.iloc[0]['exact']:.3f}, Jaccard {m.iloc[0]['jaccard']:.3f}")
        else:
            m = column_metrics(modality, cols, x, blq, pred, sources)
            reconstruction_frame(part, cols, x, pred).to_csv(
                os.path.join(folder, f"reconstruction_{args.split}.csv"), index=False)
            plots.reconstruction(modality, cols, x, blq, pred, m, model.latent_dim, args.split,
                                 os.path.join(figures, f"reconstruction_{modality}_{args.split}.png"))
            overall = m[m["source"] == "all"]
            print(f"{modality:12s} {len(x):6d} rows: mean skill {overall['skill'].mean():.3f}, lowest "
                  f"{overall['skill'].min():.3f} ({overall.loc[overall['skill'].idxmin(), 'column']})")
        metrics.append(m)
    if not metrics:
        return
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_csv(os.path.join(args.out, f"metrics_{args.split}.csv"), index=False)
    plots.skill_table(metrics, args.split, os.path.join(figures, f"skill_{args.split}.png"))
    plots.training(histories, best, os.path.join(figures, "training.png"))


if __name__ == "__main__":
    main()
