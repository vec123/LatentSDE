"""
The figures of reconstruct.py. Colors: the categorical slots and the sequential blue ramp of the project palette
(experiments/latent/run.py uses the same values).
"""
from typing import Dict, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator

from .data import Column

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#222220", "#6b6b66", "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
BLUES.set_bad(SURFACE)
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "font.size": 9,
})


def _clean(ax, grid: bool = True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def _ink_on(value: float) -> str:
    """Text color on a cell of the blue ramp."""
    return SURFACE if np.isfinite(value) and value > 0.55 else INK


def _scatter(ax, col: Column, true: np.ndarray, hat: np.ndarray, below: np.ndarray):
    ok = ~below
    ax.scatter(true[ok], hat[ok], s=9, color=SERIES[0], alpha=0.35, linewidths=0, label="observed")
    if below.any():
        ax.scatter(true[below], hat[below], s=16, facecolors="none", edgecolors=SERIES[1], linewidths=0.9,
                   label="below LOD (x = LOD)")
    values = np.concatenate([true, hat])
    lo, hi = np.nanmin(values), np.nanmax(values)
    if col.kind == "count":  # symlog: linear below 1, log above, ticks at plain numbers
        lim = (min(lo, 0) - 0.3, hi * 1.2 + 0.5)
        ticks = [t for t in (0, 1, 3, 10, 30, 100, 300) if t <= lim[1]]
        for axis, scale in ((ax.xaxis, ax.set_xscale), (ax.yaxis, ax.set_yscale)):
            scale("symlog", linthresh=1)
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(PLAIN)
    elif col.log or col.kind == "censored":
        lim = (lo / 1.15, hi * 1.15)
        subs = (1.0,) if hi / lo > 100 else (1.0, 2.0, 5.0)
        for axis, scale in ((ax.xaxis, ax.set_xscale), (ax.yaxis, ax.set_yscale)):
            scale("log")
            axis.set_major_locator(LogLocator(subs=subs))
            axis.set_minor_formatter(NullFormatter())
            axis.set_major_formatter(PLAIN)
    else:
        pad = 0.04 * (hi - lo)
        lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color=MUTED, lw=1, zorder=0)
    ax.set(xlim=lim, ylim=lim, xlabel="data", ylabel="reconstruction")
    _clean(ax)


def _confusion(ax, labels: Sequence[str], true: np.ndarray, hat: np.ndarray):
    k = len(labels)
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (true.astype(int), hat.astype(int)), 1)
    frac = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    ax.imshow(frac, cmap=BLUES, vmin=0, vmax=1)
    for a in range(k):
        for b in range(k):
            ax.text(b, a, str(counts[a, b]), ha="center", va="center", fontsize=8, color=_ink_on(frac[a, b]))
    ax.set(xticks=range(k), yticks=range(k), xticklabels=labels, yticklabels=labels,
           xlabel="reconstruction", ylabel="data")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def reconstruction(modality: str, cols: Sequence[Column], x: np.ndarray, blq: np.ndarray, pred: np.ndarray,
                   metrics: pd.DataFrame, latent_dim: int, split: str, path: str, ncols: int = 4):
    """One panel per column: data against reconstruction (scatter in data units, or a confusion matrix with the
    counts; the color is the fraction of the row)."""
    rows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(3.0 * ncols, 2.9 * rows), squeeze=False)
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    for j, (ax, col) in enumerate(zip(axes.flat, cols)):
        obs = np.isfinite(x[:, j])
        m = metrics[(metrics["column"] == col.name) & (metrics["source"] == "all")]
        if not obs.any() or m.empty:
            ax.set_visible(False)
            continue
        m = m.iloc[0]
        if col.kind in ("binary", "categorical"):
            hat = (pred[obs, j] > 0.5).astype(float) if col.kind == "binary" else pred[obs, j]
            labels = ("0", "1") if col.kind == "binary" else [str(v) for v in col.levels]
            _confusion(ax, labels, x[obs, j], hat)
            title = f"accuracy {m['accuracy']:.2f}  (constant {m['baseline']:.2f})"
        else:
            true, hat = col.inverse(x[obs, j]), col.inverse(pred[obs, j])
            _scatter(ax, col, true, hat, blq[obs, j])
            title = f"R² {m['r2']:.2f}" + (f",  below LOD kept {m['below_lod']:.2f}" if m["n_below"] > 0 else "")
        ax.set_title(f"{col.name}\n{title}", color=INK, fontsize=9)
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if len(labels) > 1:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(f"{modality}: data against MLD autoencoder reconstruction, {split} patients "
                 f"({len(x)} rows, latent size {latent_dim})", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.04 if len(labels) > 1 else 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def skill_table(metrics: pd.DataFrame, split: str, path: str, min_n: int = 5):
    """Skill of each column (rows) for all rows and for each source (columns). Cells with fewer than min_n values
    stay empty."""
    m = metrics[metrics["modality"] != "records"].copy()
    m.loc[m["n"] < min_n, "skill"] = np.nan
    m["row"] = m["modality"] + " · " + m["column"]
    order = list(dict.fromkeys(m["row"]))
    sources = ["all"] + sorted(s for s in m["source"].unique() if s != "all")
    table = m.pivot_table(index="row", columns="source", values="skill", aggfunc="first").reindex(
        index=order, columns=sources)
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.15 * len(sources) + 3.4, 0.24 * len(order) + 1.6))
    ax.pcolormesh(np.clip(values, 0, 1), cmap=BLUES, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=1.5)
    for a in range(values.shape[0]):
        for b in range(values.shape[1]):
            if np.isfinite(values[a, b]):
                ax.text(b + 0.5, a + 0.5, f"{values[a, b]:.2f}", ha="center", va="center", fontsize=7.5,
                        color=_ink_on(values[a, b]))
    modality = [r.split(" · ")[0] for r in order]
    for a in range(1, len(order)):
        if modality[a] != modality[a - 1]:
            ax.axhline(a, color=INK, lw=0.8)
    ax.set(xticks=np.arange(len(sources)) + 0.5, yticks=np.arange(len(order)) + 0.5, xlim=(0, len(sources)),
           ylim=(len(order), 0))
    ax.set_xticklabels([s.replace("_", "\n") for s in sources], fontsize=8)
    ax.set_yticklabels(order, fontsize=8)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Reconstruction skill, {split} patients: R² (normal, count, censored values above the LOD) or "
                 f"1 - error / error of the constant (binary, categorical)", color=INK, fontsize=9, pad=34)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def training(histories: Dict[str, pd.DataFrame], best: Dict[str, int], path: str):
    """Train and val loss of each modality against the step; the thin line is the kept (best val) step."""
    names = list(histories)
    ncols = 3
    rows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(4.0 * ncols, 2.8 * rows), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, names):
        h = histories[name]
        ax.plot(h["step"], h["train"], color=SERIES[0], lw=2, label="train")
        ax.plot(h["step"], h["val"], color=SERIES[1], lw=2, label="val")
        ax.axvline(best[name], color=MUTED, lw=0.8)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(PLAIN)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set(title=name, xlabel="step", ylabel="loss per row")
        ax.title.set_color(INK)
        _clean(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("MLD autoencoders: reconstruction loss (sum over the observed values of a row)", color=INK,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def records_tokens(tokens: pd.DataFrame, overall: pd.Series, split: str, path: str):
    """Per token of the records (sorted by support): the support, the F1 at threshold 0.5 and the AUROC."""
    t = tokens[tokens["support"] > 0].sort_values("support", ascending=False)
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.17 * len(t) + 1.6), sharey=True,
                             gridspec_kw=dict(width_ratios=(1.1, 1, 1)))
    for ax, key, label in zip(axes, ("support", "f1", "auroc"),
                              ("rows with the token (log)", "F1 (threshold 0.5)", "AUROC")):
        ax.barh(y, t[key], height=0.7, color=SERIES[0])
        ax.set(xlabel=label)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        _clean(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_xscale("log")
    axes[1].set_xlim(0, 1)
    axes[2].set_xlim(0.5, 1)
    axes[0].set(yticks=y, ylim=(len(t) - 0.5, -0.5))
    axes[0].set_yticklabels(t["token"], fontsize=7)
    fig.suptitle(f"records, {split} patients ({int(overall['n'])} dates): micro F1 {overall['micro_f1']:.3f}, "
                 f"exact set {overall['exact']:.3f}, mean Jaccard {overall['jaccard']:.3f}", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def records_examples(truth: np.ndarray, prob: np.ndarray, tokens: Sequence[str], sources: Sequence[str], path: str):
    """Some dates (columns): the reconstructed probability of each token (color), and a ring where the data has
    the token."""
    keep = (truth.any(0)) | (prob > 0.5).any(0)
    truth, prob, tokens = truth[:, keep], prob[:, keep], [t for t, k in zip(tokens, keep) if k]
    fig, ax = plt.subplots(figsize=(0.22 * len(truth) + 3.2, 0.2 * len(tokens) + 1.8))
    im = ax.imshow(prob.T, cmap=BLUES, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    r, c = np.nonzero(truth.T)
    ax.scatter(c, r, s=26, facecolors="none", edgecolors=SERIES[1], linewidths=1.2, label="token in the data")
    ax.set(yticks=range(len(tokens)), xticks=range(len(truth)), xlabel="dates of test patients")
    ax.set_yticklabels(tokens, fontsize=7)
    ax.set_xticklabels([s.split("_")[0] for s in sources], rotation=90, fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("reconstructed probability")
    bar.outline.set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_title("records: data and reconstruction of single dates", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
