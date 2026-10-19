"""Generate the synthetic multimodal hidradenitis suppurativa cohorts (see test_data/src/multimodal/cohorts.py).

Run from the repository root:
    python -m test_data.scripts.multimodal.generate_hs [--seed 0] [--n-pop 40000]

Writes one CSV per table under test_data/data/multimodal/, an overview figure, and diagnostics.txt with
realism statistics and the per-modality information map.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from test_data.src.multimodal.cohorts import generate
from test_data.src.multimodal.diagnostics import information_map, report_all
from test_data.src.multimodal.latent import SIM_START, STAGE_CUTS, SUBTYPES, WPY, weeks_of

SUBTYPE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STAGE_COLORS = ["#86b6ef", "#2a78d6", "#104281"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_trajectories(ax, sim, rng):
    pop = sim.pop
    for k, subtype in enumerate(SUBTYPES):
        cand = np.flatnonzero((pop.c == k) & (pop.onset_week >= sim.w0))
        for j, p in enumerate(rng.choice(cand, 12, replace=False)):
            w = np.arange(pop.onset_week[p], sim.w0 + len(sim.S))
            ax.plot((w - pop.onset_week[p]) / WPY, sim.S[w - sim.w0, p].astype(float), color=SUBTYPE_COLORS[k],
                    lw=1.3, alpha=0.8, label=subtype["name"].replace("_", " ") if j == 0 else None)
    for cut, label in zip(STAGE_CUTS, ("Hurley II", "Hurley III")):
        ax.axhline(cut, color=MUTED, lw=0.9, ls="--")
        ax.text(0.2, cut + 0.08, label, color=INK_2, fontsize=8)
    style(ax, "Latent progression by subtype (12 patients each)")
    ax.set_xlabel("years since onset")
    ax.set_ylabel("damage s")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0.0, 0.92))


def plot_patient(ax, sim, tables):
    visits = tables["specialist_imaging/visits"]
    counts = visits.patient_id.value_counts()
    pid = counts.index[int(0.1 * len(counts))]     # a well-followed but not extreme patient
    p = sim.patient_index[pid]
    vw = weeks_of(visits.visit_date[visits.patient_id == pid])
    rec = tables["specialist_imaging/hospital_records"].query("patient_id == @pid")
    w = np.arange(max(vw.min() - 52, sim.w0), min(vw.max() + 8, sim.w0 + len(sim.S)))
    activity = sim.A[w - sim.w0, p].astype(float)
    ax.plot(SIM_START + pd.to_timedelta(7 * w, unit="D"), activity, color=SUBTYPE_COLORS[0], lw=1.4,
            label="latent activity a(t)")
    ax.scatter(SIM_START + pd.to_timedelta(7 * vw, unit="D"), sim.A[vw - sim.w0, p].astype(float), s=36, color=INK,
               edgecolor=SURFACE, linewidth=1.5, zorder=3, label="registry visit (exam, photo, ultrasound)")
    rec_w = weeks_of(rec.date)
    rec_w = rec_w[(rec_w >= w[0]) & (rec_w <= w[-1])]
    ax.plot(SIM_START + pd.to_timedelta(7 * rec_w, unit="D"), np.full(len(rec_w), -0.06 * activity.max()), "|",
            color=MUTED, ms=9, mew=1.5, label="hospital record event")
    style(ax, f"Registry patient {pid}: most flares fall between visits")
    ax.set_ylabel("activity a")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))


def plot_stages(ax, tables):
    tp = tables["truth/patients"]
    cohorts = list(dict.fromkeys(tp.cohort))
    shares = np.array([[(tp[tp.cohort == c].stage_last_observation == s).mean() for s in (1, 2, 3)] for c in cohorts])
    left = np.zeros(len(cohorts))
    for j, label in enumerate(("Hurley I", "Hurley II", "Hurley III")):
        ax.barh(cohorts, shares[:, j], left=left, color=STAGE_COLORS[j], edgecolor=SURFACE, linewidth=2, height=0.62,
                label=label)
        for i, (l0, width) in enumerate(zip(left, shares[:, j])):
            if width >= 0.08:
                ax.text(l0 + width / 2, i, f"{width:.0%}", ha="center", va="center", fontsize=8,
                        color=INK if j == 0 else "white")
        left += shares[:, j]
    style(ax, "True stage at last observation: selection differs by cohort")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))


def plot_information(ax, info):
    values = info.to_numpy(dtype=float)
    ax.imshow(np.clip(values, 0, 1), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for (i, j), val in np.ndenumerate(values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="white" if val > 0.55 else INK)
    style(ax, "What each modality alone reveals (held-out patients)")
    ax.grid(False)
    ax.set_xticks(range(3), ["progression s\n(Spearman)", "activity a\n(Spearman)", "subtype\n(accuracy)"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(info)), [m for _, m in info.index], fontsize=8.5)
    cohorts = info.index.get_level_values(0)
    for cohort in dict.fromkeys(cohorts):
        rows = np.flatnonzero(cohorts == cohort)
        ax.text(2.6, rows.mean(), cohort.replace("_", " "), rotation=-90, ha="left", va="center", fontsize=9,
                color=INK_2)
        if rows[0] > 0:
            ax.axhline(rows[0] - 0.5, color=SURFACE, lw=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-pop", type=int, default=40000)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "multimodal")
    args = parser.parse_args()

    tables, sim = generate(seed=args.seed, n_pop=args.n_pop)
    for name, table in tables.items():
        path = args.out / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.6g")
        print(f"{path.relative_to(args.out)}: {len(table)} rows")

    info = information_map(tables)
    report = report_all(sim, tables, info)
    (args.out / "diagnostics.txt").write_text(report, encoding="utf-8")
    print(report)

    fig = plt.figure(figsize=(16, 12.5), facecolor=SURFACE)
    grid = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], hspace=0.42, wspace=0.42)
    plot_trajectories(fig.add_subplot(grid[0, 0]), sim, np.random.default_rng(args.seed))
    plot_patient(fig.add_subplot(grid[1, 0]), sim, tables)
    plot_stages(fig.add_subplot(grid[2, 0]), tables)
    plot_information(fig.add_subplot(grid[:, 1]), info)
    fig.savefig(args.out / "overview.png", dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print(f"figure: {args.out / 'overview.png'}")


if __name__ == "__main__":
    main()
