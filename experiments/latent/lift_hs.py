"""
Generalization and transfer lift of runs on test_data/data/multimodal, read from their hs_evaluation.json (run
experiments.latent.evaluate_hs first). Run from the repository root:

    python -m experiments.latent.lift_hs runs/<all_run> runs/<alone_run>

Generalization (Almirall challenge): the score holds on a dataset whose selection mechanism was not in training.
For each run: the scores on the test patients of the training cohorts against those on the held-out cohorts
(heldout_surgical: severe patients; heldout_community: one survey). The drop is held-out minus in-distribution.

Transfer lift: the model trained on all cohorts (first run) against the model trained on the data-poor target alone
(second run), both scored on the target's patients not used for training (val and test pooled): lift = all - alone,
positive when the other cohorts help. The target is the cohort of the second run (biomarker_study: 3 visits each).
"""
import json
import os
import sys

import pandas as pd

METRICS = ["spearman_stage", "spearman_progression", "acc_latent", "ari_component", "ari_latent", "ari_raw"]


def scores(run: str) -> dict:
    return json.load(open(os.path.join(run, "hs_evaluation.json")))["scores"]


def generalization(run: str) -> pd.DataFrame:
    s = scores(run)
    in_dist = pd.DataFrame({k.split(":")[1]: v for k, v in s.items() if k.startswith("test:")}).T
    rows = {}
    for name in ("heldout_surgical", "heldout_community"):
        held = pd.Series(s[f"{name}:{name}"])
        rows[name] = held
    held = pd.DataFrame(rows).T
    cols = [m for m in METRICS if m in held.columns]
    out = pd.concat({"in-distribution (test)": in_dist[cols], "held-out": held[cols]})
    return out.round(3)


def lift(all_run: str, alone_run: str, target: str = "biomarker_study") -> pd.DataFrame:
    a, b = scores(all_run)[f"valtest:{target}"], scores(alone_run)[f"valtest:{target}"]
    rows = {m: {"all cohorts": a[m], "target alone": b[m], "lift": a[m] - b[m]} for m in METRICS if m in a and m in b}
    return pd.DataFrame(rows).T.round(3)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    all_run, alone_run = sys.argv[1], sys.argv[2]
    for run in (all_run, alone_run):
        print(f"\nGeneralization, {os.path.basename(run)}:\n{generalization(run).to_string()}")
    print(f"\nTransfer lift on biomarker_study (val + test patients), {os.path.basename(all_run)} against "
          f"{os.path.basename(alone_run)}:\n{lift(all_run, alone_run).to_string()}")
