#!/usr/bin/env python3
"""Projected S-seed ensemble error for base and genemean (both panels),
against the best simple baseline of the same fold.

Model (per held-out donor for pb_rmse; per gene-error vector for
meanlog_rmse_gene): seeds are exchangeable draws, prediction of an S-seed
ensemble = mean of S single-seed predictions, so
   MSE_S = MSE_inf + (MSE_1 - MSE_inf) / S,   MSE_1 - MSE_inf = D/2,
with D = mean (p_seed0 - p_seed1)^2 (one seed pair per fold: noisy).
pb_rmse is projected per donor and then averaged over donors (evaluator
definition). Checks: S=2 projection reproduces the evaluator's ensemble value.
Reports, per fold, S = 1, 2, 4, 8, inf and the smallest S (<= 64) at which
the projection reaches the best of the reference baselines
(grand_mean, shrunk_stage_mean, ridge, stage_mean_log).
Writes OUT/arms_projection.json and prints markdown.
"""
from arms_common import *  # noqa: F401,F403
from arms_common import FOLDS, FITS, OUT, metric_value, load_baseline_json, load_ens_json
from arms_recompute import load_cache, load_gen, align, lcp

import json
import numpy as np

REFS = ["grand_mean", "shrunk_stage_mean", "ridge", "stage_mean_log"]
SS = [1, 2, 4, 8, float("inf")]


def project(arm, panel, fold, split="heldout"):
    r = load_cache(panel, fold)
    gs = [align(load_gen(FITS / arm / panel / f"fold{fold}" / f"seed{s}" / "generated.npz"), r) for s in [0, 1]]
    idx = np.where(r["held"] if split == "heldout" else ~r["held"])[0]
    obs = lcp(r["pb"][idx])
    p = [np.stack([lcp(g["pb_counts"][t].astype(np.float64)).mean(0) for t in idx]) for g in gs]
    m1 = np.mean([((q - obs) ** 2).mean(1) for q in p], 0)             # per donor mean single-seed MSE
    D = ((p[0] - p[1]) ** 2).mean(1)
    pb = {str(S): float(np.sqrt(np.maximum(m1 - D / 2 + (D / (2 * S) if np.isfinite(S) else 0), 0)).mean()) for S in SS}
    pb_curve = lambda S: float(np.sqrt(np.maximum(m1 - D / 2 + D / (2 * S), 0)).mean())
    e = [(g["mean_log"][idx].astype(np.float64).mean(1) - r["ml"][idx]).mean(0) for g in gs]
    m1e = float(np.mean([(x ** 2).mean() for x in e])); De = float(((e[0] - e[1]) ** 2).mean())
    ml = {str(S): float(np.sqrt(max(m1e - De / 2 + (De / (2 * S) if np.isfinite(S) else 0), 0))) for S in SS}
    ml_curve = lambda S: float(np.sqrt(max(m1e - De / 2 + De / (2 * S), 0)))
    return pb, ml, pb_curve, ml_curve


def first_reach(curve, target):
    for S in range(1, 65):
        if curve(S) <= target:
            return S
    return None


def main():
    out = {}
    lines = ["| arm | panel | fold | metric | S=1 | S=2 (evaluator ens) | S=4 | S=8 | S=inf | best baseline | S to reach |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for panel in ["hvg2000", "wide"]:
        for arm in ["base", "genemean"]:
            for f in FOLDS:
                pb, ml, pbc, mlc = project(arm, panel, f)
                ens = load_ens_json(arm, panel, f)
                for name, proj, curve in [("pb_rmse", pb, pbc), ("meanlog_rmse_gene", ml, mlc)]:
                    bl = {b: metric_value(load_baseline_json(b, panel, f), name, "heldout") for b in REFS}
                    bname = min(bl, key=bl.get)
                    S = first_reach(curve, bl[bname])
                    ev = metric_value(ens, name, "heldout")
                    out[f"{arm}|{panel}|{f}|{name}"] = {"proj": proj, "evaluator_ens": ev, "baselines": bl,
                                                        "best_baseline": bname, "S_to_reach_best_baseline": S}
                    lines.append(f"| {arm} | {panel} | {f} | {name} | {proj['1']:.3f} | {proj['2']:.3f} ({ev:.3f}) | {proj['4']:.3f} | "
                                 f"{proj['8']:.3f} | {proj['inf']:.3f} | {bname} {bl[bname]:.3f} | {'>64' if S is None else S} |")
    (OUT / "arms_projection.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
