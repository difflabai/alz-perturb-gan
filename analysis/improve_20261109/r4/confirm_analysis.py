#!/usr/bin/env python3
"""Pre-declared primary analysis of r4/PROTOCOL_CONFIRM.md (extra-fold confirmation, 2 x 2).

Factors (both two-level), arms trained on inner folds 3-8, seeds 0 and 1, 20,000 updates:

    cell A  hvg2000   pbulk            (no external donors, no dispersion change)   label  pbulk~{ref}
    cell B  hvg2000   g.dispz,pbulk    (no external donors, donor-dependent disp.)  label  g.dispz,pbulk~{ref}
    cell C  hvg2000x  pbulk            (external MADRC donors pooled)               label  pbulk~x
    cell D  hvg2000x  g.dispz,pbulk    (both)                                       label  g.dispz,pbulk~x

({ref} = the tag of the hvg2000 fits root given to evaluate_ext.py --ref-root, default r3.)  Every number is
read from the evaluate_ext.py output tree (``--results``, default results_r4): single fits from ``fits/``
and ``ref/fits/``, two-seed ensembles from ``ensembles/`` and ``ref/ensembles/``, baselines from
``baselines/hvg2000/``; all scored on SEA-AD held-out donors of hvg2000 fold k by the r3 evaluator path.

PRIMARY (decision).  Read-out ``snapens``, metric held-out ``pb_rmse``.  For every (fold, seed) with all four
cells present (a complete 2 x 2 block):
    dispersion effect   e_disp = ((B - A) + (D - C)) / 2
    external effect     e_ext  = ((C - A) + (D - B)) / 2
    interaction         e_int  = (D - C) - (B - A)
Fold value = mean over that fold's complete seeds; repeat value = mean over its folds (repeat 1 = folds 3-5,
repeat 2 = folds 6-8); pooled value = mean of the two repeat values (r3 ``repeat_balanced``).
Seed-noise SE (the evaluator's definition, r3 ``seed_noise_se``, generalised to the 2 x 2): s^2 = pooled
within-fold seed variance (ddof 1, folds with >= 2 seeds) of the REFERENCE level of the factor (cells A and C
for dispersion, A and B for external donors, all four cells for the interaction), else the round-2 prior
0.0156^2.  Each (fold, seed) main-effect term has variance s^2 (interaction 4 s^2), so for a repeat with
folds F and k_f complete seeds SE_r = sqrt(sum_f s^2 / k_f) / |F| (x 2 for the interaction) and the pooled
SE = sqrt(SE_r1^2 + SE_r2^2) / 2.  The s^2 of all four cells and the empirical SE over folds are reported
as sensitivity only.
DECISION RULE (pre-declared): a factor is CONFIRMED iff its effect is favourable (negative: lower pb_rmse)
in BOTH repeats AND the pooled effect is below -2 SE.  If any of the 12 blocks (6 folds x 2 seeds; 48 fits) is
missing the verdict is printed as PROVISIONAL with the list of missing fits.

SECONDARY (no decision rule; PROTOCOL_CONFIRM "Secondary read-outs"): the same effects at ``stagehalf`` and
``biascal@stagehalf`` (single fits, same SE rule); two-seed ensembles of snapens, stagehalf and
biascal@stagehalf (one value per fold and cell; effects per fold, repeat means, folds favourable, empirical SE
over folds); the six distribution metrics (meanlog_rmse_gene, nonzero_rmse_gene, logvar_rmse_gene,
pc_energy_rep, cell_energy_rep, |pc_coverage80_rep - 0.8|) of every cell and read-out against real-donor
resampling (``baseline:resample``; for hvg2000x cells also ``baseline:resample+ext``), paired by fold.
REFERENCE ROWS on the same folds: grand_mean, ridge, resample, resample_blind (the ideal stage-blind
generator), and for hvg2000x the baselines with external donors (+ext, +extoff, ridge_cohort+ext).
EXPLORATORY: the same 2 x 2 on the original folds 0-2 (selection-conditioned; never part of the decision).
Nothing else may be added to the primary analysis (PROTOCOL_CONFIRM).

Usage (from /tmp):
    python3 -B r4/confirm_analysis.py [--results DIR] [--ref-tag r3] [--json PATH] [--txt PATH]
Default outputs: DIR/confirm_analysis.json and DIR/confirm_analysis.txt.  Runs on whatever exists and lists
every missing fit, ensemble and baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent

CONFIRM_REPEATS = {1: (3, 4, 5), 2: (6, 7, 8)}
ORIGINAL_REPEATS = {0: (0, 1, 2)}
SEEDS = (0, 1)
PRIMARY_MODE = "snapens"
SECONDARY_MODES = ("stagehalf", "biascal@stagehalf")
ENSEMBLE_MODES = ("snapens", "stagehalf", "biascal@stagehalf")
ENS_SUFFIX = " [seed-ensemble n=2]"
METRIC = "pb_rmse"
DIST_METRICS = ("meanlog_rmse_gene", "nonzero_rmse_gene", "logvar_rmse_gene", "pc_energy_rep", "cell_energy_rep",
                "pc_coverage80_rep")
SEED_SD_PRIOR = 0.0156          # r3 evaluate.SEED_SD_PRIOR['pb_rmse'] (round-2 prior)
REF_BASELINES = ("grand_mean", "ridge", "resample", "resample_blind")
X_BASELINES = ("grand_mean+ext", "ridge+ext", "resample+ext", "resample_blind+ext", "grand_mean+extoff", "ridge+extoff",
               "resample+extoff", "resample_blind+extoff", "ridge_cohort+ext")
CELLS = ("A", "B", "C", "D")
CELL_DESC = {"A": "hvg2000 pbulk", "B": "hvg2000 g.dispz,pbulk", "C": "hvg2000x pbulk", "D": "hvg2000x g.dispz,pbulk"}
FACTORS = {   # effect: (coefficients on A, B, C, D), reference cells for s^2, variance multiplier per block
    "dispersion": ((-0.5, 0.5, -0.5, 0.5), ("A", "C"), 1.0),
    "external": ((-0.5, -0.5, 0.5, 0.5), ("A", "B"), 1.0),
    "interaction": ((1.0, -1.0, -1.0, 1.0), ("A", "B", "C", "D"), 4.0),
}


def cell_labels(ref_tag="r3"):
    return {"A": f"pbulk~{ref_tag}", "B": f"g.dispz,pbulk~{ref_tag}", "C": "pbulk~x", "D": "g.dispz,pbulk~x"}


def mode_label(base, mode):
    return base + (f"#{mode}" if mode else "")


def metric_value(flat, metric):
    v = flat.get(metric)
    if v is None or not np.isfinite(v):
        return None
    return abs(v - 0.8) if metric == "pc_coverage80_rep" else float(v)


# ==========================================================================
# Loading: table[(label, fold, seed)] -> flat metric dict (seed None for ensembles and baselines)
# ==========================================================================
def load_table(results):
    sys.path.insert(0, str(HERE))
    import evaluate_ext as X
    units = X.collect(Path(results))
    table = {}
    for label, us in units.items():
        for fold, seed, flat, _res, _npz in us:
            table[(label, int(fold), seed)] = flat
    return table


# ==========================================================================
# Core statistics (pure functions on the table; planted-truth tested)
# ==========================================================================
def cell_values(table, labels, mode, metric, folds, seeds=SEEDS, ensemble=False):
    """{cell: {(fold, seed): value}} (seed None for ensembles)."""
    out = {}
    for c in CELLS:
        lab = mode_label(labels[c], mode) + (ENS_SUFFIX if ensemble else "")
        d = {}
        for f in folds:
            for s in ([None] if ensemble else seeds):
                flat = table.get((lab, f, s))
                v = metric_value(flat, metric) if flat is not None else None
                if v is not None:
                    d[(f, s)] = v
        out[c] = d
    return out


def seed_variance(vals, cells, folds):
    """Pooled within-fold seed variance (ddof 1) over the given cells; (s2, n_groups) or (None, 0)."""
    var = []
    for c in cells:
        for f in folds:
            v = [x for (ff, s), x in vals[c].items() if ff == f]
            if len(v) >= 2:
                var.append(float(np.var(v, ddof=1)))
    return (float(np.mean(var)), len(var)) if var else (None, 0)


def effect_blocks(vals, coef):
    """{(fold, seed): effect} over complete 2 x 2 blocks."""
    keys = set.intersection(*[set(vals[c]) for c in CELLS]) if all(vals[c] for c in CELLS) else set()
    return {k: float(sum(w * vals[c][k] for w, c in zip(coef, CELLS))) for k in sorted(keys, key=lambda x: (x[0], -1 if x[1] is None else x[1]))}


def summarise_effect(blocks, repeats, s2, var_mult, seeded=True):
    """Fold means, repeat means, pooled (mean of repeat means), seed-noise SE and empirical SE over folds."""
    byf = {}
    for (f, s), v in blocks.items():
        byf.setdefault(f, []).append(v)
    fold_mean = {f: float(np.mean(v)) for f, v in sorted(byf.items())}
    reps = {}
    for r, folds in repeats.items():
        fs = [f for f in folds if f in fold_mean]
        if not fs:
            reps[r] = None
            continue
        m = float(np.mean([fold_mean[f] for f in fs]))
        se = None
        if seeded and s2 is not None:
            se = float(np.sqrt(sum(var_mult * s2 / len(byf[f]) for f in fs)) / len(fs))
        emp = float(np.std([fold_mean[f] for f in fs], ddof=1) / np.sqrt(len(fs))) if len(fs) > 1 else None
        reps[r] = {"mean": m, "folds": fs, "n_blocks": int(sum(len(byf[f]) for f in fs)), "seed_noise_se": se,
                   "empirical_se_over_folds": emp, "n_folds_favourable": int(sum(fold_mean[f] < 0 for f in fs))}
    have = [r for r in repeats if reps.get(r)]
    pooled = None
    if have:
        m = float(np.mean([reps[r]["mean"] for r in have]))
        ses = [reps[r]["seed_noise_se"] for r in have]
        se = float(np.sqrt(sum(x ** 2 for x in ses)) / len(have)) if all(x is not None for x in ses) else None
        fm = [fold_mean[f] for r in have for f in reps[r]["folds"]]
        emp = float(np.std(fm, ddof=1) / np.sqrt(len(fm))) if len(fm) > 1 else None
        pooled = {"mean": m, "repeats": have, "seed_noise_se": se, "empirical_se_over_folds": emp,
                  "z": (m / se) if se else None, "n_folds_favourable": int(sum(v < 0 for v in fm)), "n_folds": len(fm)}
    return {"per_fold": fold_mean, "per_repeat": reps, "pooled": pooled}


def decide(summary, repeats, complete):
    """Pre-declared rule: favourable (negative) in both repeats and pooled < -2 SE."""
    reps = summary["per_repeat"]; p = summary["pooled"]
    if p is None or any(reps.get(r) is None for r in repeats):
        return {"verdict": "NOT EVALUABLE (a repeat has no complete 2 x 2 block)", "confirmed": None}
    both = all(reps[r]["mean"] < 0 for r in repeats)
    se = p["seed_noise_se"]
    beyond = se is not None and p["mean"] < -2 * se
    ok = bool(both and beyond)
    v = "CONFIRMED" if ok else "NOT CONFIRMED"
    if not complete:
        v = "PROVISIONAL " + v + " (incomplete: see missing)"
    return {"verdict": v, "confirmed": ok if complete else None, "favourable_in_both_repeats": bool(both),
            "beyond_2se": bool(beyond), "criterion": "mean < 0 in repeats " + ", ".join(map(str, repeats))
                                                       + " and pooled mean < -2 x seed-noise SE"}


def factor_analysis(table, labels, mode, metric, repeats, seeds=SEEDS):
    folds = [f for fs in repeats.values() for f in fs]
    vals = cell_values(table, labels, mode, metric, folds, seeds)
    s2_all, n_all = seed_variance(vals, CELLS, folds)
    out = {"mode": mode, "metric": metric, "cells": {c: {"label": mode_label(labels[c], mode), "n_fits": len(vals[c]),
                                                          "per_repeat_mean": _cell_repeat_means(vals[c], repeats)}
                                                      for c in CELLS}}
    expected = [(f, s) for f in folds for s in seeds]
    missing = [{"cell": c, "label": mode_label(labels[c], mode), "fold": f, "seed": s}
               for c in CELLS for (f, s) in expected if (f, s) not in vals[c]]
    complete = not missing
    out["missing"] = missing
    out["complete"] = complete
    for name, (coef, ref_cells, mult) in FACTORS.items():
        s2, n = seed_variance(vals, ref_cells, folds)
        src = f"within-fold seed variance of cells {'+'.join(ref_cells)} ({n} fold-cells)"
        if s2 is None:
            s2, src = SEED_SD_PRIOR ** 2, "round-2 prior 0.0156^2"
        blocks = effect_blocks(vals, coef)
        summ = summarise_effect(blocks, repeats, s2, mult)
        summ["s2"] = s2; summ["s2_source"] = src
        sens = summarise_effect(blocks, repeats, s2_all, mult) if s2_all is not None else None
        summ["sensitivity_s2_all_cells"] = {"s2": s2_all, "n_fold_cells": n_all,
                                            "pooled_se": (sens["pooled"] or {}).get("seed_noise_se") if sens else None}
        summ["n_complete_blocks"] = len(blocks)
        if name != "interaction":
            summ["decision"] = decide(summ, list(repeats), complete)
        out[name] = summ
    return out


def _cell_repeat_means(d, repeats):
    out = {}
    for r, folds in repeats.items():
        fm = {}
        for (f, s), v in d.items():
            if f in folds:
                fm.setdefault(f, []).append(v)
        out[r] = float(np.mean([np.mean(v) for v in fm.values()])) if fm else None
    return out


def ensemble_analysis(table, labels, mode, metric, repeats):
    folds = [f for fs in repeats.values() for f in fs]
    vals = cell_values(table, labels, mode, metric, folds, ensemble=True)
    out = {"mode": mode, "metric": metric, "label_suffix": ENS_SUFFIX,
           "cells": {c: {"label": mode_label(labels[c], mode) + ENS_SUFFIX,
                         "per_repeat_mean": _cell_repeat_means(vals[c], repeats)} for c in CELLS},
           "missing": [{"cell": c, "label": mode_label(labels[c], mode) + ENS_SUFFIX, "fold": f}
                       for c in CELLS for f in folds if (f, None) not in vals[c]]}
    for name, (coef, _ref, mult) in FACTORS.items():
        summ = summarise_effect(effect_blocks(vals, coef), repeats, None, mult, seeded=False)
        out[name] = summ
    return out


def baseline_rows(table, repeats, names, metric=METRIC):
    out = {}
    for n in names:
        lab = f"baseline:{n}"
        fm = {f: metric_value(table[(lab, f, None)], metric) for fs in repeats.values() for f in fs if (lab, f, None) in table}
        fm = {f: v for f, v in fm.items() if v is not None}
        reps = {r: (float(np.mean([fm[f] for f in fs if f in fm])) if any(f in fm for f in fs) else None)
                for r, fs in repeats.items()}
        have = [v for v in reps.values() if v is not None]
        out[lab] = {"per_repeat": reps, "pooled": float(np.mean(have)) if have else None,
                    "missing_folds": [f for fs in repeats.values() for f in fs if f not in fm]}
    return out


def distribution_block(table, labels, modes, repeats, ensemble=False):
    """Six distribution metrics per cell and read-out, and paired (by fold) difference to resampling."""
    out = {}
    folds = [f for fs in repeats.values() for f in fs]
    for mode in modes:
        for c in CELLS:
            lab = mode_label(labels[c], mode) + (ENS_SUFFIX if ensemble else "")
            refs = ["baseline:resample"] + (["baseline:resample+ext"] if c in ("C", "D") else [])
            blk = {}
            for m in DIST_METRICS:
                fv = {}
                for f in folds:
                    vs = [metric_value(table[(lab, f, s)], m) for s in ([None] if ensemble else SEEDS) if (lab, f, s) in table]
                    vs = [v for v in vs if v is not None]
                    if vs:
                        fv[f] = float(np.mean(vs))
                rec = {"per_repeat": {r: (float(np.mean([fv[f] for f in fs if f in fv])) if any(f in fv for f in fs) else None)
                                      for r, fs in repeats.items()}}
                for ref in refs:
                    d = {f: fv[f] - metric_value(table[(ref, f, None)], m) for f in fv
                         if (ref, f, None) in table and metric_value(table[(ref, f, None)], m) is not None}
                    rec["minus_" + ref.removeprefix("baseline:")] = {
                        "per_repeat": {r: (float(np.mean([d[f] for f in fs if f in d])) if any(f in d for f in fs) else None)
                                       for r, fs in repeats.items()},
                        "n_folds_better": int(sum(v < 0 for v in d.values())), "n_folds": len(d)}
                blk[m] = rec
            out[f"{CELL_DESC[c]} | {lab}"] = blk
    return out


def analyse(table, ref_tag="r3"):
    labels = cell_labels(ref_tag)
    res = {"protocol": "r4/PROTOCOL_CONFIRM.md", "metric": METRIC, "cells": {c: f"{CELL_DESC[c]}: {labels[c]}" for c in CELLS},
           "seeds": list(SEEDS), "repeats": {str(k): list(v) for k, v in CONFIRM_REPEATS.items()}}
    res["primary"] = factor_analysis(table, labels, PRIMARY_MODE, METRIC, CONFIRM_REPEATS)
    res["secondary_single"] = {m: factor_analysis(table, labels, m, METRIC, CONFIRM_REPEATS) for m in SECONDARY_MODES}
    res["secondary_ensembles"] = {m: ensemble_analysis(table, labels, m, METRIC, CONFIRM_REPEATS) for m in ENSEMBLE_MODES}
    res["distribution_single"] = distribution_block(table, labels, (PRIMARY_MODE,) + SECONDARY_MODES, CONFIRM_REPEATS)
    res["distribution_ensembles"] = distribution_block(table, labels, ENSEMBLE_MODES, CONFIRM_REPEATS, ensemble=True)
    res["references"] = baseline_rows(table, CONFIRM_REPEATS, REF_BASELINES + X_BASELINES)
    res["references_distribution"] = {n: {m: baseline_rows(table, CONFIRM_REPEATS, (n,), m)[f"baseline:{n}"]["pooled"]
                                          for m in DIST_METRICS} for n in ("resample", "resample_blind", "resample+ext")}
    res["exploratory_original_folds"] = {
        "note": "folds 0-2 were used to choose both changes: selection-conditioned, never part of the decision",
        "primary_mode": factor_analysis(table, labels, PRIMARY_MODE, METRIC, ORIGINAL_REPEATS),
        "ensembles": {m: ensemble_analysis(table, labels, m, METRIC, ORIGINAL_REPEATS) for m in ENSEMBLE_MODES},
        "references": baseline_rows(table, ORIGINAL_REPEATS, REF_BASELINES + X_BASELINES)}
    missing_base = {k: v["missing_folds"] for k, v in res["references"].items() if v["missing_folds"]}
    res["missing_summary"] = {
        "primary_fits": res["primary"]["missing"],
        "secondary_fits": {m: res["secondary_single"][m]["missing"] for m in SECONDARY_MODES},
        "ensembles": {m: res["secondary_ensembles"][m]["missing"] for m in ENSEMBLE_MODES},
        "baselines_folds3_8": missing_base,
        "original_folds_primary_fits": res["exploratory_original_folds"]["primary_mode"]["missing"]}
    return res


# ==========================================================================
# Text report
# ==========================================================================
def _f(v, nd=4, plus=False):
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}"


def _effect_lines(name, s, repeats):
    reps = s["per_repeat"]; p = s["pooled"]
    parts = [f"rep{r} {_f((reps.get(r) or {}).get('mean'), plus=True)} (SE {_f((reps.get(r) or {}).get('seed_noise_se'))}, "
             f"{(reps.get(r) or {}).get('n_folds_favourable', 0)}/{len((reps.get(r) or {}).get('folds', []))} folds lower)"
             for r in repeats]
    pl = (f"pooled {_f(p['mean'], plus=True)} +- {_f(p['seed_noise_se'])} (seed-noise SE; z {_f(p['z'], 2)}; "
          f"empirical SE over folds {_f(p['empirical_se_over_folds'])})") if p else "pooled n/a"
    line = f"  {name:12s} " + "; ".join(parts) + "; " + pl
    if "decision" in s:
        line += f"\n  {'':12s} -> {s['decision']['verdict']}"
    return line


def report(res):
    L = ["improve_20261109 extra-fold confirmation (r4/PROTOCOL_CONFIRM.md): pre-declared primary analysis",
         "held-out pb_rmse on SEA-AD hvg2000 inner folds 3-8 (repeat 1 = folds 3-5, repeat 2 = folds 6-8); "
         "negative effect = lower error", ""]
    for c, d in res["cells"].items():
        L.append(f"  cell {c}: {d}")
    pr = res["primary"]
    L += ["", f"PRIMARY: read-out {pr['mode']}, {pr['metric']}; complete 2x2 blocks: "
              f"{pr['dispersion']['n_complete_blocks']} of {len(SEEDS) * 6} (6 folds x {len(SEEDS)} seeds, 4 fits each)"]
    for c in CELLS:
        cm = pr["cells"][c]
        L.append(f"  {c} {cm['label']:32s} n={cm['n_fits']:2d}  " + "  ".join(f"rep{r} {_f(v)}" for r, v in cm["per_repeat_mean"].items()))
    for name in FACTORS:
        L.append(_effect_lines(name, pr[name], CONFIRM_REPEATS))
    L.append(f"  s2 (dispersion) {pr['dispersion']['s2_source']}; (external) {pr['external']['s2_source']}")
    L += ["", "SECONDARY (no decision rule)"]
    for m, a in res["secondary_single"].items():
        L.append(f" single fits, {m}: complete blocks {a['dispersion']['n_complete_blocks']}")
        for name in FACTORS:
            s = dict(a[name]); s.pop("decision", None)
            L.append(_effect_lines(name, s, CONFIRM_REPEATS))
    for m, a in res["secondary_ensembles"].items():
        L.append(f" two-seed ensembles, {m}: cells " + "; ".join(
            f"{c} " + "/".join(_f(v) for v in a["cells"][c]["per_repeat_mean"].values()) for c in CELLS))
        for name in FACTORS:
            s = a[name]; reps = s["per_repeat"]; p = s["pooled"]
            L.append(f"  {name:12s} " + "; ".join(f"rep{r} {_f((reps.get(r) or {}).get('mean'), plus=True)}" for r in CONFIRM_REPEATS)
                     + (f"; pooled {_f(p['mean'], plus=True)} (empirical SE {_f(p['empirical_se_over_folds'])}, "
                        f"{p['n_folds_favourable']}/{p['n_folds']} folds lower)" if p else "; pooled n/a"))
    L += ["", "REFERENCES (pb_rmse, same folds): " + "  ".join(
        f"{k.removeprefix('baseline:')} {_f(v['pooled'])}" for k, v in res["references"].items())]
    L += ["", "DISTRIBUTION (pooled over repeats; minus resample, folds better)"]
    for key, blk in list(res["distribution_single"].items()) + list(res["distribution_ensembles"].items()):
        parts = []
        for m in DIST_METRICS:
            r = blk[m]; d = r["minus_resample"]
            v = [x for x in r["per_repeat"].values() if x is not None]
            dv = [x for x in d["per_repeat"].values() if x is not None]
            parts.append(f"{m.replace('_rmse_gene', '').replace('_rep', '')} {_f(np.mean(v) if v else None, 3)}"
                         f" ({_f(np.mean(dv) if dv else None, 3, True)}, {d['n_folds_better']}/{d['n_folds']})")
        L.append(f"  {key}: " + "; ".join(parts))
    L.append("  resample: " + "; ".join(f"{m} {_f(v, 3)}" for m, v in res["references_distribution"]["resample"].items()))
    ex = res["exploratory_original_folds"]["primary_mode"]
    L += ["", "EXPLORATORY: same 2x2 on the original folds 0-2 (selection-conditioned)"]
    for name in FACTORS:
        s = dict(ex[name]); s.pop("decision", None)
        L.append(_effect_lines(name, s, ORIGINAL_REPEATS))
    ms = res["missing_summary"]
    L += ["", "MISSING"]
    L.append(f"  primary fits ({len(ms['primary_fits'])}): " + ", ".join(
        f"{m['label']} f{m['fold']}s{m['seed']}" for m in ms["primary_fits"][:60]))
    for m, v in ms["secondary_fits"].items():
        L.append(f"  {m} fits: {len(v)}")
    for m, v in ms["ensembles"].items():
        L.append(f"  ensembles {m}: {len(v)} " + ", ".join(f"{x['label']} f{x['fold']}" for x in v[:24]))
    L.append(f"  baselines (folds 3-8): {ms['baselines_folds3_8'] or 'none missing'}")
    L.append("  original folds 0-2 primary fits: " + ", ".join(
        f"{m['label']} f{m['fold']}s{m['seed']}" for m in ms["original_folds_primary_fits"]))
    return "\n".join(L) + "\n"


def _jsonable(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else ("|".join(map(str, k)) if isinstance(k, tuple) else str(k))): _jsonable(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="/ml2/alz/runs/improve_20261109/results_r4")
    ap.add_argument("--ref-tag", default="r3", help="tag of the hvg2000 reference fits (evaluate_ext --ref-root TAG=DIR)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--txt", default=None)
    args = ap.parse_args(argv)
    table = load_table(args.results)
    res = analyse(table, args.ref_tag)
    txt = report(res)
    print(txt, end="")
    jp = Path(args.json) if args.json else Path(args.results) / "confirm_analysis.json"
    tp = Path(args.txt) if args.txt else Path(args.results) / "confirm_analysis.txt"
    jp.write_text(json.dumps(_jsonable(res), indent=1, allow_nan=True) + "\n")
    tp.write_text(txt)
    return res


if __name__ == "__main__":
    main()
