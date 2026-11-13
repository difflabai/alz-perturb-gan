#!/usr/bin/env python3
"""Arm effects versus base, from the evaluator's per-fit / ensemble JSONs.

For every arm != base, panel, split (heldout / train) and metric:
  * fold-balanced means of base and arm (seeds averaged within fold);
  * seed-paired differences d[f,s] = arm[f,s] - base[f,s] (6 per arm), their
    fold means (3), and the number of the 12 CROSSED within-fold comparisons
    (every arm seed vs every base seed of the same fold) in which the arm is
    better on the goodness scale (seed pairing is arbitrary, so the crossed
    count is the assumption-free statement "every arm fit beats every base fit
    of its fold");
  * seed SD (pooled within-fold SD of single fits, per arm and base) and the
    fold SD of the fold-mean difference, so effects can be read against noise;
  * seed-ensemble differences per fold (ensemble arm - ensemble base);
  * variance components of d (seed, fold) and the seeds per arm needed for the
    mean effect to be 2.8 SE away from 0 (80% power at two-sided 5%, used only
    as a design yardstick, not as a test) with (a) folds treated as fixed and
    (b) folds treated as random (fold-by-arm interaction does not shrink with
    seeds).
Writes OUT/arms_effects.json and OUT/arms_effects.md (scratch).
"""
from arms_common import *  # noqa: F401,F403
from arms_common import ARMS, FOLDS, SEEDS, METRICS, OUT, goodness, metric_value, load_fit_json, load_ens_json

import json
import numpy as np

Z = 2.8


def collect():
    vals = {}   # (arm, panel, split, metric) -> array [3 folds, 2 seeds]
    ens = {}    # (arm, panel, split, metric) -> array [3 folds]
    for panel, arms in ARMS.items():
        for arm in arms:
            fits = {(f, s): load_fit_json(arm, panel, f, s) for f in FOLDS for s in SEEDS}
            enss = {f: load_ens_json(arm, panel, f) for f in FOLDS}
            for split in ["heldout", "train"]:
                for m in METRICS:
                    vals[(arm, panel, split, m)] = np.array(
                        [[np.nan if (v := metric_value(fits[(f, s)], m, split)) is None else v for s in SEEDS] for f in FOLDS])
                    ens[(arm, panel, split, m)] = np.array(
                        [np.nan if (v := metric_value(enss[f], m, split)) is None else v for f in FOLDS])
    return vals, ens


def seeds_needed(mu, var_seed_d, var_fold_d, nf=3):
    """Seeds per arm (and per base) so that |mu| >= Z * SE.
    var_seed_d: variance of a single-seed paired difference within fold (arm + base seed noise).
    fixed folds: SE^2 = var_seed_d / (nf * n);  random folds: + var_fold_d / nf."""
    if not np.isfinite(mu) or mu == 0:
        return None, None
    tgt = (mu / Z) ** 2
    fixed = var_seed_d / (nf * tgt)
    rem = tgt - var_fold_d / nf
    rnd = var_seed_d / (nf * rem) if rem > 0 else float("inf")
    return float(max(1.0, np.ceil(fixed))), (float(max(1.0, np.ceil(rnd))) if np.isfinite(rnd) else None)


def analyse(vals, ens):
    out = {}
    for panel, arms in ARMS.items():
        for arm in arms:
            if arm == "base":
                continue
            for split in ["heldout", "train"]:
                for m, (_, kind) in METRICS.items():
                    a = vals[(arm, panel, split, m)]; b = vals[(("base"), panel, split, m)]
                    if np.isnan(a).all() or np.isnan(b).all():
                        continue
                    ga = np.vectorize(lambda x: goodness(kind, x))(a); gb = np.vectorize(lambda x: goodness(kind, x))(b)
                    d = a - b                                   # raw seed-paired differences [3,2]
                    gd = ga - gb                                # goodness differences
                    crossed = np.array([[ga[f, i] - gb[f, j] for i in SEEDS for j in SEEDS] for f in FOLDS])  # [3,4]
                    n_better_crossed = int((crossed > 0).sum())
                    n_better_paired = int((gd > 0).sum())
                    fold_d = d.mean(1)
                    fold_gd = gd.mean(1)
                    sd_seed_arm = float(np.sqrt(np.mean((a[:, 0] - a[:, 1]) ** 2) / 2))
                    sd_seed_base = float(np.sqrt(np.mean((b[:, 0] - b[:, 1]) ** 2) / 2))
                    # variance components of the goodness difference
                    var_seed_d = float(np.mean((gd[:, 0] - gd[:, 1]) ** 2) / 2)     # single paired diff, within fold
                    var_foldmean = float(np.var(fold_gd, ddof=1))
                    var_fold_d = max(0.0, var_foldmean - var_seed_d / 2)            # fold-by-arm interaction
                    mu = float(gd.mean())
                    se_now = float(np.sqrt(var_foldmean / 3))
                    n_fixed, n_random = seeds_needed(mu, var_seed_d, var_fold_d)
                    ea = ens[(arm, panel, split, m)]; eb = ens[("base", panel, split, m)]
                    gea = np.array([goodness(kind, x) for x in ea]); geb = np.array([goodness(kind, x) for x in eb])
                    rec = {
                        "base_mean": float(b.mean()), "arm_mean": float(a.mean()),
                        "base_values": b.tolist(), "arm_values": a.tolist(),
                        "raw_diff_mean": float(d.mean()), "raw_diff_fold": fold_d.tolist(), "raw_diff_paired": d.tolist(),
                        "good_diff_mean": mu, "good_diff_fold": fold_gd.tolist(),
                        "n_better_paired_of6": n_better_paired, "n_better_crossed_of12": n_better_crossed,
                        "sd_seed_arm": sd_seed_arm, "sd_seed_base": sd_seed_base,
                        "sd_seed_diff": float(np.sqrt(var_seed_d)), "sd_fold_of_foldmean_diff": float(np.sqrt(var_foldmean)),
                        "sd_fold_interaction": float(np.sqrt(var_fold_d)),
                        "effect_over_sd_seed_diff": (mu / np.sqrt(var_seed_d)) if var_seed_d > 0 else None,
                        "se_over_folds": se_now,
                        "seeds_needed_fixed_folds": n_fixed, "seeds_needed_random_folds": n_random,
                        "ens_raw_diff_fold": (ea - eb).tolist(), "ens_good_diff_fold": (gea - geb).tolist(),
                        "ens_n_better_of3": int(((gea - geb) > 0).sum()),
                        "base_ens_mean": float(np.mean(eb)), "arm_ens_mean": float(np.mean(ea)),
                    }
                    if n_better_crossed == 12:
                        v = "BETTER (12/12)"
                    elif n_better_crossed == 0:
                        v = "WORSE (0/12)"
                    elif n_better_crossed >= 10:
                        v = "lean better"
                    elif n_better_crossed <= 2:
                        v = "lean worse"
                    else:
                        v = "indistinguishable"
                    rec["verdict"] = v
                    out[f"{arm}|{panel}|{split}|{m}"] = rec
    return out


def to_md(res):
    lines = ["# arms_effects (generated by arms_effects.py)", "",
             "Raw differences are arm - base (rmse/energy: negative = arm better). crossed = number of the 12 within-fold "
             "(arm seed, base seed) comparisons the arm wins on the goodness scale (coverage: closer to 0.8; spread: closer to 1). "
             "sd_seed = pooled within-fold SD of single fits (base / arm). n_seeds = seeds per arm for |effect| = 2.8 SE "
             "(fixed folds / random folds; '-' = not reachable with 3 folds).", ""]
    for key0 in sorted(set("|".join(k.split("|")[:3]) for k in res), key=lambda s: (s.split("|")[1], s.split("|")[2], s.split("|")[0])):
        pass
    for panel, arms in ARMS.items():
        for split in ["heldout", "train"]:
            lines += [f"## {panel} / {split}", "",
                      "| arm | metric | base | arm | diff | fold diffs | paired diffs min..max | crossed | sd_seed base/arm | eff/sd_seed_diff | ens base->arm | ens fold diffs | n_seeds fixed/random | verdict |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for arm in arms:
                if arm == "base":
                    continue
                for m in METRICS:
                    r = res.get(f"{arm}|{panel}|{split}|{m}")
                    if r is None:
                        continue
                    pd = np.array(r["raw_diff_paired"]).ravel()
                    e = r["effect_over_sd_seed_diff"]
                    nf, nr = r["seeds_needed_fixed_folds"], r["seeds_needed_random_folds"]
                    lines.append(
                        f"| {arm} | {m} | {r['base_mean']:.3f} | {r['arm_mean']:.3f} | {r['raw_diff_mean']:+.3f} | "
                        + "/".join(f"{x:+.3f}" for x in r["raw_diff_fold"]) + f" | {pd.min():+.3f}..{pd.max():+.3f} | "
                        f"{r['n_better_crossed_of12']}/12 | {r['sd_seed_base']:.3f}/{r['sd_seed_arm']:.3f} | "
                        f"{'-' if e is None else f'{e:+.1f}'} | {r['base_ens_mean']:.3f}->{r['arm_ens_mean']:.3f} | "
                        + "/".join(f"{x:+.3f}" for x in r["ens_raw_diff_fold"]) + f" | "
                        f"{'-' if nf is None else int(nf)}/{'-' if nr is None else int(nr)} | {r['verdict']} |")
            lines.append("")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vals, ens = collect()
    res = analyse(vals, ens)
    (OUT / "arms_effects.json").write_text(json.dumps(res, indent=1))
    md = to_md(res)
    (OUT / "arms_effects.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
