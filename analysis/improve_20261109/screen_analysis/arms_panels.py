#!/usr/bin/env python3
"""Like-for-like panel comparison (hvg2000 vs wide) for base and genemean.

Uses the evaluator's shared-gene subsets: on the genes present in both panels'
same-fold caches, 'shared_*_renorm' recomputes log1p CP10k within the shared
genes (pseudobulk, contrast and pb-distribution metrics comparable across
panels); 'shared_*_withinpanel' keeps each panel's own denominator, so only
its nonzero metrics compare across panels (mean_log is not computed on renorm
subsets and is not comparable on withinpanel ones). PC-space metrics live in
different fold PC spaces per panel and are not compared.
Pairs: wide - hvg2000 within (fold, seed) for each arm, genemean - base within
each panel, and the same for seed-ensembles. Baselines on the same subsets as
references. Writes OUT/arms_panels.json and prints markdown.
"""
from arms_common import *  # noqa: F401,F403
from arms_common import FOLDS, SEEDS, OUT, metric_value, load_fit_json, load_ens_json, load_baseline_json, METRICS, goodness

import json
import numpy as np

SUB = {"hvg2000": ("shared_wide_renorm", "shared_wide_withinpanel"), "wide": ("shared_hvg2000_renorm", "shared_hvg2000_withinpanel")}
MET = ["pb_rmse", "pb_dev_pearson", "hn_spearman_vs_heldout", "pbdist_energy", "nonzero_rmse_gene"]
BASELINES = ["grand_mean", "shrunk_stage_mean", "ridge", "stage_mean_log", "resample"]


def val(res, m, split, panel):
    sub = SUB[panel][1] if m == "nonzero_rmse_gene" else SUB[panel][0]
    return metric_value(res, m, split, sub)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"n_shared_genes": {}}
    lines = []
    for f in FOLDS:
        r = load_fit_json("base", "hvg2000", f, 0)
        out["n_shared_genes"][f] = r["subsets"][SUB["hvg2000"][0]]["n_genes"]
    for split in ["heldout", "train"]:
        tab = {}
        for arm in ["base", "genemean"]:
            for panel in ["hvg2000", "wide"]:
                fits = {(f, s): load_fit_json(arm, panel, f, s) for f in FOLDS for s in SEEDS}
                single = np.array([[[val(fits[(f, s)], m, split, panel) for s in SEEDS] for f in FOLDS] for m in MET], dtype=float)
                ens = np.array([[val(load_ens_json(arm, panel, f), m, split, panel) for f in FOLDS] for m in MET], dtype=float)
                tab[(arm, panel)] = (single, ens)          # single [metric, fold, seed], ens [metric, fold]
        for b in BASELINES:
            for panel in ["hvg2000", "wide"]:
                tab[(f"baseline:{b}", panel)] = (None, np.array([[val(load_baseline_json(b, panel, f), m, split, panel) for f in FOLDS] for m in MET], dtype=float))
        lines += [f"\n## {split}: shared genes (n per fold {out['n_shared_genes']}); renormalised within shared genes (nonzero: within-panel)\n",
                  "| row | " + " | ".join(MET) + " |", "|---|" + "---|" * len(MET)]
        for (row, panel), (single, ens) in tab.items():
            if single is not None:
                cells = [f"{single[i].mean():.3f} [{', '.join(f'{x:.3f}' for x in single[i].mean(1))}] sd_seed {np.sqrt(np.mean((single[i][:, 0]-single[i][:, 1])**2)/2):.3f}" for i in range(len(MET))]
                lines.append(f"| {row} {panel} | " + " | ".join(cells) + " |")
                cells = [f"{ens[i].mean():.3f} [{', '.join(f'{x:.3f}' for x in ens[i])}]" for i in range(len(MET))]
                lines.append(f"| {row} {panel} [ens] | " + " | ".join(cells) + " |")
            else:
                cells = [f"{ens[i].mean():.3f}" for i in range(len(MET))]
                lines.append(f"| {row} {panel} | " + " | ".join(cells) + " |")
        lines += ["", "Paired differences (raw, mean [per fold]; crossed wins of 12 on goodness scale):", "",
                  "| comparison | " + " | ".join(MET) + " |", "|---|" + "---|" * len(MET)]
        comps = [("wide - hvg2000, base", ("base", "wide"), ("base", "hvg2000")),
                 ("wide - hvg2000, genemean", ("genemean", "wide"), ("genemean", "hvg2000")),
                 ("genemean - base, hvg2000", ("genemean", "hvg2000"), ("base", "hvg2000")),
                 ("genemean - base, wide", ("genemean", "wide"), ("base", "wide"))]
        for name, A, B in comps:
            sa, ea = tab[A]; sb, eb = tab[B]
            cells, rec = [], {}
            for i, m in enumerate(MET):
                kind = METRICS[m][1]
                d = sa[i] - sb[i]
                ga = np.vectorize(lambda x: goodness(kind, x))(sa[i]); gb = np.vectorize(lambda x: goodness(kind, x))(sb[i])
                wins = int(sum((ga[f, i1] - gb[f, j1]) > 0 for f in range(3) for i1 in SEEDS for j1 in SEEDS))
                ed = ea[i] - eb[i]
                cells.append(f"{d.mean():+.3f} [{', '.join(f'{x:+.3f}' for x in d.mean(1))}] {wins}/12; ens {ed.mean():+.3f}")
                rec[m] = {"diff_mean": float(d.mean()), "diff_fold": d.mean(1).tolist(), "crossed_wins_of12": wins,
                          "ens_diff_fold": ed.tolist()}
            out[f"{split}|{name}"] = rec
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        for k, (s, e) in tab.items():
            out[f"{split}|{k[0]}|{k[1]}"] = {"single": None if s is None else s.tolist(), "ens": e.tolist(), "metrics": MET}
    (OUT / "arms_panels.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
