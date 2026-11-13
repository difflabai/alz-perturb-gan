#!/usr/bin/env python3
"""Is the seed-ensemble gain real, or just averaging?

The evaluator's seed-ensemble pools the two seeds' generated donors (M = 64)
and its prediction is the average of the two single-seed predictions. For a
squared error this gives the exact identity
   MSE_ens = (MSE_0 + MSE_1)/2 - mean((p0 - p1)^2)/4,
and (p0 - p1) contains (i) Monte-Carlo noise from the finite M = 32 generated
donors per seed and (ii) genuine seed-to-seed model differences.
Per arm / panel / fold / split this script measures:
  * v32: Monte-Carlo variance of a 32-donor mean prediction (var over the 32
    generated donors / 32), i.e. what doubling M within ONE seed would remove;
  * D = mean (p0 - p1)^2 and its MC-only expectation 2 * v32;
  * the ensemble MSE gain D/4 split into an MC part v32/2 and a model-diversity
    part D/4 - v32/2;
  * RMSE of: single seed (M=32); single seed with the MC part removed
    (= M -> 64 within one seed, analytic); ensemble with M matched to a single
    seed (16 donors from seed 0 + 16 from seed 1, direct); ensemble (M=64,
    evaluator definition); and an infinite-seed extrapolation MSE_single - D/2
    (assumes seeds are exchangeable draws; with 2 seeds it is noisy).
  Done for pb_rmse (per donor RMSE over genes, mean over donors) and
  meanlog_rmse_gene (RMSE over genes of the donor-averaged error).
  * PC-space metrics (pc_energy, pc_coverage80, cell_energy) for single seeds,
    the M-matched 16+16 ensemble and the M=64 ensemble, to separate the
    effect of a wider pooled sample (more draws / mixture of two models) from
    a better model.
Writes OUT/arms_ensemble.json and prints markdown tables.
"""
from arms_common import *  # noqa: F401,F403
from arms_common import ARMS, FOLDS, FITS, OUT
from arms_recompute import load_cache, load_gen, align, lcp, bag, energy, cell_ed, det_sub, stable_seed

import json
import numpy as np


def pc_metrics(pcs, r, idx):
    """pcs [T,M,64,50] aligned to r; same definitions as the evaluator."""
    kpc = pcs.shape[2]
    Y = bag(pcs[idx].astype(np.float32))
    y = np.stack([bag(r["pcs"][i][det_sub(len(r["pcs"][i]), kpc, stable_seed("pc-summary", r["donors"][i]))]) for i in idx])
    pe = float(np.mean([energy(Y[j], y[j]) for j in range(len(idx))]))
    lo, hi = np.percentile(Y, [10, 90], axis=1)
    cov = float(((y >= lo) & (y <= hi)).mean())
    ce = []
    for i in idx:
        dn = r["donors"][i]
        gc = pcs[i].reshape(-1, pcs.shape[3]).astype(np.float64)
        gs = gc[det_sub(len(gc), 512, stable_seed("ed-gen", dn))]
        rs = r["pcs"][i][det_sub(len(r["pcs"][i]), 512, stable_seed("ed-real", dn))].astype(np.float32).astype(np.float64)
        ce.append(cell_ed(gs, rs))
    return {"pc_energy": pe, "pc_coverage80": cov, "cell_energy": float(np.mean(ce))}


def unit(arm, panel, fold):
    r = load_cache(panel, fold)
    gs = [align(load_gen(FITS / arm / panel / f"fold{fold}" / f"seed{s}" / "generated.npz"), r) for s in [0, 1]]
    out = {}
    for split in ["heldout", "train"]:
        idx = np.where(r["held"] if split == "heldout" else ~r["held"])[0]
        obs = lcp(r["pb"][idx])
        L = [np.stack([lcp(g["pb_counts"][t].astype(np.float64)) for t in idx]) for g in gs]   # [n,M,G]
        M = L[0].shape[1]; h = M // 2
        p = [x.mean(1) for x in L]
        v32 = np.mean([x.var(1, ddof=1) / M for x in L], 0)                      # [n,G]
        mse = [((q - obs) ** 2).mean(1) for q in p]                               # per donor
        pe = (p[0] + p[1]) / 2
        mse_e = ((pe - obs) ** 2).mean(1)
        D = ((p[0] - p[1]) ** 2).mean(1)
        vm = v32.mean(1)
        pm = (L[0][:, :h].mean(1) + L[1][:, h:].mean(1)) / 2                     # M-matched 16+16 ensemble
        mse_m = ((pm - obs) ** 2).mean(1)
        msing = (mse[0] + mse[1]) / 2
        pb = {"rmse_single": float(np.mean([np.sqrt(m).mean() for m in mse])),
              "rmse_single_M64_analytic": float(np.mean([np.sqrt(np.maximum(m - vm / 2, 0)).mean() for m in mse])),
              "rmse_ens_Mmatched_16plus16": float(np.sqrt(mse_m).mean()),
              "rmse_ens": float(np.sqrt(mse_e).mean()),
              "rmse_inf_seeds_extrap": float(np.sqrt(np.maximum(msing - D / 2, 0)).mean()),
              "D_mean": float(D.mean()), "D_expected_MC_only": float(2 * vm.mean()),
              "D_over_MC": float(D.mean() / (2 * vm.mean())),
              "mse_gain_total": float((msing - mse_e).mean()), "mse_gain_identity_D_over_4": float(D.mean() / 4),
              "mse_gain_MC_part": float(vm.mean() / 2), "mse_gain_model_part": float(D.mean() / 4 - vm.mean() / 2),
              "mse_single": float(msing.mean())}
        # meanlog gene error (donor-averaged), same decomposition
        real_ml = r["ml"][idx]
        Ml = [g["mean_log"][idx].astype(np.float64) for g in gs]                 # [n,M,G]
        e = [(x.mean(1) - real_ml).mean(0) for x in Ml]                           # [G]
        ve = np.mean([(x.var(1, ddof=1) / M).sum(0) / len(idx) ** 2 for x in Ml], 0)   # var of e_g from MC
        ee = (e[0] + e[1]) / 2
        em = ((Ml[0][:, :h].mean(1) + Ml[1][:, h:].mean(1)) / 2 - real_ml).mean(0)
        Dm = float(((e[0] - e[1]) ** 2).mean())
        ms = float(np.mean([(x ** 2).mean() for x in e]))
        ml = {"rmse_single": float(np.mean([np.sqrt((x ** 2).mean()) for x in e])),
              "rmse_single_M64_analytic": float(np.mean([np.sqrt(max((x ** 2).mean() - ve.mean() / 2, 0)) for x in e])),
              "rmse_ens_Mmatched_16plus16": float(np.sqrt((em ** 2).mean())),
              "rmse_ens": float(np.sqrt((ee ** 2).mean())),
              "rmse_inf_seeds_extrap": float(np.sqrt(max(ms - Dm / 2, 0))),
              "D_mean": Dm, "D_expected_MC_only": float(2 * ve.mean()), "D_over_MC": float(Dm / (2 * ve.mean())),
              "mse_gain_MC_part": float(ve.mean() / 2), "mse_gain_model_part": float(Dm / 4 - ve.mean() / 2)}
        # PC-space: single seeds, M-matched ensemble, M=64 ensemble
        pcs0 = gs[0]["pcs"]; pcs1 = gs[1]["pcs"]
        pcm = {"seed0": pc_metrics(pcs0, r, idx), "seed1": pc_metrics(pcs1, r, idx),
               "ens_Mmatched_16plus16": pc_metrics(np.concatenate([pcs0[:, :h], pcs1[:, h:]], 1), r, idx),
               "ens_M64": pc_metrics(np.concatenate([pcs0, pcs1], 1), r, idx)}
        out[split] = {"pb": pb, "meanlog": ml, "pc": pcm}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for panel, arms in ARMS.items():
        for arm in arms:
            for f in FOLDS:
                res[f"{arm}|{panel}|{f}"] = unit(arm, panel, f)
                print("done", arm, panel, f, flush=True)
    (OUT / "arms_ensemble.json").write_text(json.dumps(res, indent=1))
    for split in ["heldout", "train"]:
        for blk in ["pb", "meanlog"]:
            print(f"\n## {split} {blk}: fold-mean (min..max over folds)\n")
            print("| arm | panel | single M32 | single M64 (MC removed) | ens 16+16 | ens M64 | inf-seed extrap | D/D_MC | gain MC part | gain model part |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            for panel, arms in ARMS.items():
                for arm in arms:
                    xs = [res[f"{arm}|{panel}|{f}"][split][blk] for f in FOLDS]
                    def c(k, nd=3):
                        v = np.array([x[k] for x in xs])
                        return f"{v.mean():.{nd}f} ({v.min():.{nd}f}..{v.max():.{nd}f})"
                    print(f"| {arm} | {panel} | {c('rmse_single')} | {c('rmse_single_M64_analytic')} | {c('rmse_ens_Mmatched_16plus16')} | "
                          f"{c('rmse_ens')} | {c('rmse_inf_seeds_extrap')} | {c('D_over_MC', 1)} | {c('mse_gain_MC_part', 5)} | {c('mse_gain_model_part', 5)} |")
        print(f"\n## {split} PC-space: fold-mean of seed-mean single / ens 16+16 / ens M64\n")
        print("| arm | panel | pc_energy | pc_coverage80 | cell_energy |\n|---|---|---|---|---|")
        for panel, arms in ARMS.items():
            for arm in arms:
                xs = [res[f"{arm}|{panel}|{f}"][split]["pc"] for f in FOLDS]
                cells = []
                for k in ["pc_energy", "pc_coverage80", "cell_energy"]:
                    s = np.mean([(x["seed0"][k] + x["seed1"][k]) / 2 for x in xs])
                    m = np.mean([x["ens_Mmatched_16plus16"][k] for x in xs])
                    e = np.mean([x["ens_M64"][k] for x in xs])
                    cells.append(f"{s:.3f} / {m:.3f} / {e:.3f}")
                print(f"| {arm} | {panel} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
