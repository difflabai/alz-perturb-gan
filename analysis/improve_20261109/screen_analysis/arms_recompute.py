#!/usr/bin/env python3
"""Independent recomputation of the primary metrics for a sample of fits.

Does NOT import evaluate.py / metrics.py: re-implements from PROTOCOL
Interface 1/2 and the documented definitions (pb_rmse, meanlog_rmse_gene,
nonzero_rmse_gene, hn_spearman_vs_heldout, pc_energy, pc_coverage80,
cell_energy) and compares with results/fits/*.json (and one seed-ensemble).
Only the deterministic subsample rule (sha256-derived seed + numpy choice) is
copied, because it defines which cells are compared.

Also measures the Monte Carlo noise of cell_energy / pc metrics that comes
from the evaluator's fixed cell subsample, by re-drawing it with 5 other
seeds (reported as SD across redraws).
Writes OUT/arms_recompute.json.
"""
from arms_common import *  # noqa: F401,F403
from arms_common import CACHES, FITS, OUT, STUDY, load_fit_json, load_ens_json, metric_value

import hashlib
import json
import sys

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

DEV = set(json.loads((STUDY / "folds.json").read_text())["development_donors"])


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:4], "little")


def det_sub(n, k, seed):
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def lcp(X):
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(X * 1e4 / np.maximum(X.sum(-1, keepdims=True), 1))


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


_cache = {}


def load_cache(panel, fold):
    key = (panel, fold)
    if key in _cache:
        return _cache[key]
    d = CACHES / panel / f"fold{fold}"
    sp_ = dict(np.load(d / "space.npz", allow_pickle=False))
    donors, held, stage, pb, ml, nz, pcs = [], [], [], [], [], [], []
    for split in ["train", "heldout"]:
        meta = dict(np.load(d / split / "metadata.npz", allow_pickle=False))
        dl = np.asarray(meta["donors"]).astype(str)
        assert set(dl) <= DEV, "non-development donor in cache"
        C = sp.load_npz(d / split / "counts.npz").tocsr()
        cd = np.asarray(meta["donor"]).astype(str)
        for j, dn in enumerate(dl):
            X = C[np.where(cd == dn)[0]].toarray().astype(np.float64)
            L = lcp(X)
            donors.append(dn); held.append(split == "heldout"); stage.append(int(round(meta["raw_c"][j][3] * 3)))
            pb.append(X.sum(0)); ml.append(L.mean(0)); nz.append((X > 0).mean(0))
            pcs.append(((L - sp_["mean"]) / sp_["sd"]) @ sp_["W"] / sp_["pc_sd"])
    r = dict(donors=np.array(donors), held=np.array(held), stage=np.array(stage), pb=np.stack(pb),
             ml=np.stack(ml), nz=np.stack(nz), pcs=pcs, genes=np.asarray(sp_["genes"]).astype(str))
    _cache[key] = r
    return r


def load_gen(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def align(g, r):
    tg = list(np.asarray(g["targets"]).astype(str))
    order = np.array([tg.index(d) for d in r["donors"]])
    assert np.array_equal(np.asarray(g["genes"]).astype(str), r["genes"])
    return {k: g[k][order] for k in ["pb_counts", "mean_log", "nonzero", "pcs", "stage", "is_heldout"]}


def energy(Y, y):
    return np.linalg.norm(Y - y, axis=1).mean() - 0.5 * cdist(Y, Y).sum() / (len(Y) * (len(Y) - 1))


def cell_ed(A, B):
    return 2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean()


def bag(p):
    p = np.asarray(p, dtype=np.float64)
    return np.concatenate([p.mean(-2), p.std(-2)], -1)


def score(g, r, split, redraws=0):
    mask = r["held"] if split == "heldout" else ~r["held"]
    idx = np.where(mask)[0]
    assert np.array_equal(g["stage"].astype(int), r["stage"])
    pred = np.stack([lcp(g["pb_counts"][t].astype(np.float64)).mean(0) for t in idx])
    obs = lcp(r["pb"][idx])
    out = {"pb_rmse": float(np.mean(np.sqrt(((pred - obs) ** 2).mean(1))))}
    e = (g["mean_log"][idx].astype(np.float64).mean(1) - r["ml"][idx]).mean(0)
    out["meanlog_rmse_gene"] = float(np.sqrt((e ** 2).mean()))
    e = (g["nonzero"][idx].astype(np.float64).mean(1) - r["nz"][idx]).mean(0)
    out["nonzero_rmse_gene"] = float(np.sqrt((e ** 2).mean()))
    st = r["stage"][idx]
    hidx = np.where(r["held"])[0]; hst = r["stage"][hidx]
    real_c = lcp(r["pb"][hidx])[hst == 3].mean(0) - lcp(r["pb"][hidx])[hst == 0].mean(0)
    if (st == 0).any() and (st == 3).any():
        mod_c = pred[st == 3].mean(0) - pred[st == 0].mean(0)
        out["hn_spearman_vs_heldout"] = spearman(mod_c, real_c)
    kpc = g["pcs"].shape[2]
    Y = bag(g["pcs"][idx].astype(np.float32))
    y = np.stack([bag(r["pcs"][i][det_sub(len(r["pcs"][i]), kpc, stable_seed("pc-summary", r["donors"][i]))]) for i in idx])
    out["pc_energy"] = float(np.mean([energy(Y[j], y[j]) for j in range(len(idx))]))
    lo, hi = np.percentile(Y, [10, 90], axis=1)
    out["pc_coverage80"] = float(((y >= lo) & (y <= hi)).mean())

    def ced(tag_g, tag_r):
        v = []
        for i in idx:
            dn = r["donors"][i]
            gc = g["pcs"][i].reshape(-1, g["pcs"].shape[3]).astype(np.float32)
            gs = gc[det_sub(len(gc), 512, stable_seed(tag_g, dn))]
            rs = r["pcs"][i].astype(np.float32)[det_sub(len(r["pcs"][i]), 512, stable_seed(tag_r, dn))]
            v.append(cell_ed(gs.astype(np.float64), rs.astype(np.float64)))
        return float(np.mean(v))
    out["cell_energy"] = ced("ed-gen", "ed-real")
    if redraws:
        out["cell_energy_redraws"] = [ced(f"arms-redraw-gen-{k}", f"arms-redraw-real-{k}") for k in range(redraws)]
    return out


SAMPLE = [("base", "hvg2000", 0, 0), ("genemean", "hvg2000", 1, 1), ("offset", "hvg2000", 2, 1),
          ("donorpc", "hvg2000", 1, 0), ("covars", "hvg2000", 2, 0), ("covars,genemean,offset", "hvg2000", 0, 1),
          ("base", "wide", 1, 1), ("genemean", "wide", 2, 0)]
ENS_SAMPLE = [("genemean", "hvg2000", 0)]
KEYS = ["pb_rmse", "meanlog_rmse_gene", "nonzero_rmse_gene", "hn_spearman_vs_heldout", "pc_energy", "pc_coverage80", "cell_energy"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, maxdiff = [], 0.0
    jobs = [("fit",) + s for s in SAMPLE] + [("ens",) + s + (None,) for s in ENS_SAMPLE]
    for kind, arm, panel, fold, seed in jobs:
        r = load_cache(panel, fold)
        if kind == "fit":
            g = align(load_gen(FITS / arm / panel / f"fold{fold}" / f"seed{seed}" / "generated.npz"), r)
            ref = load_fit_json(arm, panel, fold, seed)
        else:
            gs = [align(load_gen(FITS / arm / panel / f"fold{fold}" / f"seed{s}" / "generated.npz"), r) for s in [0, 1]]
            g = dict(gs[0])
            for k in ["pb_counts", "mean_log", "nonzero", "pcs"]:
                g[k] = np.concatenate([x[k] for x in gs], 1)
            ref = load_ens_json(arm, panel, fold)
        for split in ["heldout", "train"]:
            mine = score(g, r, split, redraws=5 if (kind == "fit" and split == "heldout") else 0)
            for k in KEYS:
                if k not in mine:
                    continue
                theirs = metric_value(ref, k, split)
                d = None if theirs is None else abs(mine[k] - theirs)
                maxdiff = max(maxdiff, d or 0)
                rows.append({"kind": kind, "arm": arm, "panel": panel, "fold": fold, "seed": seed, "split": split,
                             "metric": k, "recomputed": mine[k], "evaluator": theirs, "abs_diff": d})
            if "cell_energy_redraws" in mine:
                rd = np.array(mine["cell_energy_redraws"])
                rows.append({"kind": kind, "arm": arm, "panel": panel, "fold": fold, "seed": seed, "split": split,
                             "metric": "cell_energy_subsample_sd", "recomputed": float(rd.std(ddof=1)),
                             "evaluator": None, "abs_diff": None, "redraws": rd.tolist()})
            print(kind, arm, panel, fold, seed, split, {k: round(v, 4) if isinstance(v, float) else v for k, v in mine.items()
                                                        if k != "cell_energy_redraws"}, flush=True)
    (OUT / "arms_recompute.json").write_text(json.dumps({"rows": rows, "max_abs_diff": maxdiff}, indent=1))
    print("\n| fit | split | metric | recomputed | evaluator | abs diff |\n|---|---|---|---|---|---|")
    for x in rows:
        print(f"| {x['kind']} {x['arm']}/{x['panel']}/f{x['fold']}/s{x['seed']} | {x['split']} | {x['metric']} | "
              f"{x['recomputed']:.5f} | {'-' if x['evaluator'] is None else format(x['evaluator'], '.5f')} | "
              f"{'-' if x['abs_diff'] is None else format(x['abs_diff'], '.1e')} |")
    print("MAX ABS DIFF", maxdiff)


if __name__ == "__main__":
    sys.exit(main())
