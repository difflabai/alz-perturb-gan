"""Adversarial review: cross-fitted donor score (claim 3). For each held-out donor,
Pearson(profile - train grand mean, train High-NotAD contrast); pooled over the 3 folds'
held-out sets (= 37 donors). Spearman(score, stage), AUC High vs Not AD; within-fold
label permutation null (2000). Also the same with the post hoc 5-gene DAM score
replaced by a train-only score: cross-fitted mean z of the top-50 train |t| genes.
Own code; caches only (development donors).
"""
import os, sys, json
sys.dont_write_bytecode = True
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

RUNS = Path("/ml2/alz/runs/improve_20261109")
DEV = set(json.loads(Path("/ml2/alz/analysis/improve_20261109/folds.json").read_text())["development_donors"])
rng = np.random.default_rng(5)


def lcp(X):
    X = np.asarray(X, float)
    return np.log1p(X / np.maximum(X.sum(-1, keepdims=True), 1) * 1e4)


def rho(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def auc(x, s):
    a, b = x[s == 3], x[s == 0]
    return float(((a[:, None] > b[None]).sum() + .5 * (a[:, None] == b[None]).sum()) / (len(a) * len(b)))


res = {}
for panel in ["hvg2000", "wide"]:
    sc, sc2, ss, ff = [], [], [], []
    for k in range(3):
        d = RUNS / "caches" / panel / f"fold{k}"
        ho, st, pb = [], [], []
        for split in ["train", "heldout"]:
            m = np.load(d / split / "metadata.npz"); assert set(m["donors"].astype(str)) <= DEV
            C = sp.load_npz(d / split / "counts.npz").tocsr(); cd = m["donor"].astype(str)
            for j, x in enumerate(m["donors"].astype(str)):
                ho.append(split == "heldout"); st.append(int(round(m["raw_c"][j][3] * 3)))
                pb.append(np.asarray(C[np.flatnonzero(cd == x)].sum(0)).ravel())
        h, s, P = np.array(ho), np.array(st), lcp(np.array(pb))
        Pt, stt = P[~h], s[~h]
        ct = Pt[stt == 3].mean(0) - Pt[stt == 0].mean(0); g = Pt.mean(0); sd = Pt.std(0) + 1e-9
        a, b = Pt[stt == 3], Pt[stt == 0]
        t = ct / np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b) + 1e-12)
        top = np.argsort(-np.abs(t))[:50]
        for i in np.flatnonzero(h):
            sc.append(np.corrcoef(P[i] - g, ct)[0, 1])
            sc2.append(float((np.sign(t[top]) * (P[i, top] - g[top]) / sd[top]).mean()))
            ss.append(s[i]); ff.append(k)
    sc, sc2, ss, ff = map(np.array, (sc, sc2, ss, ff))
    R = res[panel] = {}
    for nm, x in [("grandmean_dev_corr", sc), ("top50_t_signed_z", sc2)]:
        o_r, o_a = rho(x, ss), auc(x, ss)
        nr, na = [], []
        for _ in range(2000):
            p = ss.copy()
            for k in range(3):
                p[ff == k] = rng.permutation(ss[ff == k])
            nr.append(rho(x, p)); na.append(auc(x, p))
        R[nm] = {"spearman_stage": round(o_r, 3), "p": round(float((1 + (np.array(nr) >= o_r).sum()) / 2001), 3),
                 "auc_high_notad": round(o_a, 3), "p_auc": round(float((1 + (np.array(na) >= o_a).sum()) / 2001), 3)}
print(json.dumps(res, indent=1))
(RUNS / "screen_analysis" / "review_stage" / "donorscore.json").write_text(json.dumps(res, indent=1))
