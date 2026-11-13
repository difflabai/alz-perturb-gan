"""Adversarial review of the gene-subset claim (stage_b): recompute, with own code, the
fold-mean train-vs-held-out Spearman inside train-defined subsets (top-10% train variance,
top-10% train expression, top-10% train |t| of High-NotAD) for hn and the ordinal slope,
with a held-out label permutation null (1000 perms). Caches only (development donors).
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
rng = np.random.default_rng(99)


def lcp(X):
    X = np.asarray(X, float)
    return np.log1p(X / np.maximum(X.sum(-1, keepdims=True), 1) * 1e4)


def rho(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def hn(P, s):
    return P[s == 3].mean(0) - P[s == 0].mean(0)


def slope(P, s):
    c = s - s.mean(); return c @ (P - P.mean(0)) / (c @ c)


res = {}
for panel in ["hvg2000", "wide"]:
    obs, nul = {}, {}
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
        Pt, stt = P[~h], s[~h]; G = P.shape[1]; k10 = int(np.ceil(.1 * G))
        a, b = Pt[stt == 3], Pt[stt == 0]
        t = (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b) + 1e-12)
        subs = {"top10_var": np.argsort(-Pt.var(0))[:k10], "top10_expr": np.argsort(-Pt.mean(0))[:k10],
                "top10_abs_welch_t": np.argsort(-np.abs(t))[:k10]}
        perms = [rng.permutation(s[h]) for _ in range(1000)]
        for cn, fn in [("hn", hn), ("slope", slope)]:
            ct = fn(Pt, stt); ch = fn(P[h], s[h]); cp = [fn(P[h], q) for q in perms]
            for sn, ix in subs.items():
                obs.setdefault((sn, cn), []).append(rho(ct[ix], ch[ix]))
                nul.setdefault((sn, cn), []).append([rho(ct[ix], q[ix]) for q in cp])
    R = res[panel] = {}
    for key, o in obs.items():
        n = np.mean(np.array(nul[key]), 0); om = float(np.mean(o))
        R["|".join(key)] = {"per_fold": np.round(o, 3).tolist(), "foldmean": round(om, 3),
                            "p": round(float((1 + (n >= om).sum()) / 1001), 3)}
print(json.dumps(res, indent=1))
(RUNS / "screen_analysis" / "review_stage" / "subsets.json").write_text(json.dumps(res, indent=1))
