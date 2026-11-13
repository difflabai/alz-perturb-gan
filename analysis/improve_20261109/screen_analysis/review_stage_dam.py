"""Adversarial review: DAM genes per fold (real train / real held-out signs) and model
High-minus-NotAD contrast for the same genes (all 37 targets, mean over folds and seeds).
Also: how often would 5 genes picked by "same positive sign in all 6 train/held-out
contrasts" arise among the panel by chance (fraction of panel genes that are sign-
consistent in all 6), to put the "every fold" statement in context.
Reads caches and generated.npz only (development donors). Own code.
"""
import os, sys, json
sys.dont_write_bytecode = True
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
import scipy.sparse as sp

RUNS = Path("/ml2/alz/runs/improve_20261109")
DEV = set(json.loads(Path("/ml2/alz/analysis/improve_20261109/folds.json").read_text())["development_donors"])
GENES = ["GPNMB", "CD9", "LGALS3", "LPL", "TYROBP", "APOE", "SPP1", "CD74", "TMEM119"]


def lcp(X):
    X = np.asarray(X, float)
    return np.log1p(X / np.maximum(X.sum(-1, keepdims=True), 1) * 1e4)


def hn(P, s):
    return P[s == 3].mean(0) - P[s == 0].mean(0)


out = {}
for panel in ["hvg2000", "wide"]:
    O = out[panel] = {"train": {}, "heldout": {}, "models": {}, "frac_genes_all6_pos": [], "frac_genes_all6_same_sign": None}
    signs = []
    for k in range(3):
        d = RUNS / "caches" / panel / f"fold{k}"
        genes = np.load(d / "space.npz")["genes"].astype(str)
        dn, ho, st, pb = [], [], [], []
        for split in ["train", "heldout"]:
            m = np.load(d / split / "metadata.npz"); assert set(m["donors"].astype(str)) <= DEV
            C = sp.load_npz(d / split / "counts.npz").tocsr(); cd = m["donor"].astype(str)
            for j, x in enumerate(m["donors"].astype(str)):
                dn.append(x); ho.append(split == "heldout"); st.append(int(round(m["raw_c"][j][3] * 3)))
                pb.append(np.asarray(C[np.flatnonzero(cd == x)].sum(0)).ravel())
        dn, ho, st = np.array(dn), np.array(ho), np.array(st); P = lcp(np.array(pb))
        ct, ch = hn(P[~ho], st[~ho]), hn(P[ho], st[ho])
        gi = {g: i for i, g in enumerate(genes)}
        for g in GENES:
            if g in gi:
                O["train"].setdefault(g, []).append(round(float(ct[gi[g]]), 2))
                O["heldout"].setdefault(g, []).append(round(float(ch[gi[g]]), 2))
        signs.append(dict(zip(genes, zip(np.sign(ct), np.sign(ch)))))
        for arm_dir in sorted((RUNS / "fits").iterdir()):
            pdir = arm_dir / panel / f"fold{k}"
            if not pdir.exists():
                continue
            for sd in sorted(pdir.glob("seed*")):
                z = np.load(sd / "generated.npz"); tg = z["targets"].astype(str); assert set(tg) <= DEV
                idx = np.array([np.flatnonzero(tg == x)[0] for x in dn])
                pbc = z["pb_counts"][idx].astype(float)
                pred = np.stack([lcp(pbc[t]).mean(0) for t in range(len(idx))])
                m_ = hn(pred, st)
                for g in GENES:
                    if g in gi:
                        O["models"].setdefault(arm_dir.name, {}).setdefault(g, []).append(float(m_[gi[g]]))
    common = set.intersection(*[set(s) for s in signs])
    allpos = [g for g in common if all(s[g][0] > 0 and s[g][1] > 0 for s in signs)]
    same = [g for g in common if len({x for s in signs for x in s[g]}) == 1]
    O["n_common"] = len(common); O["n_all6_pos"] = len(allpos); O["n_all6_same_sign"] = len(same)
    O["frac_genes_all6_pos"] = round(len(allpos) / len(common), 4)
    O["frac_genes_all6_same_sign"] = round(len(same) / len(common), 4)
    O["models"] = {a: {g: round(float(np.mean(v)), 2) for g, v in d.items()} for a, d in O["models"].items()}
print(json.dumps(out, indent=1))
(RUNS / "screen_analysis" / "review_stage" / "dam.json").write_text(json.dumps(out, indent=1))
