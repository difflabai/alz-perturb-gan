"""Model-free sanity check on the 120 TRAINING perturbations only (held-out 30 never touched).

delta_i = pseudobulk(pert i, all cells) - non-targeting mean, log1p(normalize_total 1e4), all 18,080 genes
minus the 150 knockdown-target columns. Delta similarity: cosine. Feature similarity: cosine.
(1) Spearman between pairwise feature and delta similarity over usable training pairs.
(2) Inner 5-fold CV: each query's feature-nearest neighbour is searched among the other folds' genes only;
    hit@k if that neighbour is in the query's top-k by delta similarity within the same reference set.
Shuffled control: permute feature rows among usable training genes (1000 permutations, seed 0).
"""
import json, numpy as np
from scipy.stats import spearmanr
OUT = "/workspace/altgan/features/out"
dm = json.load(open("deltas/deltas_meta.json")); D = np.load("deltas/deltas.npy")
man = json.load(open(f"{OUT}/manifest.json")); order = man["gene_order"]["symbols"]
eval_set = set(dm["eval_names"]); pos = {n: i for i, n in enumerate(dm["names"])}
cols = np.array([g not in set(dm["names"]) for g in dm["genes"]])
D = D[[pos[s] for s in order]][:, cols]
train = np.array([s not in eval_set for s in order]); assert train.sum() == 120
def cos(M):
    M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12); return M @ M.T
Sd = cos(D)
rng0 = np.random.default_rng(0); folds = rng0.permutation(120) % 5
def metrics(F, idx, folds_):
    Sf = cos(F[idx]); sd = Sd[np.ix_(idx, idx)]; n = len(idx)
    iu = np.triu_indices(n, 1); rho = spearmanr(Sf[iu], sd[iu]).statistic
    hits = {1: [], 5: [], 10: []}
    for q in range(n):
        ref = np.where(folds_ != folds_[q])[0]
        nn = ref[np.argmax(Sf[q, ref])]
        rank = np.argsort(-sd[q, ref]); top = ref[rank]
        for k in hits: hits[k].append(nn in top[:k])
    return rho, {k: float(np.mean(v)) for k, v in hits.items()}, len(ref)
res = {}
for name, src in man["sources"].items():
    F = np.load(f"{OUT}/{src['file']}").astype(np.float64)
    bad = set(src["all_zero_rows"]) | {g for k, v in src["duplicate_rows"].items() for g in [k] + v}
    idx = np.array([i for i in range(150) if train[i] and order[i] not in bad])
    fidx = folds[np.searchsorted(np.where(train)[0], idx)]
    rho, hit, nref = metrics(F, idx, fidx)
    rng = np.random.default_rng(0); sh_rho = []; sh_hit = {1: [], 5: [], 10: []}
    for _ in range(1000):
        P = F.copy(); P[idx] = F[rng.permutation(idx)]
        r, h, _ = metrics(P, idx, fidx); sh_rho.append(r)
        for k in h: sh_hit[k].append(h[k])
    sh_rho = np.array(sh_rho)
    res[name] = {"n_train_usable": int(len(idx)), "spearman": float(rho),
                 "spearman_shuffle_mean": float(sh_rho.mean()), "spearman_shuffle_p": float((1 + (sh_rho >= rho).sum()) / 1001),
                 **{f"hit@{k}": hit[k] for k in hit},
                 **{f"hit@{k}_shuffle_mean": float(np.mean(sh_hit[k])) for k in hit},
                 **{f"hit@{k}_shuffle_p": float((1 + (np.array(sh_hit[k]) >= hit[k]).sum()) / 1001) for k in hit}}
    print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res[name].items()}), flush=True)
json.dump({"protocol": __doc__, "results": res, "heldout_reported": False}, open(f"{OUT}/sanity_train120.json", "w"), indent=1)
