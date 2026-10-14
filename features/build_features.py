"""Build frozen per-gene feature matrices for the 150 Arc knockdown genes (CPU only).

Row order: genes sorted by the dataset's Ensembl ID (var["gene_id"]); shared by all sources.
Missing genes get an all-zero row and are listed in the manifest; no imputation.
"""
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

COV = Path("/workspace/altgan/cov"); OUT = Path("/workspace/altgan/features/out"); OUT.mkdir(parents=True, exist_ok=True)
SEED = 0; K = 64
rows = sorted((l.split("\t") for l in (COV / "genes150_ensembl.tsv").read_text().split("\n") if l), key=lambda r: r[1])
symbols = [r[0] for r in rows]; ensembl = [r[1] for r in rows]
sym_for_lookup = ["TAFAZZIN" if (s, e) == ("TAZ", "ENSG00000102125") else s for s, e in rows]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def finish(name, X, missing, info):
    X = np.ascontiguousarray(X, dtype=np.float32)
    path = OUT / f"{name}.npy"; np.save(path, X)
    zero = [symbols[i] for i in range(len(X)) if not np.any(X[i])]
    dup = {}
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if np.any(X[i]) and np.array_equal(X[i], X[j]): dup.setdefault(symbols[i], []).append(symbols[j])
    usable = len(X) - len(set(zero) | {g for k, v in dup.items() for g in [k] + v})
    info.update({"file": path.name, "sha256": sha(path), "dtype": str(X.dtype), "shape": list(X.shape),
                 "missing": missing, "all_zero_rows": zero, "duplicate_rows": dup, "usable_rows": usable})
    print(name, X.shape, "usable", usable, "missing", missing, "dup", dup, flush=True)
    return info

manifest = {"gene_order": {"key": "dataset var['gene_id'] (Ensembl)", "ensembl": ensembl, "symbols": symbols,
            "lookup_symbols": sym_for_lookup, "rule": "TAZ (ENSG00000102125) is looked up as TAFAZZIN; never via the WWTR1 alias"},
            "sources": {}}

# --- Gene2vec ---
g2v = {}
for l in open("/workspace/altgan/g2v.txt"):
    p = l.split()
    if len(p) > 2: g2v[p[0]] = np.array(p[1:], dtype=np.float32)
X = np.zeros((150, 200), np.float32); miss = []
for i, s in enumerate(symbols):  # Gene2vec (2018) predates the TAFAZZIN rename; its TAZ key = old tafazzin symbol
    v = g2v.get(s if s != "TAZ" else "TAZ")
    if v is None: miss.append(s)
    else: X[i] = v
manifest["sources"]["gene2vec"] = finish("gene2vec", X, miss, {
    "url": "https://github.com/jingcheng-du/Gene2vec/blob/master/pre_trained_emb/gene2vec_dim_200_iter_9.txt",
    "input_sha256": sha("/workspace/altgan/g2v.txt"), "version": "gene2vec_dim_200_iter_9 (2018 release)",
    "license_model_or_data": "MIT (repository)", "license_outputs": "MIT (vectors are distributed data files in the repo)",
    "dim": 200, "processing": "raw vectors, no normalization", "symbol_note": "TAZ key used as-is (pre-rename symbol for tafazzin; WWTR1 not aliased)"})

# --- GO (TF-IDF + truncated SVD on the 150-gene universe) ---
mg = json.load(open(COV / "mygene.json")); taf = json.load(open(COV / "tafazzin_go.json"))["hits"][0]
best = {}
for x in mg:
    if x.get("notfound"): continue
    if x["query"] not in best or (x.get("go") and not best[x["query"]].get("go")): best[x["query"]] = x
best["TAZ"] = taf
terms = []
for s in symbols:
    t = set()
    for aspect in ("BP", "MF", "CC"):
        v = (best.get(s) or {}).get("go", {}).get(aspect) or []
        v = v if isinstance(v, list) else [v]
        t |= {e["id"] for e in v}
    terms.append(t)
vocab = sorted(set().union(*terms)); idx = {t: i for i, t in enumerate(vocab)}
B = np.zeros((150, len(vocab)), np.float32)
for i, t in enumerate(terms):
    for x in t: B[i, idx[x]] = 1
df = B.sum(0); idf = np.log((1 + 150) / (1 + df)) + 1
T = B * idf; T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
U, S, Vt = np.linalg.svd(T, full_matrices=False)  # exact LAPACK SVD (deterministic); keep top K
sign = np.sign(Vt[:K, :].sum(1)); sign[sign == 0] = 1
X = (U[:, :K] * S[:K]) * sign
go_in = OUT.parent / "go_annotations_150.json"; json.dump(dict(zip(ensembl, [sorted(t) for t in terms])), open(go_in, "w"))
manifest["sources"]["go_svd"] = finish("go_svd", X, [s for s, t in zip(symbols, terms) if not t], {
    "url": "https://mygene.info/v3/query (fields=go; GOA annotations as served by MyGene.info / NCBI)",
    "version": "MyGene.info live API, queried 2026-10-14 UTC; GO release not pinned by the API (pin a GO release for the final freeze)",
    "license_model_or_data": "GO data CC BY 4.0", "license_outputs": "derived SVD features: CC BY 4.0 attribution applies",
    "dim": K, "fit": {"input_file": go_in.name, "input_sha256": sha(go_in), "terms": len(vocab), "aspects": "BP+MF+CC, all evidence codes",
                      "score_threshold": "none (binary membership)", "normalization": "TF-IDF (smooth idf over the 150 genes) then L2 row norm",
                      "solver": "numpy.linalg.svd (exact, LAPACK); sign fixed so each component's loadings sum >= 0", "seed": "n/a (deterministic)",
                      "n_components": K, "gene_universe": "the 150 knockdown genes only (transductive; uses annotations, never expression)"}})

# --- STRING (spectral embedding of the full human network) ---
alias = {}
with gzip.open(COV / "string_aliases.txt.gz", "rt") as f:
    for l in f:
        p = l.rstrip("\n").split("\t")
        if len(p) == 3 and p[2] == "Ensembl_gene": alias.setdefault(p[1], p[0])
prot = {}; r = []; c = []; w = []; THR = 700
with gzip.open(COV / "string_links.txt.gz", "rt") as f:
    next(f)
    for l in f:
        a, b, s = l.split()
        s = int(s)
        if s < THR: continue
        r.append(prot.setdefault(a, len(prot))); c.append(prot.setdefault(b, len(prot))); w.append(s / 1000.0)
n = len(prot)
A = sp.coo_matrix((w, (r, c)), shape=(n, n)).tocsr(); A = A.maximum(A.T)
d = np.asarray(A.sum(1)).ravel(); Dm = sp.diags(1 / np.sqrt(np.maximum(d, 1e-12)))
N = (Dm @ A @ Dm).tocsr()
rng = np.random.default_rng(SEED); v0 = rng.standard_normal(n)
U, S, Vt = svds(N, k=K, v0=v0, solver="arpack", random_state=SEED)
o = np.argsort(-S); U, S = U[:, o], S[o]
sign = np.sign(U.sum(0)); sign[sign == 0] = 1; U = U * sign
X = np.zeros((150, K), np.float32); miss = []
for i, e in enumerate(ensembl):
    p = alias.get(e)
    if p is None or p not in prot: miss.append(symbols[i])
    else: X[i] = U[prot[p]] * S
manifest["sources"]["string_spectral"] = finish("string_spectral", X, miss, {
    "url": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
    "version": "STRING v12.0", "license_model_or_data": "CC BY 4.0", "license_outputs": "derived features: CC BY 4.0 attribution applies", "dim": K,
    "fit": {"input_file": "9606.protein.links.v12.0.txt.gz", "input_sha256": sha(COV / "string_links.txt.gz"),
            "id_map": "9606.protein.aliases.v12.0.txt.gz (Ensembl_gene -> first listed STRING protein)", "id_map_sha256": sha(COV / "string_aliases.txt.gz"),
            "score_threshold": f"combined_score >= {THR}", "normalization": "symmetric, weights score/1000, D^-1/2 A D^-1/2",
            "solver": "scipy.sparse.linalg.svds (arpack), top singular vectors, features = U*S, sign fixed by column sum", "seed": SEED,
            "n_components": K, "gene_universe": f"all {n} human STRING proteins with an edge >= {THR}"}})

json.dump(manifest, open(OUT / "manifest_partial.json", "w"), indent=1)
