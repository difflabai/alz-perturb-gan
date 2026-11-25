"""Restart-resilient Perez iteration-2 driver (cycle 13, v2).

Why v2: the test's own _prepare died silently TWICE in cycle 13 —
anndata's backed row slice (adata[rows]) over the 30 GB object
spikes memory exactly as diagnosed for NeuroKO astro in cycle 12
(4.5 GB peak for a 1.8 GB subset there). This driver monkeypatches
_prepare with a lean version that replicates the test's design
EXACTLY (same donor eligibility, same stratified 80/20 donor
split, same 300-cell cap, same normalize/log1p/variance/top-2000/
PCA-30/standardize pipeline via the shared harness) but reads rows
through the preallocated h5py CSR block reader proven in the
NeuroKO test (tests/test_neuroko_perturbation.py::_read_csr_rows,
copied here as runner plumbing). One logged deviation: capped-cell
draws iterate donors in sorted order (the test iterates a set —
hash order, already nondeterministic across processes; the split
itself is identical). Prep output caches to npz per seed; results
append to a JSONL per seed. The test file itself is untouched.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import test_perez_genotype_fit as T  # noqa: E402
from alztest.data import (load_perez_untreated_metadata,  # noqa: E402
                          perez_bin1_genotypes, perez_untreated_h5ad)
from alztest.harness import pca_fit_transform  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data"
OUT = CACHE / "perez_fit_results.jsonl"
MIN_CELLS_PER_DONOR = 200
MAX_CELLS_PER_DONOR = 300
N_COMP = 30


def _read_csr_rows(path, rows, block: int = 4096):
    """Preallocated CSR row-subset straight from the h5ad arrays
    (pattern from tests/test_neuroko_perturbation.py, cycle 12)."""
    import h5py
    with h5py.File(path, "r") as f:
        X = f["X"]
        assert X.attrs.get("encoding-type") in (b"csr_matrix", "csr_matrix")
        indptr = X["indptr"][:]
        shape = tuple(int(v) for v in X.attrs["shape"])
        total = int(sum(int(indptr[r + 1]) - int(indptr[r]) for r in rows))
        data = np.empty(total, dtype=np.float32)
        idx = np.empty(total, dtype=np.int32)
        new_indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        cur = 0
        for i, r in enumerate(rows):
            s, e = int(indptr[r]), int(indptr[r + 1])
            if e > s:
                data[cur:cur + (e - s)] = X["data"][s:e]
                idx[cur:cur + (e - s)] = X["indices"][s:e]
                cur += e - s
            new_indptr[i + 1] = cur
    return sparse.csr_matrix((data, idx, new_indptr),
                             shape=(len(rows), shape[1]))


def _donor_ids(path):
    import h5py
    with h5py.File(path, "r") as f:
        g = f["obs"]["donor_id"]
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c)
                         for c in g["categories"][:]])
        codes = g["codes"][:]
    return cats[codes]


def lean_prepare(seed: int):
    meta = load_perez_untreated_metadata()
    geno = perez_bin1_genotypes()
    counts = meta["donor_id"].astype(str).value_counts()
    donors = [d for d in counts.index[counts >= MIN_CELLS_PER_DONOR]
              if d in geno]
    assert len(donors) >= 60, f"only {len(donors)} donors"
    print(f"donors>={MIN_CELLS_PER_DONOR} genotyped: {len(donors)}",
          flush=True)
    rng = np.random.default_rng(seed)
    train_d, test_d = [], []
    for g in (0, 1, 2):
        gd = [d for d in donors if geno[d] == g]
        rng.shuffle(gd)
        k = max(1, int(round(0.2 * len(gd))))
        test_d += gd[:k]
        train_d += gd[k:]
    path = perez_untreated_h5ad()
    did = _donor_ids(path)
    rows_by = {}
    for split, dset in (("train", train_d), ("test", test_d)):
        rows = []
        for d in sorted(dset):
            idx = np.where(did == d)[0]
            if len(idx) > MAX_CELLS_PER_DONOR:
                idx = rng.choice(idx, MAX_CELLS_PER_DONOR, replace=False)
            rows.append(np.sort(idx))
        rows_by[split] = np.sort(np.concatenate(rows))
        print(f"{split}: {len(rows_by[split])} cells", flush=True)

    def load(rows):
        X = _read_csr_rows(path, rows).astype(np.float32)
        tot = np.asarray(X.sum(1)).ravel() + 1e-9
        X = X.multiply(1e4 / tot[:, None]).tocsr()
        X.data = np.log1p(X.data)
        return X, did[rows]

    Xtr, dtr = load(rows_by["train"])
    Xte, dte = load(rows_by["test"])
    print(f"loaded Xtr {Xtr.shape} Xte {Xte.shape}", flush=True)
    n = Xtr.shape[0]
    s1 = np.asarray(Xtr.sum(0)).ravel()
    s2 = np.asarray(Xtr.multiply(Xtr).sum(0)).ravel()
    var = s2 / n - (s1 / n) ** 2
    top = np.argsort(var)[::-1][:2000]
    Ptr, proj = pca_fit_transform(
        np.asarray(Xtr[:, top].todense(), dtype=np.float64),
        n_comp=N_COMP, seed=seed)
    mu, sd = Ptr.mean(0), Ptr.std(0) + 1e-9
    Ptr = (Ptr - mu) / sd
    Pte = ((np.asarray(Xte[:, top].todense(), dtype=np.float64)
            - proj[0]) @ proj[1].T - mu) / sd
    print("prep complete", flush=True)
    return (Ptr, np.array([geno[d] for d in dtr]),
            Pte, np.array([geno[d] for d in dte]))


def cached_prepare(seed, **kw):
    f = CACHE / f"perez_prep_seed{seed}.npz"
    if f.exists():
        z = np.load(f)
        return z["Xtr"], z["ctr"], z["Xte"], z["cte"]
    out = lean_prepare(seed)
    np.savez_compressed(f, Xtr=out[0], ctr=out[1], Xte=out[2], cte=out[3])
    return out


T._prepare = cached_prepare

if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [0]
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["seed"])
    for s in seeds:
        if s in done:
            print(f"seed {s} already done, skipping", flush=True)
            continue
        r = T.run_once(seed=s)
        with OUT.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)
