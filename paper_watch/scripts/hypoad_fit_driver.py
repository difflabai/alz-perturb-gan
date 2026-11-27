"""HypoAD iteration-1 driver — lean prep, test's own run_once.

Cycle 14 (2026-11-27). The test's _prepare densifies the full
55,765 x 36,601 matrix in float64 (~16 GB) — impossible on this VM.
This driver replaces ONLY _prepare with a streaming h5py prep over
the slim extract (data/hypoad_microglia.h5ad, built by
scripts/hypoad_extract_local.py from the assembled 51 GB object):

- CP10k + log1p computed per row-block in float32 from layers/counts
- column variance by streaming sums (top-1500 selection as in test)
- selected columns (top-1500 + DAM/homeostatic module genes, sorted
  as in the test) gathered by manual CSR column gather and densified
  at 55,765 x ~1.52k float32 (~340 MB)
- prep cached to data/hypoad_prep.npz (seed-independent)

Two logged deviations from the test's _prepare, both data handling:
(1) float32 throughout (the NeuroKO/Perez lean-prep precedent;
    top-gene boundary churn is possible, design unchanged);
(2) condition 'AD_44' (donor AD7, the familial APOE4/4 case) is
    EXCLUDED, as in the paper's own differential analyses — the
    test's string mapping would fold 'AD_44' into 'AD'. 'Young'
    donors are excluded exactly as the test excludes them.
run_once, arms and assertions are the test's, unmodified.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import test_hypoad_microglia_readout as T  # noqa: E402
from alztest.data import DataUnavailable  # noqa: E402

SLIM = Path("data/hypoad_microglia.h5ad")
CACHE = Path("data/hypoad_prep.npz")
OUT = Path("data/hypoad_fit_results.jsonl")


def lean_prepare():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return (z["X"], z["group"], z["donors"],
                list(z["homeo_pos"]), list(z["dam_pos"]))
    f = h5py.File(SLIM, "r")
    counts = f["layers/counts"]
    indptr = counts["indptr"][:]
    n_rows = len(indptr) - 1
    # obs: donor + condition from categorical codes
    def cat(name):
        g = f[f"obs/{name}"]
        cats = [x.decode() if isinstance(x, bytes) else str(x)
                for x in g["categories"][:]]
        return np.array(cats, dtype=object)[g["codes"][:]]
    donors_all = cat("donor_id")
    cond_all = cat("condition")
    var_names = np.array([x.decode() if isinstance(x, bytes) else str(x)
                          for x in f["var/_index"][:]])
    keep = np.isin(cond_all, ["AD", "CTL"])
    rows = np.flatnonzero(keep)
    donors = donors_all[rows]
    group = np.where(cond_all[rows] == "AD", "AD", "CTRL").astype(object)
    # streaming CP10k+log1p: first pass row sums + column sums/sumsq
    ncol = len(var_names)
    s1 = np.zeros(ncol, dtype=np.float64)
    s2 = np.zeros(ncol, dtype=np.float64)
    lib = np.zeros(len(rows), dtype=np.float64)
    data_ds, idx_ds = counts["data"], counts["indices"]
    for i, r in enumerate(rows):
        a, b = int(indptr[r]), int(indptr[r + 1])
        if b == a:
            continue
        d = data_ds[a:b].astype(np.float64)
        lib[i] = d.sum()
    scale = 1e4 / (lib + 1e-9)
    for i, r in enumerate(rows):
        a, b = int(indptr[r]), int(indptr[r + 1])
        if b == a:
            continue
        d = data_ds[a:b].astype(np.float64) * scale[i]
        v = np.log1p(d)
        ix = idx_ds[a:b]
        np.add.at(s1, ix, v)
        np.add.at(s2, ix, v * v)
    n = float(len(rows))
    mean = s1 / n
    var = s2 / n - mean * mean
    mod_idx = [i for i, g in enumerate(var_names) if g in T.HOMEO | T.DAM]
    if len(mod_idx) < 15:
        raise DataUnavailable(f"only {len(mod_idx)} DAM/homeostatic genes found")
    top = [i for i in np.argsort(var)[::-1][:1500] if i not in mod_idx]
    cols = np.sort(np.array(top + mod_idx))
    colset = {int(c): j for j, c in enumerate(cols)}
    # second pass: densify selected columns only
    X = np.zeros((len(rows), len(cols)), dtype=np.float32)
    for i, r in enumerate(rows):
        a, b = int(indptr[r]), int(indptr[r + 1])
        if b == a:
            continue
        ix = idx_ds[a:b]
        v = np.log1p(data_ds[a:b].astype(np.float64) * scale[i]).astype(np.float32)
        for k, c in enumerate(ix):
            j = colset.get(int(c))
            if j is not None:
                X[i, j] = v[k]
    pos = {int(g): i for i, g in enumerate(cols)}
    homeo_pos = [pos[i] for i in mod_idx if var_names[i] in T.HOMEO]
    dam_pos = [pos[i] for i in mod_idx if var_names[i] in T.DAM]
    np.savez_compressed(CACHE, X=X, group=group, donors=donors,
                        homeo_pos=np.array(homeo_pos), dam_pos=np.array(dam_pos))
    print(f"prep cached: X {X.shape}, AD donors {sorted(set(donors[group=='AD']))}, "
          f"CTL donors {sorted(set(donors[group=='CTRL']))}", flush=True)
    return X, group, donors, homeo_pos, dam_pos


T._prepare = lean_prepare

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)["seed"])
print(f"already done: {sorted(done)}", flush=True)
for seed in (0, 1, 2):
    if seed in done:
        continue
    r = T.run_once(seed=seed, steps=300)
    with OUT.open("a") as fh:
        fh.write(json.dumps(r) + "\n")
    print("SEED DONE", json.dumps(r), flush=True)
print("HYPOAD GAUNTLET COMPLETE", flush=True)
