"""Extract the HypoAD microglia compartment from the local GEO object.

Yu et al., Nature 2026 (doi 10.1038/s41586-026-11305-w); processed data
GEO GSE342635, file GSE342635_HypoAD.h5ad (51,107,823,722 B). Download it
first (scripts/parallel_download.py) into a gitignored directory with
more than 60 GB free, then run

    python paper_watch/scripts/extract_hypoad_microglia.py \
        [--input .../GSE342635_HypoAD.h5ad] [--output .../hypoad_microglia.h5ad]

Defaults resolve from this file's location, not the working directory:
paper_watch/data/GSE342635_HypoAD.h5ad -> paper_watch/data/hypoad_microglia.h5ad.

The obs and var tables are read with anndata's element reader and the
microglia rows are pulled from the count matrix in row windows with h5py.
This keeps memory to the microglia rows. anndata's backed='r' mode backs
only X and would load every layer of the 51 GB object into memory.

The output keeps raw counts (layers/counts when the source has it, else X),
gene symbols as var_names, and obs columns donor, condition and cell_type
under those standard names. uns['hypoad_extract'] records the provenance.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

PW = Path(__file__).resolve().parent.parent
DATA_DIR = PW / "data"
DEFAULT_INPUT = DATA_DIR / "GSE342635_HypoAD.h5ad"
DEFAULT_OUTPUT = DATA_DIR / "hypoad_microglia.h5ad"

CELLTYPE_COLS = ("cell_type", "celltype", "CellType", "cell_class", "major_celltype",
                 "major_cell_type", "broad_celltype", "class", "annotation", "subclass")
DONOR_COLS = ("donor", "donor_id", "sample", "Sample", "individual", "Donor")
COND_COLS = ("condition", "diagnosis", "group", "Group", "disease", "status")
GENE_COLS = ("feature_name", "gene_symbols", "gene_symbol", "gene_name", "gene_names",
             "symbol", "gene_ids")
MICROGLIA = r"(?i)^(microglia|micro|mg)([^a-z].*)?$"


def _read_elem(obj):
    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem
    return read_elem(obj)


def _pick(columns, candidates, what, given=None):
    if given:
        if given not in columns:
            raise SystemExit(f"--{what}-col {given!r} not in obs/var columns {list(columns)}")
        return given
    for c in candidates:
        if c in columns:
            return c
    raise SystemExit(f"no {what} column among {list(columns)}; pass --{what}-col")


def celltype_column(obs: pd.DataFrame, pattern: str, given: str | None = None) -> str:
    """The obs column whose labels match the microglia pattern."""
    if given:
        return _pick(obs.columns, (), "celltype", given)
    rx = re.compile(pattern)
    hits = []
    for c in obs.columns:
        values = obs[c].astype(str).unique()
        if len(values) > 500:
            continue
        if any(rx.match(v) for v in values):
            hits.append(c)
    if not hits:
        raise SystemExit(f"no obs column has a label matching {pattern!r}; pass --celltype-col")
    preferred = [c for c in CELLTYPE_COLS if c in hits]
    if preferred:
        return preferred[0]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"several obs columns match {pattern!r}: {hits}; pass --celltype-col")


def matrix_key(f: h5py.File) -> str:
    if "layers" in f and "counts" in f["layers"]:
        return "layers/counts"
    return "X"


def read_rows(obj, rows: np.ndarray, window: int = 10_000) -> sparse.csr_matrix:
    """Rows of an on-disk CSR group (or dense dataset), reading in row windows."""
    rows = np.asarray(rows, dtype=np.int64)
    if np.any(np.diff(rows) <= 0):
        raise ValueError("rows must be strictly increasing")
    if isinstance(obj, h5py.Dataset):
        blocks = [obj[rows[i:i + window]] for i in range(0, len(rows), window)]
        return sparse.csr_matrix(np.vstack(blocks) if blocks else np.zeros((0, obj.shape[1])))
    enc = obj.attrs.get("encoding-type")
    enc = enc.decode() if isinstance(enc, bytes) else enc
    if enc != "csr_matrix":
        raise ValueError(f"expected a csr_matrix group, found encoding-type {enc!r}")
    n_cols = int(obj.attrs["shape"][1])
    indptr = obj["indptr"][:]
    out, i = [], 0
    while i < len(rows):
        start = int(rows[i])
        j = int(np.searchsorted(rows, start + window, side="left"))
        stop = int(rows[j - 1]) + 1
        lo, hi = int(indptr[start]), int(indptr[stop])
        block = sparse.csr_matrix(
            (obj["data"][lo:hi], obj["indices"][lo:hi], indptr[start:stop + 1] - lo),
            shape=(stop - start, n_cols))
        out.append(block[rows[i:j] - start])
        i = j
    if not out:
        return sparse.csr_matrix((0, n_cols))
    return sparse.vstack(out, format="csr")


def gene_names(var: pd.DataFrame, given: str | None = None) -> pd.Index:
    if given:
        return pd.Index(var[_pick(var.columns, (), "gene", given)].astype(str))
    index = var.index.astype(str)
    if index.str.startswith("ENS").mean() > 0.5:
        for c in GENE_COLS:
            if c in var.columns and not var[c].astype(str).str.startswith("ENS").all():
                return pd.Index(var[c].astype(str))
    return pd.Index(index)


def extract(src: Path, dst: Path, *, pattern: str = MICROGLIA, celltype_col=None,
            donor_col=None, condition_col=None, gene_col=None) -> dict:
    import anndata as ad

    with h5py.File(src, "r") as f:
        obs = _read_elem(f["obs"])
        var = _read_elem(f["var"])
        ct = celltype_column(obs, pattern, celltype_col)
        dcol = _pick(obs.columns, DONOR_COLS, "donor", donor_col)
        ccol = _pick(obs.columns, COND_COLS, "condition", condition_col)
        labels = obs[ct].astype(str)
        mask = labels.str.match(pattern).to_numpy()
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            raise SystemExit(f"no cells match {pattern!r} in obs[{ct!r}]")
        key = matrix_key(f)
        counts = read_rows(f[key], rows)
    sub = obs.iloc[rows]
    out_obs = pd.DataFrame({
        "donor": sub[dcol].astype(str).to_numpy(),
        "condition": sub[ccol].astype(str).to_numpy(),
        "cell_type": labels.iloc[rows].to_numpy(),
    }, index=sub.index.astype(str))
    names = gene_names(var, gene_col)
    adata = ad.AnnData(X=counts.astype(np.float32), obs=out_obs,
                       var=pd.DataFrame(index=names))
    adata.var_names_make_unique()
    info = {
        "source": str(src), "matrix": key, "n_cells_total": int(len(obs)),
        "n_cells_kept": int(rows.size), "celltype_col": ct, "donor_col": dcol,
        "condition_col": ccol, "pattern": pattern,
        "labels_kept": sorted(set(labels.iloc[rows])),
    }
    adata.uns["hypoad_extract"] = info
    dst.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(dst)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--pattern", default=MICROGLIA, help="regex on the cell-type label")
    ap.add_argument("--celltype-col")
    ap.add_argument("--donor-col")
    ap.add_argument("--condition-col")
    ap.add_argument("--gene-col")
    a = ap.parse_args(argv)
    if not a.input.exists():
        raise SystemExit(f"{a.input} not found; download GSE342635_HypoAD.h5ad first")
    info = extract(a.input, a.output, pattern=a.pattern, celltype_col=a.celltype_col,
                   donor_col=a.donor_col, condition_col=a.condition_col, gene_col=a.gene_col)
    for k, v in info.items():
        print(f"{k}: {v}", flush=True)
    print(f"wrote {a.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
