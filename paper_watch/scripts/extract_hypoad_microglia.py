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

The output keeps raw counts (layers/counts when the source has it, then
raw/X, then X), gene symbols as var_names, and obs columns donor, condition and cell_type
under those standard names. uns['hypoad_extract'] records the provenance,
including the cell-type labels kept and excluded and the cells per raw
condition label; check these on the first real run.

Two guards apply. Labels that match the microglia pattern but also name
macrophages, perivascular or border-associated cells, or monocytes (for
example 'Micro-PVM' or 'Microglia/Macrophage') are excluded by
--exclude-pattern ('' disables this). The chosen matrix must hold integer
counts; a scaled or log-transformed matrix is refused unless
--allow-noninteger is passed, because the readout normalizes raw counts.

States. The input must be a finished download: SOURCE_DONE means
parallel_download.py left INPUT.receipt.json (and INPUT.state.json says
DONE) and the input still has the receipt's size and size+mtime stamp.
NO_SOURCE (no input) refuses; SOURCE_INCOMPLETE (an assembly, a plan or
parts, or a file without a receipt) and SOURCE_CHANGED (finished, then
replaced or modified) refuse unless --input-unverified REASON is given,
and the reason is recorded. The receipt's verified_by ('sha256',
'validator' or 'size-only') is recorded as input_verified_by. The output
is written to a hidden ``.<name>.<pid>.partial`` beside it and moved into
place with os.replace, so a killed extract leaves no file at --output.
uns['hypoad_extract'] records schema 2, the parameters and the input's
sha256 from its receipt. An existing output is EXTRACTED (the parameters
recorded equal the ones REQUESTED now, and the same input: nothing to do,
unless --force), STALE (other parameters or input: re-extracted) or CORRUPT
(unreadable: re-extracted). Non-default parameters are allowed and
idempotent here; the loader warns about them (HYPOAD_STRICT=1 refuses).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

PW = Path(__file__).resolve().parent.parent
if str(PW) not in sys.path:
    sys.path.insert(0, str(PW))

from alztest.data import (  # noqa: E402
    HYPOAD_EXTRACT_SCHEMA, HYPOAD_MICROGLIA, HYPOAD_NOT_MICROGLIA, hypoad_extract_provenance,
    hypoad_nondefault_params, hypoad_source_state,
)

DATA_DIR = PW / "data"
DEFAULT_INPUT = DATA_DIR / "GSE342635_HypoAD.h5ad"
DEFAULT_OUTPUT = DATA_DIR / "hypoad_microglia.h5ad"

CELLTYPE_COLS = ("cell_type", "celltype", "CellType", "cell_class", "major_celltype",
                 "major_cell_type", "broad_celltype", "class", "annotation", "subclass")
DONOR_COLS = ("donor", "donor_id", "sample", "Sample", "individual", "Donor")
COND_COLS = ("condition", "diagnosis", "group", "Group", "disease", "status")
GENE_COLS = ("feature_name", "gene_symbols", "gene_symbol", "gene_name", "gene_names",
             "symbol", "gene_ids")
MICROGLIA = HYPOAD_MICROGLIA
NOT_MICROGLIA = HYPOAD_NOT_MICROGLIA


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
    if "raw" in f and "X" in f["raw"]:
        return "raw/X"
    return "X"


def microglia_mask(labels: pd.Series, pattern: str, exclude: str) -> np.ndarray:
    mask = labels.str.match(pattern).to_numpy()
    if exclude:
        mask &= ~labels.str.contains(exclude, regex=True).to_numpy()
    return mask


def is_integer_valued(m) -> bool:
    data = m.data if sparse.issparse(m) else np.asarray(m)
    return bool(np.all(np.isfinite(data)) and np.all(data == np.round(data)))


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


def extract(src: Path, dst: Path, *, pattern: str = MICROGLIA,
            exclude_pattern: str = NOT_MICROGLIA, celltype_col=None,
            donor_col=None, condition_col=None, gene_col=None,
            allow_noninteger: bool = False, input_sha256: str = "",
            input_unverified: str = "", input_verified_by: str = "") -> dict:
    import anndata as ad

    with h5py.File(src, "r") as f:
        obs = _read_elem(f["obs"])
        var = _read_elem(f["var"])
        ct = celltype_column(obs, pattern, celltype_col)
        dcol = _pick(obs.columns, DONOR_COLS, "donor", donor_col)
        ccol = _pick(obs.columns, COND_COLS, "condition", condition_col)
        labels = obs[ct].astype(str)
        matched = labels.str.match(pattern).to_numpy()
        mask = microglia_mask(labels, pattern, exclude_pattern)
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            raise SystemExit(f"no cells match {pattern!r} in obs[{ct!r}]")
        key = matrix_key(f)
        if key == "raw/X":
            var = _read_elem(f["raw/var"])
        counts = read_rows(f[key], rows)
    integer = is_integer_valued(counts)
    if not integer and not allow_noninteger:
        raise SystemExit(
            f"{key} is not integer-valued, so it is not raw counts; the readout "
            "normalizes raw counts. Pass --allow-noninteger to write it anyway.")
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
    st = Path(src).stat()
    info = {
        "schema": HYPOAD_EXTRACT_SCHEMA,
        "params": extract_params(pattern=pattern, exclude_pattern=exclude_pattern,
                                 allow_noninteger=allow_noninteger, celltype_col=celltype_col,
                                 donor_col=donor_col, condition_col=condition_col,
                                 gene_col=gene_col),
        "input_sha256": input_sha256 or "",
        "input_unverified": input_unverified or "",
        "input_verified_by": input_verified_by or "",
        "input_size": int(st.st_size),
        "source": str(src), "matrix": key, "n_cells_total": int(len(obs)),
        "n_cells_kept": int(rows.size), "celltype_col": ct, "donor_col": dcol,
        "condition_col": ccol, "pattern": pattern,
        "labels_kept": sorted(set(labels.iloc[rows])),
        "labels_excluded": sorted(set(labels[matched & ~mask])),
        "integer_counts": integer,
        # parallel lists, because uns keys become HDF5 names
        "condition_labels": sorted(set(out_obs["condition"])),
        "condition_cells": [int((out_obs["condition"] == c).sum())
                            for c in sorted(set(out_obs["condition"]))],
    }
    adata.uns["hypoad_extract"] = info
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.partial")
    try:
        adata.write_h5ad(tmp)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    dirfd = os.open(dst.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)
    return info


def extract_params(*, pattern, exclude_pattern, allow_noninteger, celltype_col=None,
                   donor_col=None, condition_col=None, gene_col=None) -> dict:
    """The parameters recorded in uns (h5ad cannot store None, so '' stands for unset)."""
    return {"pattern": pattern, "exclude_pattern": exclude_pattern,
            "allow_noninteger": bool(allow_noninteger), "celltype_col": celltype_col or "",
            "donor_col": donor_col or "", "condition_col": condition_col or "",
            "gene_col": gene_col or ""}


def output_state(dst: Path, params: dict, source_receipt: dict | None) -> tuple[str, list[str]]:
    """NO_OUTPUT, CORRUPT, STALE, STALE_UNKNOWN or EXTRACTED for --output."""
    if not dst.exists() and not dst.is_symlink():
        return "NO_OUTPUT", []
    if not dst.is_file():
        return "CORRUPT", [f"{dst} is not a regular file"]
    try:
        import anndata as ad
        backed = ad.read_h5ad(dst, backed="r")
        try:
            info = backed.uns.get("hypoad_extract")
            info = dict(info) if info is not None else None
        finally:
            backed.file.close()
    except Exception as e:  # noqa: BLE001
        return "CORRUPT", [repr(e)]
    # Provenance only: the parameter check below is against the REQUESTED
    # parameters, so an extract made with --allow-noninteger is current for
    # a rerun with --allow-noninteger.
    state, why = hypoad_extract_provenance(info, source_receipt)
    if state != "EXTRACTED":
        return state, why
    recorded = {k: (bool(v) if isinstance(params[k], bool) else str(v))
                for k, v in dict(info.get("params") or {}).items() if k in params}
    if recorded != params:
        return "STALE", [f"recorded parameters {recorded} != requested {params}"]
    want = (source_receipt or {}).get("sha256", "")
    if str(info.get("input_sha256") or "") != want:
        return "STALE", ["the input differs from the one recorded"]
    return "EXTRACTED", []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--pattern", default=MICROGLIA, help="regex on the cell-type label")
    ap.add_argument("--exclude-pattern", default=NOT_MICROGLIA,
                    help="regex; matching labels are dropped ('' disables)")
    ap.add_argument("--allow-noninteger", action="store_true",
                    help="write a matrix that is not integer counts")
    ap.add_argument("--celltype-col")
    ap.add_argument("--donor-col")
    ap.add_argument("--condition-col")
    ap.add_argument("--gene-col")
    ap.add_argument("--input-unverified", metavar="REASON", default="",
                    help="extract from an input without a DONE download receipt; "
                         "the reason is recorded in uns")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even when --output is already EXTRACTED")
    a = ap.parse_args(argv)
    src_state, receipt = hypoad_source_state(a.input)
    if src_state == "NO_SOURCE" or not a.input.is_file():
        raise SystemExit(f"{a.input} not found ({src_state}); download GSE342635_HypoAD.h5ad first")
    if src_state != "SOURCE_DONE":
        if not a.input_unverified:
            raise SystemExit(
                f"{a.input} is {src_state}: no DONE receipt from parallel_download.py "
                f"({a.input}.receipt.json with its size). Finish the download, or pass "
                "--input-unverified REASON to extract it anyway")
        print(f"warning: extracting an unverified input ({a.input_unverified})", flush=True)
    params = extract_params(pattern=a.pattern, exclude_pattern=a.exclude_pattern,
                            allow_noninteger=a.allow_noninteger, celltype_col=a.celltype_col,
                            donor_col=a.donor_col, condition_col=a.condition_col,
                            gene_col=a.gene_col)
    nondefault = hypoad_nondefault_params({"params": params})
    if nondefault:
        print(f"note: non-default extraction parameters ({'; '.join(nondefault)}); "
              "load_hypoad_microglia() will warn about this extract, and refuse it "
              "under HYPOAD_STRICT=1", flush=True)
    if receipt is not None and receipt.get("verified_by") == "size-only":
        print(f"warning: {a.input} was verified by size only ({receipt.get('unvalidated')}); "
              "input_verified_by='size-only' is recorded", flush=True)
    out_state, why = output_state(a.output, params, receipt)
    if out_state == "EXTRACTED" and not a.force and receipt is not None:
        print(f"{a.output}: already EXTRACTED from this input with these parameters", flush=True)
        return 0
    if out_state not in ("NO_OUTPUT", "EXTRACTED"):
        print(f"{a.output} is {out_state} ({'; '.join(why)}); re-extracting", flush=True)
    info = extract(a.input, a.output, pattern=a.pattern,
                   exclude_pattern=a.exclude_pattern, celltype_col=a.celltype_col,
                   donor_col=a.donor_col, condition_col=a.condition_col,
                   gene_col=a.gene_col, allow_noninteger=a.allow_noninteger,
                   input_sha256=(receipt or {}).get("sha256", ""),
                   input_unverified=a.input_unverified if receipt is None else "",
                   input_verified_by=str((receipt or {}).get("verified_by") or ""))
    for k, v in info.items():
        print(f"{k}: {v}", flush=True)
    print(f"wrote {a.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
