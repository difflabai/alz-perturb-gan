"""Data loaders for the paper-watch tests.

Each loader either returns real data or raises DataUnavailable with a
precise, checkable reason (access terms, accession, what was tried).
Tests convert DataUnavailable into a clean pytest skip — a gated dataset
is a finding about access, never a silent pass and never a fake run.

Loaders for the closed, access-blocked NeuroKO, panAD and GNPC finds
were removed; their access facts and revival requirements are recorded
in notes/blocked_finds.md.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class DataUnavailable(RuntimeError):
    """Raised when a find's data cannot be obtained; carries the reason."""


def seaad_region_path(region: str = "LEC") -> Path:
    return DATA_DIR / f"{region}-10X-log2.h5ad"


ABC_BASE = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"


def load_seaad_obs(region: str = "LEC"):
    """SEA-AD multiregion processed snRNA (Allen Brain Cell Atlas, open).

    Source: expression_matrices/SEA-AD-Multiregion-10X/20260630/ in the
    public allen-brain-cell-atlas S3 bucket (Allen Institute Terms of
    Use, noncommercial research). Returns the backed AnnData so callers
    can subsample rows without loading the full matrix. The expression
    file's own obs carries only barcodes; annotations come from
    seaad_annotations().
    """
    p = seaad_region_path(region)
    if not p.exists():
        raise DataUnavailable(
            f"SEA-AD {region} file not downloaded to {p}. It is open at "
            f"{ABC_BASE}/expression_matrices/SEA-AD-Multiregion-10X/20260630/"
            f"{region}-10X-log2.h5ad"
        )
    import anndata as ad
    return ad.read_h5ad(p, backed="r")


def seaad_annotations(region: str = "LEC", *, include_cognition: bool = False):
    """Per-cell annotations for a SEA-AD region, as a DataFrame indexed
    by cell_label, with donor_label, subclass, and adnc columns.
    Pass include_cognition to also require the Cognitive Status field.

    Assembled from the ABC metadata release (all verified present in the
    public bucket on 2026-11-17):
      - metadata/SEA-AD-Multiregion-10X/20260801/cell_metadata.csv
        streamed and filtered to this region -> data/<region>_cells.csv
      - metadata/SEA-AD-Multiregion-taxonomy/20260711/
        cell_to_cluster_membership.csv filtered to this region's cells
        -> data/<region>_clusters.csv, joined to subclass names via
        cluster_annotation_term.csv -> data/cluster_terms.csv
      - metadata/SEA-AD-Multiregion-10X/20260801/disease.csv (donor
        pathology: 'Overall AD neuropathological Change' in
        Not AD / Low / Intermediate / High) -> data/disease.csv
    """
    import pandas as pd
    cells_p = DATA_DIR / f"{region.lower()}_cells.csv"
    clus_p = DATA_DIR / f"{region.lower()}_clusters.csv"
    terms_p = DATA_DIR / "cluster_terms.csv"
    dis_p = DATA_DIR / "disease.csv"
    missing = [str(p) for p in (cells_p, clus_p, terms_p, dis_p) if not p.exists()]
    if missing:
        raise DataUnavailable(f"SEA-AD annotation pieces missing: {missing}")
    cells = pd.read_csv(cells_p, usecols=[
        "cell_label", "cell_barcode", "barcoded_cell_sample_label", "donor_label"])
    clus = pd.read_csv(clus_p, usecols=["cell_label", "label"])
    clus = clus.rename(columns={"label": "cluster_label"})
    terms = pd.read_csv(terms_p)  # cluster_label (supertype) -> subclass
    disease_columns = ["donor_label", "Overall AD neuropathological Change"]
    if include_cognition:
        disease_columns.append("Cognitive Status")
    dis = pd.read_csv(dis_p, usecols=disease_columns)
    df = (cells.merge(clus, on="cell_label", how="inner")
               .merge(terms, on="cluster_label", how="left")
               .merge(dis, on="donor_label", how="left"))
    df = df.rename(columns={"Overall AD neuropathological Change": "adnc",
                            "Cognitive Status": "cognition"})
    return df.set_index("cell_label")


class DataMalformed(Exception):
    """Data is present but wrong: corrupt, stale, or missing what a test needs.

    Deliberately NOT a subclass of DataUnavailable: tests skip on
    DataUnavailable (the data is absent) and must FAIL on DataMalformed
    (the data is there and broken), so a bad file never reads as a skip.
    """


# ------------------------------------------------------------------ PsychAD

PSYCHAD_RECEIPT_KEY = "psychad_receipt"


def psychad_receipt_paths(path: Path) -> tuple[Path, Path]:
    """(receipt, pending receipt) sidecars of a PsychAD h5ad."""
    path = Path(path)
    return (path.with_name(path.stem + ".receipt.json"),
            path.with_name(path.stem + ".receipt.pending.json"))


def file_stamp(path: Path) -> dict:
    """Cheap identity of a file: size and mtime in ns (os.replace keeps both)."""
    st = Path(path).stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def file_sha256(path: Path) -> str:
    import hashlib
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def psychad_content_sha256(adata) -> str:
    """sha256 over X (as CSR, float64/int64) and the obs and var names.

    Stored inside the file (uns), where a whole-file hash cannot live.
    """
    import hashlib
    import numpy as np
    from scipy import sparse
    X = sparse.csr_matrix(adata.X)
    h = hashlib.sha256()
    h.update(repr(tuple(X.shape)).encode())
    for arr, dtype in ((X.data, np.float64), (X.indices, np.int64), (X.indptr, np.int64)):
        h.update(np.ascontiguousarray(arr, dtype=dtype).tobytes())
    for names in (adata.obs_names, adata.var_names):
        h.update("\0".join(map(str, names)).encode())
    return h.hexdigest()


def _psychad_sidecar(path: Path, full_sha: bool):
    """The sidecar receipt that describes the h5ad at ``path``, or None if none exists.

    A receipt carries a size+mtime ``stamp`` (new fetches) or only a
    ``sha256`` (fetches before the stamp). fetch_psychad writes the pending
    receipt, then replaces the h5ad, then replaces the receipt, so after a
    crash either the receipt or the pending receipt matches the h5ad.
    """
    import json
    import warnings
    found = []
    for candidate in psychad_receipt_paths(path):
        if not candidate.is_file():
            continue
        receipt = json.loads(candidate.read_text())
        found.append(candidate.name)
        stamp = receipt.get("stamp")
        if stamp is not None:
            if stamp != file_stamp(path):
                continue
            if full_sha and receipt.get("sha256") != file_sha256(path):
                raise DataMalformed(f"{path}: sha256 differs from {candidate.name} "
                                    "although size and mtime match")
            return receipt
        if full_sha:  # legacy receipt: only a full hash can tie it to the file
            if receipt.get("sha256") == file_sha256(path):
                return receipt
            continue
        warnings.warn(f"{candidate.name} has no size/mtime stamp (written before "
                      "receipts were stamped); pass full_sha=True to check it", stacklevel=3)
        return receipt
    if found:
        raise DataMalformed(f"{path} does not match its receipt(s) {found}: the h5ad "
                            "changed after it was fetched, or a fetch was interrupted")
    return None


def load_psychad(path: Path | None = None, *, verify: bool = False, full_sha: bool = False):
    """Public PsychAD MSSM microglial raw counts and three diagnosis labels.

    ``verify=True`` checks the receipt before returning: the sidecar's
    size+mtime stamp (``full_sha=True`` adds the whole-file sha256), then
    the shape (and with ``full_sha`` the content hash) in
    ``uns['psychad_receipt']``. Files fetched before receipts moved into
    uns fall back to the sidecar's shape. A mismatch raises DataMalformed.
    """
    if path is None:
        path = Path(os.environ.get("PSYCHAD_H5AD", DATA_DIR / "psychad-mssm-microglia.h5ad"))
    path = Path(path)
    if not path.is_file():
        raise DataUnavailable(
            f"PsychAD microglia export missing at {path}. Set PSYCHAD_H5AD "
            "to the public CELLxGENE extract described in notes/psychad_cross_disorder.md."
        )
    sidecar = _psychad_sidecar(path, full_sha) if verify else None
    import anndata as ad
    adata = ad.read_h5ad(path)
    if verify:
        inner = adata.uns.get(PSYCHAD_RECEIPT_KEY)
        ref = dict(inner) if inner is not None else sidecar
        if ref is None:
            raise DataMalformed(f"{path} has no receipt in uns or beside it")
        if [int(n) for n in ref["shape"]] != list(adata.shape):
            raise DataMalformed(f"{path}: shape {adata.shape} != receipt {list(ref['shape'])}")
        if full_sha and inner is not None and \
                str(inner.get("content_sha256")) != psychad_content_sha256(adata):
            raise DataMalformed(f"{path}: content hash differs from uns['{PSYCHAD_RECEIPT_KEY}']")
    return adata


# ------------------------------------------------------------------ HypoAD

HYPOAD_SOURCE_NAME = "GSE342635_HypoAD.h5ad"
HYPOAD_EXTRACT_SCHEMA = 2
# Extraction parameters that change which cells or values the readout sees.
# The extractor's defaults come from here. An extract made with others is
# EXTRACTED_NONDEFAULT: the loader warns (HYPOAD_STRICT=1 refuses it).
HYPOAD_MICROGLIA = r"(?i)^(microglia|micro|mg)([^a-z].*)?$"
HYPOAD_NOT_MICROGLIA = r"(?i)(?:macro|pvm|perivascular|border|mono|(?:^|[^a-z])bam(?:[^a-z]|$))"
HYPOAD_EXPECTED_PARAMS = {"pattern": HYPOAD_MICROGLIA, "exclude_pattern": HYPOAD_NOT_MICROGLIA,
                          "allow_noninteger": False}
# Suffixes written by scripts/parallel_download.py next to the source.
DOWNLOAD_STATE_SUFFIX = ".state.json"
DOWNLOAD_RECEIPT_SUFFIX = ".receipt.json"
DOWNLOAD_ASSEMBLY_SUFFIX = ".assembling"


def hypoad_microglia_path() -> Path:
    env = os.environ.get("HYPOAD_H5AD")
    if env:
        return Path(env)
    return DATA_DIR / "hypoad_microglia.h5ad"


def hypoad_source_path() -> Path:
    env = os.environ.get("HYPOAD_SOURCE")
    if env:
        return Path(env)
    return DATA_DIR / HYPOAD_SOURCE_NAME


def download_receipt_state(source: Path) -> tuple[str, dict | None]:
    """("DONE", receipt), ("CHANGED", receipt) or ("NONE", None).

    DONE: the receipt exists, the state file (if present) says DONE, and the
    file still has the receipt's size and size+mtime ``stamp`` (receipts
    without a stamp are checked by size only). CHANGED: a DONE receipt whose
    file was replaced or modified afterwards, so its sha256 no longer
    describes the bytes.
    """
    import json
    source = Path(source)
    receipt_p = Path(str(source) + DOWNLOAD_RECEIPT_SUFFIX)
    state_p = Path(str(source) + DOWNLOAD_STATE_SUFFIX)
    try:
        receipt = json.loads(receipt_p.read_text())
    except (FileNotFoundError, ValueError):
        return "NONE", None
    if not isinstance(receipt, dict):
        return "NONE", None
    if state_p.exists():
        try:
            if json.loads(state_p.read_text()).get("state") != "DONE":
                return "NONE", None
        except ValueError:
            return "NONE", None
    if not source.is_file():
        return "NONE", None
    stamp = receipt.get("stamp")
    if source.stat().st_size != receipt.get("size") or (
            stamp is not None and stamp != file_stamp(source)):
        return "CHANGED", receipt
    return "DONE", receipt


def download_receipt(source: Path) -> dict | None:
    """The DONE receipt of a parallel_download.py download, or None
    (see download_receipt_state)."""
    state, receipt = download_receipt_state(source)
    return receipt if state == "DONE" else None


def hypoad_source_state(source: Path) -> tuple[str, dict | None]:
    """NO_SOURCE, SOURCE_INCOMPLETE, SOURCE_CHANGED or SOURCE_DONE, with the
    DONE receipt (SOURCE_CHANGED: finished, then modified; not usable)."""
    source = Path(source)
    rstate, receipt = download_receipt_state(source)
    if rstate == "DONE":
        return "SOURCE_DONE", receipt
    if rstate == "CHANGED":
        return "SOURCE_CHANGED", None
    traces = [source, Path(str(source) + DOWNLOAD_STATE_SUFFIX),
              Path(str(source) + DOWNLOAD_ASSEMBLY_SUFFIX),
              Path(str(source) + ".plan"), Path(str(source) + ".part00")]
    if any(t.exists() for t in traces):
        return "SOURCE_INCOMPLETE", None
    return "NO_SOURCE", None


def hypoad_extract_provenance(info, source_receipt: dict | None = None) -> tuple[str, list[str]]:
    """EXTRACTED, STALE or STALE_UNKNOWN from provenance alone (no parameter policy).

    STALE_UNKNOWN: no schema-2 provenance (an extract from before the
    acquisition state machine). STALE: made from an input whose sha256
    differs from the source's DONE receipt. The extractor compares the
    recorded parameters with the REQUESTED ones on top of this.
    """
    if not info or int(info.get("schema", 0) or 0) != HYPOAD_EXTRACT_SCHEMA:
        return "STALE_UNKNOWN", ["no schema-2 provenance in uns['hypoad_extract']"]
    have = str(info.get("input_sha256") or "")
    if source_receipt and have and have != source_receipt.get("sha256"):
        return "STALE", [f"input_sha256 {have[:12]}… differs from the source receipt "
                         f"{str(source_receipt.get('sha256'))[:12]}…"]
    return "EXTRACTED", []


def hypoad_nondefault_params(info) -> list[str]:
    """The HYPOAD_EXPECTED_PARAMS an extract was NOT made with."""
    why = []
    params = dict((info or {}).get("params") or {})
    for key, want in HYPOAD_EXPECTED_PARAMS.items():
        got = params.get(key)
        if isinstance(want, bool):
            got = bool(got)
        if got != want:
            why.append(f"{key}={got!r}, default {want!r}")
    return why


def hypoad_extract_state(info, source_receipt: dict | None = None) -> tuple[str, list[str]]:
    """The loader's view: hypoad_extract_provenance, and EXTRACTED_NONDEFAULT
    for a current extract made with documented non-default parameters
    (--allow-noninteger, --exclude-pattern '', another --pattern)."""
    state, why = hypoad_extract_provenance(info, source_receipt)
    if state == "EXTRACTED":
        nondefault = hypoad_nondefault_params(info)
        if nondefault:
            return "EXTRACTED_NONDEFAULT", nondefault
    return state, why


def load_hypoad_microglia():
    """HypoAD hypothalamic microglia (Yu et al., Nature 2026).

    Source: GEO GSE342635 (public), processed series object
    GSE342635_HypoAD.h5ad — 51,107,823,722 bytes for 614,403 nuclei,
    with NO per-sample supplementary files and no per-cell-type
    subset except a neuron-only object (9.3 GB). The microglia
    compartment (~52k nuclei; 7 mostly early-stage AD + 7 age-matched
    controls among the post-mortem donors, with the familial donor AD_44
    excluded downstream by alztest.hypoad.EXCLUDE_DONORS) is produced by
    downloading the object (scripts/parallel_download.py) and running
    scripts/extract_hypoad_microglia.py, which writes
    data/hypoad_microglia.h5ad with raw counts and obs columns donor,
    condition and cell_type. Or set HYPOAD_H5AD.

    Raises DataUnavailable only when no extract exists (the source may be
    absent, downloading, or downloaded and not yet extracted; the message
    says which). Raises DataMalformed when something is at the path but is
    not a readable extract (CORRUPT) or was made from another input
    (STALE). An extract without schema-2 provenance (STALE_UNKNOWN), or one
    made with non-default extraction parameters (EXTRACTED_NONDEFAULT, for
    example --allow-noninteger), loads with a warning, or raises
    DataMalformed when HYPOAD_STRICT=1.
    """
    p = hypoad_microglia_path()
    source = hypoad_source_path()
    if not p.exists() and not p.is_symlink():
        src_state, _ = hypoad_source_state(source)
        raise DataUnavailable(
            f"HypoAD microglia extract not present at {p} (source {source}: {src_state}). "
            "GEO GSE342635 is open but ships one 51 GB object with no per-sample files; "
            "download GSE342635_HypoAD.h5ad with scripts/parallel_download.py, "
            "run scripts/extract_hypoad_microglia.py, or set HYPOAD_H5AD."
        )
    if not p.is_file():
        raise DataMalformed(f"HypoAD extract path {p} exists but is not a regular file (CORRUPT)")
    import anndata as ad
    try:
        adata = ad.read_h5ad(p)
    except Exception as e:  # noqa: BLE001 - any read failure means the file is unusable
        raise DataMalformed(f"HypoAD extract {p} is unreadable (CORRUPT): {e!r}") from e
    info = adata.uns.get("hypoad_extract")
    state, why = hypoad_extract_state(dict(info) if info is not None else None,
                                      download_receipt(source))
    if state == "STALE":
        raise DataMalformed(f"HypoAD extract {p} is STALE: {'; '.join(why)}; rerun "
                            "scripts/extract_hypoad_microglia.py")
    if state in ("STALE_UNKNOWN", "EXTRACTED_NONDEFAULT"):
        if os.environ.get("HYPOAD_STRICT") == "1":
            raise DataMalformed(f"HypoAD extract {p} is {state}: {'; '.join(why)} "
                                "(HYPOAD_STRICT=1)")
        import warnings
        hint = ("re-extract to record it" if state == "STALE_UNKNOWN"
                else "made with non-default extraction parameters")
        warnings.warn(f"HypoAD extract {p} is {state}: {'; '.join(why)}; {hint} "
                      "(HYPOAD_STRICT=1 makes this an error)", stacklevel=2)
    return adata
