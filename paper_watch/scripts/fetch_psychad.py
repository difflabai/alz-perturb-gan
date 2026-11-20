"""Fetch an analysis-ready slice of the public PsychAD MSSM cohort.

Receipt ordering (so a crash never leaves an h5ad whose receipt describes
another file): the receipt (query, shape, content sha256) is written into
uns['psychad_receipt'] of a hidden partial h5ad; then the sidecar
``<stem>.receipt.pending.json`` (whole-file sha256 plus a size+mtime
stamp) is written atomically; then the partial replaces the h5ad; then the
pending receipt replaces ``<stem>.receipt.json``. After a crash at any step
either the receipt or the pending receipt matches the h5ad, which is what
alztest.data.load_psychad(verify=True) checks. The whole sequence holds an
exclusive lock on the h5ad (``<h5ad>.lock``), so two concurrent fetches
cannot interleave into an h5ad from one and a receipt from the other.

Fault injection for tests: FETCH_PSYCHAD_FAULT=<point> exits with 137 at
after_partial, after_pending or after_h5ad_replace.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

PW = Path(__file__).resolve().parent.parent
if str(PW) not in sys.path:
    sys.path.insert(0, str(PW))

from alztest import _durable_state as ds  # noqa: E402
from alztest.data import (  # noqa: E402
    PSYCHAD_RECEIPT_KEY, file_sha256, file_stamp, psychad_content_sha256, psychad_receipt_paths,
)

DATASET = "37a17b78-4864-4a42-b67b-31c00962795a"
RELEASE = "2025-11-17"
GROUPS = {
    "normal": "control",
    "dementia || Alzheimer disease": "Alzheimer",
    "Lewy body dementia": "Lewy",
    "dementia || Lewy body dementia": "Lewy",
}
LIMIT = 128
columns = ["soma_joinid", "dataset_id", "donor_id", "disease", "cell_type",
           "is_primary_data", "sex", "tissue"]
query = (
    f"dataset_id == '{DATASET}' and cell_type_ontology_term_id == 'CL:0000129' "
    f"and disease in {list(GROUPS)!r}"
)
def _fault(point: str) -> None:
    if os.environ.get("FETCH_PSYCHAD_FAULT") == point:
        os._exit(137)


def write_with_receipt(matrix, path: Path, receipt: dict) -> dict:
    """Write ``matrix`` to ``path`` with its receipt in uns and beside it,
    crash-safely and under an exclusive lock on ``path`` (a second writer waits)."""
    path = Path(path)
    with ds.FileLock(path, purpose="fetch_psychad.write_with_receipt"):
        return _write_with_receipt_locked(matrix, path, receipt)


def _write_with_receipt_locked(matrix, path: Path, receipt: dict) -> dict:
    receipt = dict(receipt, shape=list(matrix.shape),
                   content_sha256=psychad_content_sha256(matrix))
    # Nested mappings go into uns as JSON text (their keys are not HDF5-safe).
    matrix.uns[PSYCHAD_RECEIPT_KEY] = {
        k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in receipt.items()}
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        matrix.write_h5ad(partial, compression="gzip", compression_opts=1)
        with open(partial, "rb") as fh:
            os.fsync(fh.fileno())
        _fault("after_partial")
        full = dict(receipt, sha256=file_sha256(partial), stamp=file_stamp(partial))
        final_receipt, pending = psychad_receipt_paths(path)
        ds.atomic_write_bytes(pending, (json.dumps(full, indent=2) + "\n").encode())
        _fault("after_pending")
        os.replace(partial, path)
        _fault("after_h5ad_replace")
        os.replace(pending, final_receipt)
        dirfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        if partial.exists():
            partial.unlink()
    return full


def fetch(out: Path):
    import cellxgene_census as census_api

    out.mkdir(parents=True, exist_ok=True)
    with census_api.open_soma(census_version=RELEASE) as census:
        dataset = census["census_info"]["datasets"].read(
            value_filter=f"dataset_id == '{DATASET}'"
        ).concat().to_pandas().iloc[0]
        cells = census_api.get_obs(census, "homo_sapiens", value_filter=query, column_names=columns)
        cells = cells[cells["donor_id"].notna() & (cells["donor_id"] != "unknown")]
        if set(cells["disease"]) != set(GROUPS):
            raise ValueError("the public release must include all three diagnosis groups")
        if (cells.groupby("donor_id", observed=True)["disease"].nunique() > 1).any():
            raise ValueError("a donor appears in more than one diagnosis group")
        cells = cells.sort_values("soma_joinid")
        rng = np.random.default_rng(0)
        readers = []
        for _, donor in cells.groupby("donor_id", observed=True, sort=True):
            rows = donor["soma_joinid"].to_numpy()
            readers.extend(rng.choice(rows, LIMIT, replace=False) if len(rows) > LIMIT else rows)
        selected = np.sort(np.asarray(readers, dtype=np.int64))
        print("Fetching", len(selected), "nuclei from", cells["donor_id"].nunique(), "donors", flush=True)
        presence = census["census_data"]["homo_sapiens"].ms["RNA"]["feature_dataset_presence_matrix"]
        feature_rows = presence.read(([int(dataset["soma_joinid"])],)).tables().concat()
        genes = np.sort(feature_rows["soma_dim_1"].to_numpy())
        matrix = census_api.get_anndata(
            census, "homo_sapiens", obs_coords=selected,
            X_name="raw", obs_column_names=columns,
            var_column_names=["soma_joinid", "feature_id", "feature_name"],
        )

    matrix = matrix[:, np.isin(matrix.var["soma_joinid"].to_numpy(), genes)].copy()
    matrix.X = csr_matrix(matrix.X)
    matrix.obs["diagnosis"] = matrix.obs["disease"].map(GROUPS).astype(str)
    matrix.var_names = matrix.var["feature_id"].astype(str)
    if matrix.n_obs != len(selected) or not matrix.var_names.is_unique:
        raise ValueError("the downloaded expression axes differ from the query")
    by_group = matrix.obs.groupby("diagnosis", observed=True)["donor_id"].agg(["size", "nunique"])
    path = out / "psychad-mssm-microglia.h5ad"
    receipt = {
        "source": str(dataset["citation"]), "census_release": RELEASE,
        "dataset_id": DATASET, "dataset_version_id": str(dataset["dataset_version_id"]),
        "query": query, "source_to_analysis_labels": GROUPS,
        "cap_per_donor": LIMIT, "sampling_seed": 0,
        "diagnosis_counts": by_group.rename(
            columns={"size": "nuclei", "nunique": "donors"}).to_dict(orient="index"),
    }
    print(json.dumps(write_with_receipt(matrix, path, receipt), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    fetch(args.output_dir)


if __name__ == "__main__":
    main()
