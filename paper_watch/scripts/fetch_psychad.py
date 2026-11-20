"""Fetch an analysis-ready slice of the public PsychAD MSSM cohort."""
import argparse
import hashlib
import json
from pathlib import Path

import cellxgene_census as census_api
import numpy as np
from scipy.sparse import csr_matrix

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
def fetch(out: Path):
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
    partial = out / "psychad-mssm-microglia.partial.h5ad"
    matrix.write_h5ad(partial, compression="gzip", compression_opts=1)
    partial.replace(path)
    with path.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    receipt = {
        "source": str(dataset["citation"]), "census_release": RELEASE,
        "dataset_id": DATASET, "dataset_version_id": str(dataset["dataset_version_id"]),
        "query": query, "source_to_analysis_labels": GROUPS,
        "cap_per_donor": LIMIT, "sampling_seed": 0,
        "shape": list(matrix.shape), "sha256": sha,
        "diagnosis_counts": by_group.rename(columns={"size": "nuclei", "nunique": "donors"}).to_dict(orient="index"),
    }
    (out / "psychad-mssm-microglia.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    fetch(args.output_dir)


if __name__ == "__main__":
    main()
