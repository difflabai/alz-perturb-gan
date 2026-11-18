"""Data loaders for the paper-watch tests.

Each loader either returns real data or raises DataUnavailable with a
precise, checkable reason (access terms, accession, what was tried).
Tests convert DataUnavailable into a clean pytest skip — a gated dataset
is a finding about access, never a silent pass and never a fake run.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class DataUnavailable(RuntimeError):
    """Raised when a find's data cannot be obtained; carries the reason."""


def neuroko_path() -> Path:
    env = os.environ.get("NEUROKO_H5AD")
    if env:
        return Path(env)
    return DATA_DIR / "neuroko_annotated.h5ad"


def load_neuroko():
    """NeuroKO Perturb-seq (Dominicus et al., bioRxiv 2026.10.13.758778).

    Stated access in the preprint: annotated integrated scRNA-seq objects
    on Zenodo (DOI as printed: 10.5281/zenodo.2326038861 — the identifier
    does not resolve as printed; verified 2026-11-17 against the Zenodo
    API, Europe PMC, and the authors' GitHub), raw sequencing at ENA
    ERP134820, code at github.com/Lotfollahi-lab/NeuroID-reproducibility.
    """
    p = neuroko_path()
    if not p.exists():
        raise DataUnavailable(
            "NeuroKO annotated objects not present locally and the Zenodo "
            "DOI printed in the preprint (10.5281/zenodo.2326038861) does "
            "not resolve via the Zenodo API (checked 2026-11-17); raw data "
            "exists at ENA ERP134820 but is unprocessed FASTQ. Place the "
            "annotated h5ad at data/neuroko_annotated.h5ad or set "
            "NEUROKO_H5AD."
        )
    import anndata as ad
    return ad.read_h5ad(p)


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


def seaad_annotations(region: str = "LEC"):
    """Per-cell annotations for a SEA-AD region, as a DataFrame indexed
    by cell_label, with columns: donor_label, subclass, adnc.

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
    dis = pd.read_csv(dis_p, usecols=["donor_label",
                                       "Overall AD neuropathological Change",
                                       "Cognitive Status"])
    df = (cells.merge(clus, on="cell_label", how="inner")
               .merge(terms, on="cluster_label", how="left")
               .merge(dis, on="donor_label", how="left"))
    df = df.rename(columns={"Overall AD neuropathological Change": "adnc",
                            "Cognitive Status": "cognition"})
    return df.set_index("cell_label")


def load_panad():
    """panAD integrated atlas (bioRxiv 2026.09.03.747935).

    The preprint's full text is not PMC-archived (license) and no direct
    download of the integrated object or the MONET donor subtype labels
    was located on the preprint server / CELLxGENE as of 2026-11-17.
    """
    p = DATA_DIR / "panad.h5ad"
    if not p.exists():
        raise DataUnavailable(
            "panAD integrated object + MONET subtype labels not openly "
            "downloadable as of 2026-11-17 (preprint full text not "
            "archived; no CELLxGENE collection found). Place at "
            "data/panad.h5ad with donor subtype labels in obs to enable."
        )
    import anndata as ad
    return ad.read_h5ad(p)


def load_psychad():
    """PsychAD DLPFC atlas (Lee et al., Nature 2026).

    Controlled access: AD Knowledge Portal / Synapse syn60084804,
    requires a data use agreement. No credentialed Synapse session is
    configured in this environment (checked 2026-11-17).
    """
    p = DATA_DIR / "psychad.h5ad"
    if not p.exists():
        raise DataUnavailable(
            "PsychAD is controlled-access (Synapse syn60084804, DUA "
            "required); no approved download present. Place an approved "
            "export at data/psychad.h5ad to enable."
        )
    import anndata as ad
    return ad.read_h5ad(p)


def load_gnpc():
    """GNPC longitudinal plasma proteomics (medRxiv 2026.07.01.26356385).

    The v4.1 SomaScan resource (17,269 samples / 13,449 participants) is
    distributed by the Global Neurodegeneration Proteomics Consortium
    under consortium access terms; no public download exists.
    """
    p = DATA_DIR / "gnpc_plasma.parquet"
    if not p.exists():
        raise DataUnavailable(
            "GNPC plasma proteomics is consortium-access only; no public "
            "download. Place an approved export at data/gnpc_plasma.parquet "
            "to enable."
        )
    import pandas as pd
    return pd.read_parquet(p)
