"""Data loaders for the paper-watch tests.

Each loader either returns real data or raises DataUnavailable with a
precise, checkable reason (access terms, accession, what was tried).
Tests convert DataUnavailable into a clean pytest skip — a gated dataset
is a finding about access, never a silent pass and never a fake run.
"""
from __future__ import annotations

import csv
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
    dis = pd.read_csv(dis_p, usecols=["donor_label", "Overall AD neuropathological Change"])
    df = (cells.merge(clus, on="cell_label", how="inner")
               .merge(terms, on="cluster_label", how="left")
               .merge(dis, on="donor_label", how="left"))
    df = df.rename(columns={"Overall AD neuropathological Change": "adnc"})
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
PEREZ_UNTREATED_H5AD_SIZE = 31365866293


def perez_untreated_dir() -> Path:
    return DATA_DIR / "perez_untreated" / "untreated_h5ad_and_metadata"


def load_perez_untreated_metadata():
    """Perez-Alcantara pooled iPSC-microglia, Untreated arm — cell metadata.

    Washer & Perez-Alcantara et al., Nature Genetics 2026. Zenodo
    record 10.5281/zenodo.21471267 (v3, CC-BY 4.0): the Untreated zip
    (5,727,317,598 B, assembled byte-exact 2026-11-22) carries
    untreated.h5ad (31,365,866,293 B uncompressed) and
    untreated_metadata.csv (862,274 cells x 242 donors, demultiplexed:
    donor_id / prob_max / best_singlet). Iteration-1 sanity lives in
    tests/test_perez_genotype.py; the counts-vs-normalized check on
    the h5ad itself waits on the local extraction completing.
    """
    p = perez_untreated_dir() / "untreated_metadata.csv"
    if not p.exists():
        raise DataUnavailable(
            "Perez-Alcantara Untreated metadata not present locally. "
            "Download untreated_h5ad_and_metadata.zip from Zenodo "
            "record 10.5281/zenodo.21471267 into data/ and extract "
            "into data/perez_untreated/.")
    import pandas as pd
    return pd.read_csv(p, index_col=0, low_memory=False)


def perez_untreated_h5ad() -> Path:
    """Path to the extracted Untreated h5ad; DataUnavailable until the
    extraction is complete (partial files fail the size check)."""
    p = perez_untreated_dir() / "untreated.h5ad"
    if not p.exists() or p.stat().st_size != PEREZ_UNTREATED_H5AD_SIZE:
        have = p.stat().st_size if p.exists() else 0
        raise DataUnavailable(
            f"Perez-Alcantara Untreated h5ad extraction incomplete: "
            f"{have} of {PEREZ_UNTREATED_H5AD_SIZE} bytes present at {p}.")
    return p


def perez_bin1_genotypes() -> dict:
    """Donor -> BIN1 eQTL (2_127135234_C_T) dosage 0/1/2, extracted
    from the record VCF by scripts/perez_extract_bin1.py."""
    p = DATA_DIR / "perez_bin1_genotypes.csv"
    if not p.exists():
        raise DataUnavailable(
            f"Perez BIN1 genotypes not extracted (expected {p}; "
            "run scripts/perez_extract_bin1.py)")
    out = {}
    with p.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            name = str(row["donor_id"])
            dosage = int(row["bin1_dosage"])
            out[name] = dosage
            # The VCF sample names carry the full HipSci id
            # (HPSI0114i-bezi_3) while the h5ad/metadata donor_id is
            # the short suffix (bezi_3). Suffixes are unique across
            # all 261 VCF samples (verified 2026-11-25, no dosage
            # conflicts), so register the suffix as an alias key.
            if name.startswith("HPSI") and "-" in name:
                out[name.split("-", 1)[1]] = dosage
    return out
