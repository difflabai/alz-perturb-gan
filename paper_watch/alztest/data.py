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


def load_gse289721():
    """GSE289721 Perturb-seq (Cardona et al., J Neuroinflammation 2025).

    CRISPRi of AD GWAS genes in human iPSC-derived microglia; the
    Perturb-seq arm covers 6 targets (AGFG2, EED, INPP5D, MS4A6A,
    PVR, RABEP1) + non-targeting control, 72 guide-capture features
    in the same matrices as the 36,601 genes. Public on GEO since
    2025-12-31 (GSE289721_RAW.tar, 4 libraries SS001/SS002/AS001/
    AS002). Guide assignment: dominant guide >= 50% of a cell's
    guide UMIs with >= 3 guide UMIs total; guide -> target via
    GSE289721_feature_ref4.csv.gz. Built object cached at
    data/gse289721_perturbseq.h5ad.
    """
    cache = DATA_DIR / "gse289721_perturbseq.h5ad"
    import anndata as ad
    if cache.exists():
        return ad.read_h5ad(cache)
    raw_dir = DATA_DIR / "gse289721"
    ref_p = DATA_DIR / "feature_ref4.csv.gz"
    if not raw_dir.exists() or not ref_p.exists():
        raise DataUnavailable(
            "GSE289721 raw files not present. Download GSE289721_RAW.tar "
            "and GSE289721_feature_ref4.csv.gz from GEO series GSE289721 "
            "into data/ and extract the tar into data/gse289721/.")
    import numpy as np
    import pandas as pd
    from scipy.io import mmread
    from scipy import sparse
    ref = pd.read_csv(ref_p)
    g2t = dict(zip(ref["id"], ref["target_gene_name"]))
    blocks, obs_frames = [], []
    for mtx_p in sorted(raw_dir.glob("*_matrix.mtx.gz")):
        stem = mtx_p.name.replace("_matrix.mtx.gz", "")
        feats = pd.read_csv(raw_dir / f"{stem}_features.tsv.gz", sep="\t",
                            header=None, names=["id", "name", "type"])
        barcodes = pd.read_csv(raw_dir / f"{stem}_barcodes.tsv.gz",
                               sep="\t", header=None)[0].astype(str)
        M = mmread(str(mtx_p)).tocsc()
        guide_rows = np.where(feats["type"].values == "CRISPR Guide Capture")[0]
        gene_rows = np.where(feats["type"].values == "Gene Expression")[0]
        G = M[guide_rows, :].toarray().astype(float)
        tot, am, mx = G.sum(0), G.argmax(0), G.max(0)
        ok = (tot >= 3) & (mx >= 0.5 * np.maximum(tot, 1))
        guide_ids = feats["id"].values[guide_rows]
        targets = np.array([g2t.get(guide_ids[i], "Unassigned") for i in am])
        targets = np.where(ok, targets, "Unassigned")
        X = M[gene_rows, :].T.tocsr()[ok]
        obs = pd.DataFrame({
            "target": targets[ok],
            "library": stem.split("_", 1)[1],
            "n_guide_umi": tot[ok].astype(int),
        }, index=[f"{stem}:{b}" for b in barcodes[ok]])
        blocks.append(X)
        obs_frames.append(obs)
        var_names = feats["id"].values[gene_rows]
    X = sparse.vstack(blocks).tocsr()
    obs = pd.concat(obs_frames)
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [str(v) for v in var_names]
    adata.obs["target"] = adata.obs["target"].astype(str)
    adata.write_h5ad(cache)
    return adata
