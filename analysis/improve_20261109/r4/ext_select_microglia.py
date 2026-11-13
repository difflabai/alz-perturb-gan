"""Microglia selection for the external MADRC (GSE268599) ITG nuclei, and the
external donor store README (round 4, external-donor workstream).

STORE LAYOUT  /ml2/alz/runs/improve_20261109/ext/GSE268599/
  counts/{donor}.npz        scipy CSR int32, cells x 36,601 raw UMI counts
                            (same file format as SEA-AD rebuilt/counts/{donor}.npz)
  counts/{donor}_names.npy  str cell names '{10x barcode}-{GSM}'
  counts/genes.npy          the 36,601 gene symbols, identical (same order) to
                            SEA-AD rebuilt/counts/genes.npy
  counts/gene_ids.npy       Ensembl IDs of the same axis; equal, element by
                            element, to the SEA-AD h5ad var gene_ids and to every
                            MADRC features.tsv (checked per library)
  cellmeta/{donor}.npz      per kept cell: umi, n_genes, mito, library, cluster,
                            lineage scores (mean log1p CP10k over marker sets),
                            hetero-doublet score, pvm_prob (SEA-AD-trained
                            Micro-PVM_1 classifier), pvm_flag
  receipts/{donor}.json     file hashes, raw-file hashes and the per-library
                            selection receipt
  donors.json               per donor: cohort, region, A, B, C, ADNC (NIA-AA 2012
                            table, recomputed), Braak, CERAD, sex, age (+ how 90+
                            is coded), APOE4 dose, cells kept, median UMIs, receipt;
                            plus the stage table, access terms and caveats
  validation.json           written by `python3 -B ext_select_microglia.py validate`
  models/seaad_ref.npz      SEA-AD development-donor reference statistics
Donor ids are 'MADRC-{Table S1 Donor.ID:02d}'. Donors are TRAINING donors only.

HOW SEA-AD DEFINED ITS SET  The rebuilt store holds SEA-AD MTG nuclei whose
'Supertype' is one of Micro-PVM_1, Micro-PVM_2, Micro-PVM_2_1-SEAAD,
Micro-PVM_2_3-SEAAD, Micro-PVM_3-SEAAD, Micro-PVM_4-SEAAD (the whole
'Microglia-PVM' subclass of the Immune class) with 'Used in analysis' True and
>= 100 such nuclei per donor (benchmark/extract_official_counts.py). It
therefore INCLUDES perivascular macrophages: Micro-PVM_1 (641 of the 17,032
development-donor nuclei, 3.8 %) has a PVM marker score (MRC1, CD163, F13A1,
LYVE1) of 2.0 versus 0.4-0.6 in the other supertypes. It excludes lymphocytes
and monocytes (separate SEA-AD supertypes). SEA-AD used unsorted nuclei; the
development microglia have >= 1,063 genes at the 1st percentile.

SELECTION (per library, version below). No author per-cell labels are openly
available: the GEO series has only per-library filtered matrices; the authors'
atlas download page (ad-progression-atlas.partners.org/download) returns HTTP
500 and the app is a Shiny session; the code repository (mindds/
ad-progression-study) ships no per-cell annotation. So a documented,
conservative marker pipeline is used:
  1. QC: n_genes >= 1000 (SEA-AD microglia 1st percentile 1,063), mitochondrial
     fraction < 5 %, log UMIs <= library median + 3 MAD (Wachter 2024 used
     median +- 3 MAD; their lower bounds of 800 genes/UMIs are looser).
  2. Library-wise log1p CP10k, 2,000 dispersion HVGs, z-scores clipped at 10,
     30-PC PCA, k-means k=20 (seed 0).
  3. Lineage scores = mean log1p CP10k over marker sets (MARKERS). A cluster is
     myeloid if (median microglia score 'mg' >= 0.75, or median PVM score >= 1.0
     with mg >= 0.5), max(mg, pvm) exceeds every non-myeloid lineage median by
     >= 0.5, and its T and B medians are < 0.3. (v1 required mg >= 1.0 and
     dropped whole PVM clusters; v2 fixes that so PVMs are kept as in SEA-AD.)
     v3 (2026-11-11 review finding D1) adds an AMBIENT-AWARE alternative: a
     cluster with median mg >= 1.0 (cluster_amb_mg_min), T/B < 0.3, max(mg, pvm)
     >= 0.5 above every non-ASTROCYTE lineage median, and an astrocyte median
     < 1.0 (cluster_amb_ast_max; true astrocyte clusters sit at about 2) is also
     myeloid. Under v2 such clusters (clean microglia/PVM carrying ambient
     astrocyte RNA from the astrocyte-enriched sort) were dropped whole in
     MADRC-04/-08/-23 (899 nuclei in 4 clusters); the loss tracked each
     library's ambient level, not cell identity. The cell-level dominance rule
     (step 5) still compares against astrocytes, so astrocyte-microglia
     doublets inside a rescued cluster are still removed, and the heterotypic
     doublet simulation now also covers the rescued clusters. Clusters rescued
     this way carry 'myeloid_rule': 'ambient_ast' in the receipt.
  4. Heterotypic doublets: as many simulated doublets as myeloid-cluster
     nuclei, each the count sum of a random myeloid-cluster nucleus and a random
     non-myeloid QC nucleus, projected into the same PCA; score = fraction of
     simulated doublets among the 30 nearest neighbours; flagged if > 0.3.
  5. Cell rules inside myeloid clusters: max(mg, pvm) >= 0.5 (SEA-AD
     microglia mg 1st percentile 0.47); max(mg, pvm) above every non-myeloid
     lineage score; T and B scores <= the SEA-AD development-microglia 99.5th
     percentile; not a heterotypic doublet.
  PVMs are KEPT (as in SEA-AD). Their share is estimated in both cohorts with a
  logistic regression trained on SEA-AD development nuclei (Micro-PVM_1 vs the
  other five supertypes) over PVM_GENES; pvm_flag = probability > 0.5.
  Counts are NOT ambient-corrected (the SEA-AD store is not either); ambient
  astrocyte RNA is measured and reported per library (validation.json).

CAVEATS  Sorting (NeuN-/OLIG2- FANS vs unsorted), region (ITG vs MTG), Cell
Ranger version (GEO says 3.0.0 pre-mRNA; Wachter 2024 says v4 with the
GRCh38-2020-A pre-mRNA reference; the 36,601-ID axis matches 2020-A) and
depth differ from SEA-AD. Ambient profiles differ: astrocyte RNA in MADRC,
neuronal RNA in SEA-AD.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "3"
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
EXT = Path("/ml2/alz/runs/improve_20261109/ext")
OUT1 = EXT / "GSE268599"
REBUILT = Path("/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/rebuilt")
SELECTION_VERSION = "mg-select-v3"

MARKERS = {
    "mg": ["P2RY12", "CSF1R", "CX3CR1", "C3", "DOCK8", "APBB1IP", "PTPRC", "TYROBP", "C1QA", "SPI1"],
    "pvm": ["MRC1", "CD163", "F13A1", "LYVE1"],
    "ast": ["GFAP", "AQP4", "SLC1A2", "ADGRV1", "GJA1"],
    "olig": ["PLP1", "MBP", "MOBP", "ST18"],
    "opc": ["PDGFRA", "VCAN", "OLIG1"],
    "endo": ["CLDN5", "FLT1", "VWF"],
    "peri": ["PDGFRB", "RGS5", "DCN"],
    "neu": ["RBFOX3", "SNAP25", "SYT1", "GAD1", "SLC17A7"],
    "t": ["CD3E", "CD247", "SKAP1", "THEMIS"],
    "b": ["MS4A1", "CD79A", "BANK1"],
    "mono": ["FCN1", "S100A8", "VCAN", "CD14"],
}
NONMYELOID = ["ast", "olig", "opc", "endo", "peri", "neu", "t", "b"]
PVM_GENES = ["MRC1", "CD163", "F13A1", "LYVE1", "SIGLEC1", "STAB1", "COLEC12", "MS4A7", "CD14", "MS4A6A",
             "P2RY12", "CX3CR1", "TMEM119", "P2RY13", "GPR34", "SALL1"]
REPORT_GENES = ["P2RY12", "CX3CR1", "CSF1R", "TMEM119", "C3", "DOCK8", "APBB1IP", "PTPRC", "TYROBP", "SPI1",
                "APOE", "SPP1", "GPNMB", "CD74", "HLA-DRA", "MRC1", "CD163", "F13A1", "LYVE1",
                "GFAP", "AQP4", "SLC1A2", "PLP1", "MBP", "PDGFRA", "CLDN5", "PDGFRB", "SNAP25", "RBFOX3",
                "CD3E", "MS4A1", "FCN1", "MALAT1"]
PARAMS = {"min_genes": 1000, "max_mito": 0.05, "umi_mad": 3.0, "n_hvg": 2000, "n_pcs": 30, "k_clusters": 20,
          "cluster_mg_min": 0.75, "cluster_pvm_min": 1.0, "cluster_pvm_mg_min": 0.5, "cluster_margin": 0.5, "cluster_tb_max": 0.3,
          "cluster_amb_mg_min": 1.0, "cluster_amb_ast_max": 1.0, "cell_mg_min": 0.5,
          "tb_quantile": 99.5, "dbl_k": 30, "dbl_max": 0.3, "seed": 0, "pvm_prob": 0.5,
          "val_nonmyeloid_per_lib": 300}
SELECTION_SUMMARY = {"version": SELECTION_VERSION, "params": PARAMS, "markers": MARKERS,
                     "pvm_classifier_genes": PVM_GENES, "pvm_kept": True,
                     "author_annotation": "none openly available per cell (see module docstring)"}


def log_cp10k(M):
    M = M.tocsr().astype(np.float32)
    umi = np.asarray(M.sum(1)).ravel()
    L = sp.diags(1e4 / np.maximum(umi, 1)).dot(M).tocsr()
    L.data = np.log1p(L.data)
    return L, umi


def lineage_scores(L, sym):
    ix = {s: i for i, s in enumerate(sym)}
    return {k: np.asarray(L[:, [ix[g] for g in v if g in ix]].mean(1)).ravel().astype(np.float32)
            for k, v in MARKERS.items()}


# ----------------------------------------------------------------------------- SEA-AD reference
def seaad_reference(force=False):
    """Statistics of the 37 SEA-AD development donors' Micro-PVM nuclei (only those
    donors' counts are read): lineage scores, depth, supertype, PVM classifier,
    T/B thresholds, report-gene summaries and pseudobulk on the 36,601 axis."""
    out = OUT1 / "models" / "seaad_ref.npz"
    if out.exists() and not force:
        z = np.load(out, allow_pickle=False)
        return {k: z[k] for k in z.files}
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    dev = sorted(json.loads((STUDY / "folds.json").read_text())["development_donors"])
    meta = json.loads((REBUILT / "mg_meta.json").read_text())
    cst = meta["cols"].index("Supertype")
    sym = np.load(REBUILT / "counts/genes.npy", allow_pickle=False)
    ix = {s: i for i, s in enumerate(sym)}
    pj = [ix[g] for g in PVM_GENES]
    rj = [ix[g] for g in REPORT_GENES]
    rows = {k: [] for k in ["donor", "st", "umi", "ng", "mito"] + list(MARKERS)}
    Xp, Xr, Nz, pb = [], [], [], np.zeros(len(sym))
    mt = np.array([s.startswith("MT-") for s in sym])
    for d in dev:
        M = sp.load_npz(REBUILT / "counts" / f"{d}.npz").tocsr()
        names = np.load(REBUILT / "counts" / f"{d}_names.npy", allow_pickle=False)
        L, umi = log_cp10k(M)
        sc = lineage_scores(L, sym)
        for k in MARKERS:
            rows[k].append(sc[k])
        rows["donor"].append(np.full(len(names), d))
        rows["st"].append(np.array([meta["rows"][n][cst] for n in names]))
        rows["umi"].append(umi)
        rows["ng"].append(np.diff(M.indptr))
        rows["mito"].append(np.asarray(M[:, mt].sum(1)).ravel() / umi)
        Xp.append(L[:, pj].toarray())
        Xr.append(L[:, rj].toarray())
        Nz.append((M[:, rj] > 0).toarray())
        pb += np.asarray(M.sum(0)).ravel()
    R = {k: np.concatenate(v) for k, v in rows.items()}
    Xp = np.vstack(Xp)
    y = (R["st"] == "Micro-PVM_1").astype(int)
    # Donor-grouped 3-fold CV AUC, then a final fit on all development nuclei.
    donors = np.array(dev)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(donors))
    aucs = []
    for f in range(3):
        te = np.isin(R["donor"], donors[perm[f::3]])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xp[~te], y[~te])
        aucs.append(roc_auc_score(y[te], m.predict_proba(Xp[te])[:, 1]))
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Xp, y)
    prob = m.predict_proba(Xp)[:, 1]
    R.update({"pvm_coef": m.coef_[0], "pvm_intercept": m.intercept_, "pvm_cv_auc": np.array(aucs),
              "pvm_prob": prob.astype(np.float32),
              "t_thr": np.percentile(R["t"], PARAMS["tb_quantile"]), "b_thr": np.percentile(R["b"], PARAMS["tb_quantile"]),
              "report_mean": np.vstack(Xr).mean(0), "report_nonzero": np.vstack(Nz).mean(0),
              "report_genes": np.array(REPORT_GENES), "pvm_genes": np.array(PVM_GENES),
              "pseudobulk_counts": pb, "dev_donors": donors})
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **R)
    return R


def pvm_prob(L, sym, ref):
    ix = {s: i for i, s in enumerate(sym)}
    X = L[:, [ix[g] for g in PVM_GENES]].toarray()
    z = X @ ref["pvm_coef"] + ref["pvm_intercept"][0]
    return (1 / (1 + np.exp(-z))).astype(np.float32)


# ----------------------------------------------------------------------------- per-library selection
def _hvg(L, n):
    mu = np.asarray(L.mean(0)).ravel()
    var = np.asarray(L.multiply(L).mean(0)).ravel() - mu ** 2
    disp = np.where(mu > 0.02, var / (mu + 1e-3), -np.inf)
    return np.argsort(-disp)[:n]


def select_library(M, sym, barcodes, library, ref=None):
    """Return {'keep': bool[n_barcodes], 'cellmeta': dict of arrays for kept cells,
    'receipt': dict}. M: cells x 36,601 int CSR; sym: gene symbols on that axis."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    P = PARAMS
    ref = ref or seaad_reference()
    rng = np.random.default_rng(P["seed"])
    n0 = M.shape[0]
    umi0 = np.asarray(M.sum(1)).ravel()
    ng0 = np.diff(M.tocsr().indptr)
    mt = np.array([s.startswith("MT-") for s in sym])
    mito0 = np.asarray(M[:, mt].sum(1)).ravel() / umi0
    lu = np.log(umi0)
    med = np.median(lu)
    mad = np.median(np.abs(lu - med)) * 1.4826
    qc = (ng0 >= P["min_genes"]) & (mito0 < P["max_mito"]) & (lu <= med + P["umi_mad"] * mad)
    qi = np.where(qc)[0]
    Mq = M[qi].tocsr()
    L, umi = log_cp10k(Mq)
    sc = lineage_scores(L, sym)
    hv = _hvg(L, P["n_hvg"])
    X = L[:, hv].toarray()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.clip((X - mu) / sd, -10, 10)
    pca = PCA(P["n_pcs"], random_state=P["seed"]).fit(Z)
    Pc = pca.transform(Z)
    lab = KMeans(P["k_clusters"], n_init=3, random_state=P["seed"]).fit_predict(Pc)
    clus = []
    my_clusters = []
    for c in range(P["k_clusters"]):
        w = lab == c
        m = {k: float(np.median(sc[k][w])) for k in MARKERS}
        m["n"] = int(w.sum())
        other = max(m[k] for k in NONMYELOID)
        mye = max(m["mg"], m["pvm"])
        level = (m["mg"] >= P["cluster_mg_min"]
                 or (m["pvm"] >= P["cluster_pvm_min"] and m["mg"] >= P["cluster_pvm_mg_min"]))
        no_tb = m["t"] < P["cluster_tb_max"] and m["b"] < P["cluster_tb_max"]
        strict = bool(level and mye - other >= P["cluster_margin"] and no_tb)
        other_na = max(m[k] for k in NONMYELOID if k != "ast")
        ambient = bool(not strict and no_tb and m["mg"] >= P["cluster_amb_mg_min"]
                       and mye - other_na >= P["cluster_margin"] and m["ast"] < P["cluster_amb_ast_max"])
        m["myeloid"] = strict or ambient
        m["myeloid_rule"] = "strict" if strict else ("ambient_ast" if ambient else None)
        clus.append(m)
        if m["myeloid"]:
            my_clusters.append(c)
    my = np.isin(lab, my_clusters)
    # Heterotypic doublet simulation.
    myi, oth = np.where(my)[0], np.where(~my)[0]
    dbl = np.zeros(len(qi), dtype=np.float32)
    if len(myi) and len(oth):
        ns = len(myi)
        D = Mq[rng.choice(myi, ns)] + Mq[rng.choice(oth, ns)]
        LD, _ = log_cp10k(D)
        ZD = np.clip((LD[:, hv].toarray() - mu) / sd, -10, 10)
        PD = pca.transform(ZD)
        nn = NearestNeighbors(n_neighbors=P["dbl_k"]).fit(np.vstack([Pc, PD]))
        _, I = nn.kneighbors(Pc[myi])
        dbl[myi] = (I >= len(Pc)).mean(1)
    maxoth = np.max(np.vstack([sc[k] for k in NONMYELOID]), 0)
    mye = np.maximum(sc["mg"], sc["pvm"])
    r_mg = mye >= P["cell_mg_min"]
    r_dom = mye > maxoth
    r_tb = (sc["t"] <= ref["t_thr"]) & (sc["b"] <= ref["b_thr"])
    r_dbl = dbl <= P["dbl_max"]
    sel = my & r_mg & r_dom & r_tb & r_dbl
    keep = np.zeros(n0, dtype=bool)
    keep[qi[sel]] = True
    pp = pvm_prob(L, sym, ref)
    # Ambient astrocyte RNA: astrocyte-marker CP10k in kept cells / in astrocyte clusters.
    ix = {s: i for i, s in enumerate(sym)}
    aj = [ix[g] for g in MARKERS["ast"]]
    cp = np.asarray(Mq[:, aj].sum(1)).ravel() / umi * 1e4
    ast_cl = [c for c, m in enumerate(clus) if m["ast"] == max(x["ast"] for x in clus)]
    astro = np.isin(lab, [c for c, m in enumerate(clus) if m["ast"] >= 1.5 and not m["myeloid"]])
    amb = float(np.mean(cp[sel]) / np.mean(cp[astro])) if astro.any() and sel.any() else None
    # Validation sample (transient): panel counts of rejected myeloid-cluster nuclei
    # and a random subsample of non-myeloid QC nuclei.
    rej = np.where(my & ~sel)[0]
    nonm = rng.choice(oth, min(len(oth), P["val_nonmyeloid_per_lib"]), replace=False) if len(oth) else oth
    vdir = OUT1 / "valsample"
    vdir.mkdir(parents=True, exist_ok=True)
    vi = np.concatenate([np.where(sel)[0], rej, nonm])
    vkind = np.concatenate([np.zeros(sel.sum(), int), np.ones(len(rej), int), np.full(len(nonm), 2)])
    sp.save_npz(vdir / f"{library}.npz", Mq[vi].astype(np.int32).tocsr())
    np.savez_compressed(vdir / f"{library}_meta.npz", kind=vkind, umi=umi[vi], lab=lab[vi],
                        **{f"s_{k}": sc[k][vi] for k in MARKERS}, dbl=dbl[vi])
    cellmeta = {"library": np.full(sel.sum(), library), "umi": umi[sel].astype(np.float32),
                "n_genes": np.diff(Mq.indptr)[sel].astype(np.int32), "mito": mito0[qi][sel].astype(np.float32),
                "cluster": lab[sel].astype(np.int16), "doublet_score": dbl[sel],
                "pvm_prob": pp[sel], "pvm_flag": pp[sel] > P["pvm_prob"]}
    cellmeta.update({f"score_{k}": sc[k][sel] for k in MARKERS})
    receipt = {"library": library, "n_barcodes": int(n0), "n_qc_pass": int(qc.sum()),
               "qc_fail": {"genes_lt_min": int((ng0 < P["min_genes"]).sum()),
                           "mito_ge_max": int((mito0 >= P["max_mito"]).sum()),
                           "umi_gt_median_plus_3mad": int((lu > med + P["umi_mad"] * mad).sum())},
               "barcode_umi_min": int(umi0.min()), "clusters": clus, "myeloid_clusters": my_clusters,
               "n_myeloid_cluster": int(my.sum()),
               "rejected": {"mg_lt_min": int((my & ~r_mg).sum()), "other_lineage_dominates": int((my & ~r_dom).sum()),
                            "t_or_b": int((my & ~r_tb).sum()), "hetero_doublet": int((my & ~r_dbl).sum())},
               "n_kept": int(sel.sum()), "frac_kept_of_barcodes": float(sel.sum() / n0),
               "frac_myeloid_cluster_of_qc": float(my.mean()),
               "median_umis_kept": float(np.median(umi[sel])) if sel.any() else None,
               "median_genes_kept": float(np.median(np.diff(Mq.indptr)[sel])) if sel.any() else None,
               "pvm_flag_frac": float((pp[sel] > P["pvm_prob"]).mean()) if sel.any() else None,
               "pvm_prob_mean": float(pp[sel].mean()) if sel.any() else None,
               "ambient_astro_ratio": amb, "n_astro_cluster_cells": int(astro.sum()),
               "mono_score_median_kept": float(np.median(sc["mono"][sel])) if sel.any() else None}
    return {"keep": keep, "cellmeta": cellmeta, "receipt": receipt}


# ----------------------------------------------------------------------------- validation
def validate():
    """Write ext/GSE268599/validation.json. Needs the transient valsample/ written by
    select_library (panel-free counts of rejected myeloid-cluster nuclei and <= 300
    random non-myeloid QC nuclei per library); valsample/ was deleted after the
    2026-11-11 validation run, so rerunning this requires rerunning the selection."""
    from sklearn.linear_model import LogisticRegression
    ref = seaad_reference()
    sym = np.load(REBUILT / "counts/genes.npy", allow_pickle=False)
    ix = {s: i for i, s in enumerate(sym)}
    doc = json.loads((OUT1 / "donors.json").read_text())
    donors = doc["donors"]
    rj = [ix[g] for g in REPORT_GENES]
    # Selected cells summary.
    Xr, Nz, umi, ng, scs, pvmp, pb = [], [], [], [], {k: [] for k in MARKERS}, [], np.zeros(len(sym))
    per_donor = {}
    for d, e in donors.items():
        M = sp.load_npz(OUT1 / e["counts_file"]).tocsr()
        cm = np.load(OUT1 / e["cellmeta_file"], allow_pickle=False)
        L, u = log_cp10k(M)
        Xr.append(np.asarray(L[:, rj].mean(0)).ravel() * M.shape[0])
        Nz.append(np.asarray((M[:, rj] > 0).sum(0)).ravel())
        umi.append(u)
        ng.append(np.diff(M.indptr))
        for k in MARKERS:
            scs[k].append(cm[f"score_{k}"])
        pvmp.append(cm["pvm_prob"])
        pb += np.asarray(M.sum(0)).ravel()
        per_donor[d] = {"ADNC": e["ADNC"], "cells": e["cells_kept"], "median_umis": e["median_umis_per_cell"],
                        "median_genes": e["median_genes_per_cell"],
                        "pvm_flag_frac": float((cm["pvm_prob"] > PARAMS["pvm_prob"]).mean()),
                        "libraries": [{k: l[k] for k in ["library", "n_barcodes", "n_qc_pass", "n_myeloid_cluster",
                                                          "n_kept", "frac_kept_of_barcodes", "frac_myeloid_cluster_of_qc",
                                                          "ambient_astro_ratio", "rejected"]}
                                      for l in e["selection_receipt"]["libraries"]]}
    n = sum(len(x) for x in umi)
    umi, ng = np.concatenate(umi), np.concatenate(ng)
    scs = {k: np.concatenate(v) for k, v in scs.items()}
    pvmp = np.concatenate(pvmp)
    ext_mean = np.sum(Xr, 0) / n
    ext_nz = np.sum(Nz, 0) / n
    markers = {g: {"madrc_mean_log1p_cp10k": round(float(ext_mean[i]), 3), "seaad_mean_log1p_cp10k": round(float(ref["report_mean"][i]), 3),
                   "madrc_frac_nonzero": round(float(ext_nz[i]), 3), "seaad_frac_nonzero": round(float(ref["report_nonzero"][i]), 3)}
               for i, g in enumerate(REPORT_GENES)}
    q = lambda a: [round(float(x), 3) for x in np.percentile(a, [5, 25, 50, 75, 95])]
    scores = {k: {"madrc_q5_25_50_75_95": q(scs[k]), "seaad_q5_25_50_75_95": q(ref[k])} for k in MARKERS}
    # Rules applied to SEA-AD development microglia (sensitivity of the cell-level rules).
    maxoth = np.max(np.vstack([ref[k] for k in NONMYELOID]), 0)
    rmye = np.maximum(ref["mg"], ref["pvm"])
    s_rules = {"mye_ge_min": float((rmye >= PARAMS["cell_mg_min"]).mean()),
               "mye_dominates": float((rmye > maxoth).mean()),
               "t_b_within": float(((ref["t"] <= ref["t_thr"]) & (ref["b"] <= ref["b_thr"])).mean()),
               "genes_ge_min": float((ref["ng"] >= PARAMS["min_genes"]).mean()),
               "mito_lt_max": float((ref["mito"] < PARAMS["max_mito"]).mean())}
    s_rules["all_cell_rules"] = float(((rmye >= PARAMS["cell_mg_min"]) & (rmye > maxoth) & (ref["t"] <= ref["t_thr"])
                                       & (ref["b"] <= ref["b_thr"]) & (ref["ng"] >= PARAMS["min_genes"]) & (ref["mito"] < PARAMS["max_mito"])).mean())
    s_rules["by_supertype_all_rules"] = {}
    ok = (rmye >= PARAMS["cell_mg_min"]) & (rmye > maxoth) & (ref["ng"] >= PARAMS["min_genes"])
    for st in np.unique(ref["st"]):
        s_rules["by_supertype_all_rules"][str(st)] = round(float(ok[ref["st"] == st].mean()), 3)
    # PVM fractions.
    pvm = {"seaad_micro_pvm_1_frac": float((ref["st"] == "Micro-PVM_1").mean()),
           "seaad_classifier_flag_frac": float((ref["pvm_prob"] > 0.5).mean()),
           "seaad_classifier_mean_prob": float(ref["pvm_prob"].mean()),
           "classifier_donor_cv_auc": [round(float(a), 4) for a in ref["pvm_cv_auc"]],
           "madrc_classifier_flag_frac": float((pvmp > 0.5).mean()), "madrc_classifier_mean_prob": float(pvmp.mean()),
           "madrc_pvm_marker_score_ge_1.4_frac": float((scs["pvm"] >= 1.4).mean()),
           "seaad_pvm_marker_score_ge_1.4_frac": float((ref["pvm"] >= 1.4).mean())}
    # Pseudobulk agreement on the three hvg2000 panels' union.
    hv = set()
    for k in range(3):
        hv |= set(np.load(f"/ml2/alz/runs/improve_20261109/caches/hvg2000/fold{k}/space.npz")["genes"].tolist())
    hj = [ix[g] for g in sorted(hv)]
    lpb = lambda v: np.log1p(v / v.sum() * 1e6)
    pb_corr = float(np.corrcoef(lpb(pb)[hj], lpb(ref["pseudobulk_counts"])[hj])[0, 1])
    # Leave-library-out classifier: SEA-AD dev microglia (+) vs MADRC non-myeloid nuclei (-).
    vdir = OUT1 / "valsample"
    libs = sorted(p.stem for p in vdir.glob("GSM*.npz") if not p.stem.endswith("_meta"))
    feat = sorted(set(hj) | {ix[g] for v in MARKERS.values() for g in v})
    seaad_X = []
    dev = ref["dev_donors"]
    rng = np.random.default_rng(0)
    for d in dev:
        M = sp.load_npz(REBUILT / "counts" / f"{d}.npz").tocsr()
        L, _ = log_cp10k(M)
        take = rng.choice(M.shape[0], min(M.shape[0], 150), replace=False)
        seaad_X.append(L[take][:, feat].toarray())
    seaad_X = np.vstack(seaad_X)
    V = {}
    for g in libs:
        M = sp.load_npz(vdir / f"{g}.npz").tocsr()
        m = np.load(vdir / f"{g}_meta.npz")
        L, _ = log_cp10k(M)
        V[g] = (L[:, feat].toarray(), m["kind"])
    agree = {}
    folds = [libs[i::4] for i in range(4)]
    for fl in folds:
        tr = [g for g in libs if g not in fl]
        Xn = np.vstack([V[g][0][V[g][1] == 2] for g in tr])
        Xtr = np.vstack([seaad_X, Xn])
        ytr = np.r_[np.ones(len(seaad_X)), np.zeros(len(Xn))]
        clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        for g in fl:
            X, k = V[g]
            p = clf.predict_proba(X)[:, 1]
            agree[g] = {"selected_prob_gt_0.5": round(float((p[k == 0] > .5).mean()), 4),
                        "rejected_myeloid_prob_gt_0.5": round(float((p[k == 1] > .5).mean()), 4) if (k == 1).any() else None,
                        "nonmyeloid_sample_prob_gt_0.5": round(float((p[k == 2] > .5).mean()), 4)}
    a_sel = np.mean([v["selected_prob_gt_0.5"] for v in agree.values()])
    a_non = np.mean([v["nonmyeloid_sample_prob_gt_0.5"] for v in agree.values()])
    # Library-level fractions.
    libsum = [l for d in per_donor.values() for l in d["libraries"]]
    fr = np.array([l["frac_kept_of_barcodes"] for l in libsum])
    frm = np.array([l["frac_myeloid_cluster_of_qc"] for l in libsum])
    amb = np.array([l["ambient_astro_ratio"] for l in libsum if l["ambient_astro_ratio"] is not None])
    nrej = {k: int(sum(l["rejected"][k] for l in libsum)) for k in libsum[0]["rejected"]}
    stage = {}
    for s in ["Not AD", "Low", "Intermediate", "High"]:
        ds = {d: v["cells"] for d, v in per_donor.items() if v["ADNC"] == s}
        stage[s] = {"donors": len(ds), "cells": int(sum(ds.values())), "per_donor": ds}
    out = {"version": SELECTION_VERSION, "script": "r4/ext_select_microglia.py validate",
           "n_cells": int(n), "n_donors": len(per_donor),
           "depth": {"madrc_median_umis": float(np.median(umi)), "madrc_median_genes": float(np.median(ng)),
                     "seaad_dev_median_umis": float(np.median(ref["umi"])), "seaad_dev_median_genes": float(np.median(ref["ng"])),
                     "madrc_umi_q5_25_50_75_95": q(umi), "seaad_umi_q5_25_50_75_95": q(ref["umi"])},
           "fraction_kept_per_library": {"min": float(fr.min()), "median": float(np.median(fr)), "max": float(fr.max())},
           "fraction_myeloid_clusters_of_qc_nuclei": {"min": float(frm.min()), "median": float(np.median(frm)), "max": float(frm.max()),
                                                     "wachter2024_itg_myeloid_fraction_of_all_nuclei": 0.28},
           "rejected_in_myeloid_clusters_total": nrej,
           "ambient_astro_ratio_per_library": {"min": float(amb.min()), "median": float(np.median(amb)), "max": float(amb.max()),
                                                "meaning": "astrocyte-marker CP10k in kept microglia / in astrocyte clusters"},
           "lineage_scores": scores, "marker_genes": markers, "seaad_rule_pass_rates": s_rules, "pvm": pvm,
           "pseudobulk_log_cpm_corr_hvg_union": {"madrc_vs_seaad_dev": pb_corr, "n_genes": len(hj)},
           "leave_library_out_classifier": {"features": f"{len(feat)} genes (hvg2000 union + markers), log1p CP10k",
                                            "positives": "SEA-AD dev microglia (150 per donor)",
                                            "negatives": "MADRC non-myeloid QC nuclei of other libraries (<=300 per library)",
                                            "mean_selected_called_microglia": float(a_sel),
                                            "mean_nonmyeloid_called_microglia": float(a_non), "per_library": agree},
           "stage_table": stage, "per_donor": per_donor}
    tmp = OUT1 / "validation.json.partial"
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    tmp.replace(OUT1 / "validation.json")
    print(json.dumps({k: out[k] for k in ["n_cells", "depth", "fraction_kept_per_library", "pvm",
                                          "pseudobulk_log_cpm_corr_hvg_union", "seaad_rule_pass_rates"]}, indent=1))
    print("classifier", a_sel, a_non)


# ----------------------------------------------------------------------------- GSE264648
def write_gse264648(M, genes, cells, cols, rownames, got, out):
    """Write ext/GSE264648/ (MTG = 'mTemp' microglia only) on the 36,601 SEA-AD symbol
    axis; genes not in the source object are zero and flagged in genes_present.npy."""
    sym = np.load(REBUILT / "counts/genes.npy", allow_pickle=False)
    ix = {s: i for i, s in enumerate(sym)}
    src = np.array([ix.get(g, -1) for g in genes])
    present = np.zeros(len(sym), bool)
    present[src[src >= 0]] = True
    if not np.all(M.data == np.round(M.data)) or (M.data < 0).any():
        raise ValueError("GSE264648 counts are not non-negative integers")
    T = sp.csr_matrix((np.ones((src >= 0).sum()), (np.where(src >= 0)[0], src[src >= 0])), shape=(len(genes), len(sym)))
    lost_umis = float(M[:, np.where(src < 0)[0]].sum() / M.sum())
    Y = (M @ T).tocsr()
    Y.data = np.round(Y.data)
    Y = Y.astype(np.int32)
    Y.sort_indices()
    bc = np.asarray(cols["barcode"]).astype(str)
    if not np.array_equal(bc, cells):
        order = {c: i for i, c in enumerate(bc)}
        idx = np.array([order[c] for c in cells])
        cols = {k: np.asarray(v)[idx] for k, v in cols.items()}
    region = np.asarray(cols["brain_region"]).astype(str)
    indiv = np.asarray(cols["individual"]).astype(str)
    (out / "counts").mkdir(parents=True, exist_ok=True)
    np.save(out / "counts" / "genes.npy", sym)
    np.save(out / "counts" / "genes_present.npy", present)
    ref = seaad_reference()
    donors = {}
    for ind in sorted(set(indiv[region == "mTemp"])):
        w = np.where((indiv == ind) & (region == "mTemp"))[0]
        name = "IMP-" + ind.replace("/", "_")
        X = Y[w]
        sp.save_npz(out / "counts" / f"{name}.npz", X)
        np.save(out / "counts" / f"{name}_names.npy", cells[w].astype("<U64"))
        L, u = log_cp10k(X)
        sc = lineage_scores(L, sym)
        first = lambda k: str(np.asarray(cols[k])[w][0])
        donors[name] = {"cohort": "Imperial (GSE264648)", "cohort_code": 2, "individual": ind, "region": "MTG (mTemp)",
                        "diagnosis": first("diagnosis"), "braak": int(float(first("braak"))),
                        "aBeta_pct_area_region": float(first("aBeta")), "pTau_region": float(first("pTau")),
                        "age": float(first("age")), "sex": first("sex"), "PMI": float(first("PMI")),
                        "RIN": float(first("RIN")), "brain_bank": first("brain_bank"),
                        "ADNC": None, "ADNC_note": "not derivable: no Thal/CERAD/A score in open metadata",
                        "cells_kept": int(len(w)), "median_umis_per_cell": float(np.median(u)),
                        "median_genes_per_cell": float(np.median(np.diff(X.indptr))),
                        "author_subclusters": {k: int(v) for k, v in zip(*np.unique(np.asarray(cols["subclusters_manual"])[w].astype(str), return_counts=True))},
                        "allan_celltype": {k: int(v) for k, v in zip(*np.unique(np.asarray(cols["allan_celltype"])[w].astype(str), return_counts=True))},
                        "median_scores": {k: round(float(np.median(v)), 3) for k, v in sc.items()},
                        "pvm_flag_frac": float((pvm_prob(L, sym, ref) > 0.5).mean())}
    doc = {"dataset": "GSE264648 Imperial microglia object, MTG ('mTemp') cells only; cohort-offset check ONLY",
           "source": got, "selection": "author annotation celltype_global == 'Micro' for every cell in the object "
                                         "(is_singlet == 1, is_empty_drop == 0 for all); no re-selection",
           "gene_axis": {"source_genes": int(len(genes)), "mapped_to_36601_by_symbol": int((src >= 0).sum()),
                         "unmapped_source_genes": int((src < 0).sum()), "unmapped_examples": genes[src < 0][:20].tolist(),
                         "fraction_of_umis_on_unmapped_genes": lost_umis,
                         "note": "source object carries gene symbols only (Cell Ranger 3.1 reference, not 2020-A); genes "
                                 "absent from the object are zero columns, flagged False in counts/genes_present.npy"},
           "cells_all_regions": int(Y.shape[0]), "regions": {k: int(v) for k, v in zip(*np.unique(region, return_counts=True))},
           "donors": donors}
    (out / "donors.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({d: (v["cells_kept"], v["diagnosis"], v["braak"], v["median_umis_per_cell"]) for d, v in donors.items()}))
    print(json.dumps(doc["gene_axis"])[:600])


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "reference":
        R = seaad_reference(force=True)
        print("cv auc", R["pvm_cv_auc"], "t_thr", R["t_thr"], "b_thr", R["b_thr"], "n", len(R["umi"]))
    elif cmd == "validate":
        validate()
    else:
        print(__doc__)
