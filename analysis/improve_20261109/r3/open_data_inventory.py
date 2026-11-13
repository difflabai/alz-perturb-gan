"""R8 open-data inventory helper (round 3, research only; no training, no counts read).

Reads ONLY public metadata of open human brain single-nucleus/single-cell datasets:

  peek   URL [budget_MB] [donor_col] [celltype_col] [celltype_regex] [only_cols,comma,list]
         HTTPS range reads of a remote AnnData .h5ad. Reads `obs` only (never X, raw or
         layers) under a hard byte budget (default 12 MB), and prints JSON with: file size,
         obs column names, whether raw counts exist (`raw/X` or `layers/*`), var length,
         the number of microglia cells per donor (cells whose cell-type label matches the
         regex, default 'microglia|microglial'), and every obs column that is constant within
         each donor (the donor-level metadata, e.g. disease, Braak, CERAD, age, sex, APOE).
         Columns with more than 64 distinct values per file are skipped unless their name
         looks like a pathology field.
  cxg    PATH_TO_datasets.json
         Summarises the CELLxGENE Discover curation index (all public datasets) for human
         datasets that contain microglia: cell counts, donors, disease labels, assay, size.

  build  Assembles r3/open_data_inventory.json from the small tables saved under
         runs/improve_20261109/smoke_r3/data_inventory/ (Gerrits 2021 Table S1, Serrano-Pozo 2024
         Table S1, GSE264648 and GSE174367 per-donor microglia tallies, obs peeks) plus curated,
         source-quoted records. ADNC is derived with adnc(A, B, C) (NIA-AA 2012 table).

peek/cxg outputs go to stdout; the caller saves them under
runs/improve_20261109/smoke_r3/data_inventory/. Uses the RangeFile/col readers of
../data_expansion/peek_h5ad_obs.py (read-only import). Run with python3 -I -B and keep the total
transferred bytes under the study's 50 MB limit (each result reports bytes_read).
"""
import io
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_expansion"))
from peek_h5ad_obs import RangeFile, col  # noqa: E402

PATHO = re.compile(r"braak|cerad|thal|adnc|amyloid|tau|plaque|tangle|apoe|diag|disease|"
                   r"cogn|dement|niareagan|age|sex|abeta|path|stage|status|group|cond|pmi|"
                   r"region|tissue|assay|chem|batch|cohort|project", re.I)


def _str(v):
    return v.astype(str)


def peek(url, budget_mb=12.0, donor_col="donor_id", ct_col="cell_type",
         ct_regex=r"microglia|microglial", only_cols=None):
    f = RangeFile(url, int(budget_mb * 1e6))
    res = {"url": url, "file_bytes": f.size}
    with h5py.File(io.BufferedReader(f, buffer_size=1 << 16), "r") as h:
        obs = h["obs"]
        res["obs_columns"] = sorted(obs.keys())
        res["raw_X"] = "raw" in h and "X" in h["raw"]
        res["layers"] = sorted(h["layers"].keys()) if "layers" in h else []
        try:
            vk = h["var"].attrs.get("_index", "_index")
            res["n_var"] = int(h["var"][vk].shape[0])
            if "raw" in h and "var" in h["raw"]:
                rk = h["raw"]["var"].attrs.get("_index", "_index")
                res["n_raw_var"] = int(h["raw"]["var"][rk].shape[0])
        except Exception as e:  # pragma: no cover - metadata only
            res["n_var_error"] = repr(e)
        donors = _str(col(obs, donor_col))
        res["n_obs"] = int(len(donors))
        ct = _str(col(obs, ct_col)) if ct_col in obs else None
        keep = (np.array([bool(re.search(ct_regex, x, re.I)) for x in ct])
                if ct is not None else np.ones(len(donors), bool))
        if ct is not None:
            u, n = np.unique(ct, return_counts=True)
            res["cell_type_counts"] = {a: int(b) for a, b in zip(u, n)
                                       if re.search(ct_regex + "|macrophage|myeloid|immune", a, re.I)}
        res["n_microglia"] = int(keep.sum())
        per = {}
        for d in np.unique(donors):
            per[d] = {"n_cells": 0, "n_microglia": 0}
        ud, nd = np.unique(donors, return_counts=True)
        for d, n in zip(ud, nd):
            per[d]["n_cells"] = int(n)
        ud, nd = np.unique(donors[keep], return_counts=True)
        for d, n in zip(ud, nd):
            per[d]["n_microglia"] = int(n)
        skipped = []
        for c in res["obs_columns"]:
            if c in (donor_col, ct_col) or c.startswith("_"):
                continue
            if only_cols is not None and c not in only_cols:
                skipped.append(c)
                continue
            g = obs[c]
            is_cat = isinstance(g, h5py.Group)
            if not is_cat and not PATHO.search(c):
                skipped.append(c)
                continue
            if is_cat and "categories" in g and len(g["categories"]) > 64 and not PATHO.search(c):
                skipped.append(c)
                continue
            try:
                v = _str(col(obs, c))
            except Exception as e:
                skipped.append(f"{c}:{e!r}"[:80])
                continue
            vals = {}
            const = True
            for d, x in zip(donors, v):
                if vals.setdefault(d, x) != x:
                    const = False
                    break
            if const:
                for d, x in vals.items():
                    per[d][c] = x
        res["skipped_columns"] = skipped
        res["donor_level_columns"] = sorted({k for p in per.values() for k in p} - {"n_cells", "n_microglia"})
    res["donors"] = per
    res["n_donors"] = len(per)
    res["bytes_read"] = f.used
    return res


def cxg(path):
    d = json.load(open(path))
    out = []
    for x in d:
        org = [o["label"] for o in x.get("organism", [])]
        ct = [c["label"] for c in x.get("cell_type", [])]
        if "Homo sapiens" not in org or not any(re.search("microglia", c, re.I) for c in ct):
            continue
        out.append({
            "collection_id": x["collection_id"], "collection_name": x["collection_name"],
            "doi": x.get("collection_doi"), "dataset_id": x["dataset_id"], "title": x["title"],
            "cells": x.get("cell_count"), "n_donors": len(x.get("donor_id", [])),
            "disease": [t["label"] for t in x.get("disease", [])],
            "tissue": [t["label"] for t in x.get("tissue", [])],
            "assay": [a["label"] for a in x.get("assay", [])],
            "h5ad_url": next((a["url"] for a in x.get("assets", []) if a["filetype"] == "H5AD"), None),
            "h5ad_bytes": next((a["filesize"] for a in x.get("assets", []) if a["filetype"] == "H5AD"), None),
            "is_primary_data": x.get("is_primary_data"),
        })
    return out


# --------------------------------------------------------------------------------------------
# build: assemble r3/open_data_inventory.json from the small tables saved under DATA and the
# curated records below (every record says which facts were verified and how).
# --------------------------------------------------------------------------------------------
DATA = Path("/ml2/alz/runs/improve_20261109/smoke_r3/data_inventory")
OUT = Path(__file__).resolve().parent / "open_data_inventory.json"
ADNC_LEVELS = ["Not AD", "Low", "Intermediate", "High"]


def adnc(a, b, c):
    """NIA-AA 2012 (Montine et al.) ABC -> ADNC. a = Thal-derived A0-3, b = Braak band B0-3,
    c = CERAD C0-3. A0 -> Not AD; B0/B1 with any A>0 -> Low; A1 with C0/C1 -> Low;
    A3 + C2/C3 + B3 -> High; everything else -> Intermediate. Checked against the SEA-AD
    (Braak band, CERAD) x ADNC cross-tab in DATA_EXPANSION.md section 4.4."""
    if a is None or b is None:
        return None
    if a == 0:
        return "Not AD"
    if b <= 1:
        return "Low"
    if c is None:
        return "Low" if a == 1 else None
    if a == 1 and c <= 1:
        return "Low"
    if a == 1:
        return "Intermediate"
    if a == 3 and c >= 2 and b == 3:
        return "High"
    return "Intermediate"


def _abc_range(s):
    """Parse strings like 'A2-3B2C2', 'A1B0C?', 'A3B2-3C1-2' into the set of possible ADNC levels."""
    m = re.fullmatch(r"A([0-3])(?:-([0-3]))?B([0-3])(?:-([0-3]))?C([0-3?])(?:-([0-3]))?", s)
    if not m:
        return None
    a = range(int(m[1]), int(m[2] or m[1]) + 1)
    b = range(int(m[3]), int(m[4] or m[3]) + 1)
    c = [None] if m[5] == "?" else range(int(m[5]), int(m[6] or m[5]) + 1)
    return sorted({adnc(x, y, z) for x in a for y in b for z in c}, key=lambda v: ADNC_LEVELS.index(v) if v else 9)


def _serrano_pozo():
    import csv
    rows = list(csv.DictReader(open(DATA / "papers/serrano_pozo2024_tableS1_donors.csv")))
    out = {"donors": len(rows), "adnc": {k: 0 for k in ADNC_LEVELS}, "adnc_with_ITG": {k: 0 for k in ADNC_LEVELS},
           "not_ad_donors": []}
    for r in rows:
        lv = adnc(int(r["A"]), int(r["B"]), int(r["C"]))
        out["adnc"][lv] += 1
        if r["ITG"] == "yes":
            out["adnc_with_ITG"][lv] += 1
        if lv == "Not AD":
            out["not_ad_donors"].append({k: r[k] for k in ("Donor.ID", "A", "B", "C", "Braak", "CERAD", "APOE", "Sex", "Age", "EC", "ITG", "PFC", "V2", "V1")})
    return out


def _gerrits():
    import csv
    rows = list(csv.DictReader(open(DATA / "papers/gerrits2021_tableS1_donors.csv")))
    donors = {}
    for r in rows:
        d = donors.setdefault(r["donor #"], {"group": r["experimental group"], "age": round(float(r["age"]), 1),
                                             "sex": r["gender"], "apoe": r["apoe_genotype"], "stage": r["Braak or ABC score"],
                                             "microglia": {}})
        d["microglia"][r["region"]] = int(r["# of microglia"])
    for d in donors.values():
        rng = _abc_range(d["stage"])
        d["adnc_possible"] = rng if rng else "Braak tau stage only (no A/C): ADNC not derivable"
        d["microglia_total"] = sum(d["microglia"].values())
    return donors


def _simple_csv(path):
    import csv
    return list(csv.DictReader(open(path)))


def _peek_summary(fname, keys):
    r = json.load(open(DATA / "cxg" / fname))
    out = []
    for d, p in r["donors"].items():
        out.append({"donor": d, "n_microglia": p["n_microglia"], **{k: p.get(k) for k in keys}})
    return {"bytes_read": r["bytes_read"], "n_obs": r["n_obs"], "n_var": r.get("n_var"), "raw_X": r["raw_X"],
            "n_microglia": r["n_microglia"], "donors": out}


def build():
    sp = _serrano_pozo()
    ge = _gerrits()
    imp = _simple_csv(DATA / "geo/GSE264648_microglia_per_donor_region.csv")
    imp_donors = {}
    for r in imp:
        d = imp_donors.setdefault(r["individual"], {"diagnosis": r["diagnosis"], "braak": int(r["braak"]), "age": int(r["age"]),
                                                    "sex": r["sex"], "microglia": {}, "aBeta_pct_area": {}})
        d["microglia"][r["brain_region"]] = int(r["n_microglia"])
        d["aBeta_pct_area"][r["brain_region"]] = float(r["aBeta"])
    mor = _simple_csv(DATA / "geo/GSE174367_per_donor.csv")
    leng_ec = _peek_summary("peek_9f1049ac.json", ["BraakStage", "development_stage", "sex", "disease"])
    leng_sfg = _peek_summary("peek_bdacc907.json", ["BraakStage", "development_stage", "sex", "disease"])
    csf1r = _peek_summary("peek_fe2eecbc.json", ["Braak.stage", "Amyloid", "APOE", "development_stage", "sex", "disease"])
    vasc = _peek_summary("peek_203025fe.json", ["disease", "development_stage", "sex"])
    rex = _peek_summary("peek_ac0c6561.json", ["disease", "development_stage", "sex"])
    olah = _peek_summary("peek_olah.json", ["disease", "tissue", "sex"])

    def tally(rows, key):
        t = {}
        for r in rows:
            t[str(r[key])] = t.get(str(r[key]), 0) + 1
        return t

    GEO_TERMS = ("GEO has no access control or login; NCBI's disclaimer (https://www.ncbi.nlm.nih.gov/geo/info/"
                 "disclaimer.html) says NCBI 'cannot provide comment or unrestricted permission concerning the use, "
                 "copying, or distribution of the information contained in GEO' because some submitters may claim "
                 "patent, copyright or other IP rights (older wording: 'NCBI places no restrictions on the use or "
                 "distribution of the GEO data'). No licence file is attached to these series; treat as open with "
                 "attribution, no IP claim seen.")
    CXG_TERMS = ("CELLxGENE Discover: 'anyone will be able to access it subject to a CC-BY 4.0 license' "
                 "(https://cellxgene.cziscience.com/docs/032__Contribute%20and%20Publish%20Data). No account.")

    datasets = [
        {
            "id": "serrano_pozo_wachter_2024_madrc",
            "rank": 1,
            "papers": ["Serrano-Pozo et al. Nat Neurosci 2024 doi:10.1038/s41593-024-01791-4 (astrocytes)",
                       "Wachter et al. Acta Neuropathol 2024 147:65 doi:10.1007/s00401-024-02704-2 (myeloid; CC BY 4.0 article)"],
            "accession": "GEO GSE268599 (146 libraries, per-library matrix.mtx/features/barcodes; RAW.tar 10.9 GB); SRA PRJNA916657",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268599",
            "access": "open (GEO, no login). SRA biosample attribute 'consent: public' on all 146 samples [verified in series matrix].",
            "access_quote": "Wachter 2024: 'Raw data of this study are available via SRA accession PRJNA916657 ... Processed data can be browsed at https://ad-progression-atlas.partners.org.' Serrano-Pozo 2024: 'The snRNA-seq dataset generated for the study will be made available at Gene Expression Omnibus (GEO: GSE268599)'.",
            "terms": GEO_TERMS,
            "region": "5 regions per donor: entorhinal cortex (EC), inferior temporal gyrus (ITG, BA20), dorsolateral PFC (BA46), V2, V1. ITG is the temporal neocortex next to MTG.",
            "technology": "FANS-sorted NeuN-negative OLIG2-negative nuclei (neuron and oligodendrocyte depleted: glia-enriched, like Gerrits 2021); 10x Chromium 3' v3; NovaSeq/HiSeq; Cell Ranger (paper: 3.0.0 pre-mRNA).",
            "gene_axis": "VERIFIED: features.tsv of GSM8295013 has exactly the same 36,601 Ensembl IDs as the 10x GRCh38-2020-A reference (identical set to SEA-AD).",
            "donors": {"total": 32, "adnc_from_published_ABC": sp["adnc"], "adnc_donors_with_ITG": sp["adnc_with_ITG"],
                       "not_ad_donors": sp["not_ad_donors"]},
            "per_donor_variables_open": "Supplementary Table 1 of Serrano-Pozo 2024 (open, saved as smoke_r3/data_inventory/papers/serrano_pozo2024_tableS1_donors.csv): A (Thal-derived), B, C, Braak, CERAD, pathology stage, APOE, sex, age (90+ capped), regions profiled. Thal-based A score IS published, so ADNC is exactly computable.",
            "microglia": "Wachter 2024: 337,512 brain myeloid nuclei over 32 donors x 5 regions (about 2,100 per library on average; per-region split not verified). Example library GSM8295013 has 10,218 barcodes. Myeloid annotations: processed objects downloadable from ad-progression-atlas.partners.org (Microglia tab; download page returned HTTP 500 on 2026-11-11, so not verified).",
            "processed_subset": "Microglia/myeloid processed data on ad-progression-atlas.partners.org (not verified); GEO has per-library filtered count matrices only (cell types must be re-annotated with a train-only classifier).",
            "file_sizes": "GSE268599_RAW.tar 10.9 GB total; per-library matrix.mtx.gz ~50-100 MB; one region (ITG, ~31 libraries) about 2-3 GB.",
            "raw_counts_on_36601_axis": True,
            "licence": "GEO (see terms). Wachter 2024 article CC BY 4.0; Serrano-Pozo article CC BY-NC-ND.",
            "caveats": ["Sorting changes the composition and possibly per-nucleus profiles relative to SEA-AD unsorted nuclei; same lab sorted all 5 regions, so region effects are estimable within the cohort.",
                        "Not AD donors here are Braak 0-II, CERAD none: tangle-poorer than SEA-AD Not AD (7/9 SEA-AD Not AD have Braak IV-V).",
                        "All 8 Not AD donors are APOE e3/e3."],
        },
        {
            "id": "gerrits_2021",
            "rank": 2,
            "paper": "Gerrits et al. Acta Neuropathol 2021 141:681 doi:10.1007/s00401-021-02263-w (CC BY 4.0)",
            "accession": "GEO GSE148822",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148822",
            "access": "open (GEO)",
            "access_quote": "'The data reported in this study are available through Gene Expression Omnibus at https://www.ncbi.nlm.nih.gov/geo with accession number GSE148822.'",
            "terms": GEO_TERMS,
            "region": "occipital cortex (OC) and occipitotemporal cortex (OTC), 2 per donor; NBB-IBB Antwerp",
            "technology": "FANS NeuN-negative OLIG2-negative nuclei; 10x Chromium 3' v3; Cell Ranger 3.0.0 pre-mRNA",
            "gene_axis": "VERIFIED: 33,538 genes (10x GRCh38-3.0.0, Ensembl 93). 32,732 IDs shared with the 36,601 2020-A axis; 1,850 of the study's 2,000 hvg2000 panel genes present by Ensembl ID (missing mostly lncRNAs).",
            "donors": {"total": 18, "groups": {"CTR (no detectable Abeta/tau in sampled regions)": 5, "CTR+ (mild Abeta, no tau)": 3, "AD": 10},
                       "per_donor": ge},
            "per_donor_variables_open": "GEO characteristics and Table S1 (saved as papers/gerrits2021_tableS1_donors.csv): age, sex, APOE, RIN, PMD, clinical diagnosis, group, and EITHER an NIA-AA ABC score (9 donors, incl. Thal-based A) OR a Braak tau stage only (9 donors). Microglia count per donor-region.",
            "microglia": "147,242 microglia nuclei (Table S1 sum; paper: 148,606), 798-8,029 per donor-region",
            "processed_subset": "No processed microglia object on GEO (per-library filtered matrices; separate small NeuN/OLIG2 bulk counts).",
            "file_sizes": "GSE148822_RAW.tar 4.4 GB (36 snRNA libraries ~60-85 MB matrix.mtx.gz each + bulk)",
            "raw_counts_on_36601_axis": False,
            "licence": "GEO; article CC BY 4.0",
            "caveats": ["Younger (60-80 y) than SEA-AD (mean ~88).", "Occipital regions are late in Braak staging: the AD donors' local tau burden is low.",
                        "The non-ABC 'Braak or ABC score' values of the AD donors (II, II-III, III, IV) are low for clinico-pathological AD; the paper diagnosed pre-2011 cases by Braak criteria, so treat these as tau stages of uncertain meaning.",
                        "Only 3 donors are Not AD-like with confidence (A0: donors 3, 4; CTR with no detectable amyloid and Braak 0-I: donor 5)."],
        },
        {
            "id": "imperial_smith2022_and_gse264648",
            "rank": 3,
            "papers": ["Smith et al. Acta Neuropathol 2022 doi:10.1007/s00401-021-02372-6 (GSE160936; EC + SSC, Sox10-/NeuN- sorted)",
                       "Imperial/UK DRI 'premature cell senescence in AD' GSE264648 (EC, MTG ('mTemp'), SSC; unsorted, 49 blocks)"],
            "accession": "GEO GSE264648 (microglia count object GSE264648_counts_microglia.rds.gz 42 MB, metadata 1.5 MB) and GSE160936 (RAW.tar 1.0 GB)",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264648",
            "access": "open (GEO processed counts). FASTQ on Synapse syn36812517 (UK DRI Multi-omics Atlas): 'Submitter states that missing raw files are due to patient privacy concerns.'",
            "access_quote": "GSE264648 overall design: 'For raw files, visit the project space in https://www.synapse.org/#!Synapse:syn36812517'",
            "terms": GEO_TERMS,
            "region": "MTG for 15 of 17 donors (also EC, SSC). London Neurodegenerative Diseases Brain Bank + Parkinson's UK Brain Bank.",
            "technology": "10x Chromium 3' v3, Cell Ranger 3.1, GRCh38 (GSE160936: gencode v30 pre-mRNA).",
            "gene_axis": "Cell Ranger 3.1 GRCh38 (likely 3.0.0 reference, 33,538 genes) - not verified for GSE264648's rds.",
            "donors": {"total": 17, "control_braak_0_2": 8, "ad_braak_3_6": 9, "per_donor": imp_donors},
            "per_donor_variables_open": "diagnosis, Braak tangle stage, age, sex, PMI, RIN, brain bank, and per-region quantitative pTau (% cells) and amyloid-beta (% area) from IHC. No CERAD, no Thal, no APOE in the open metadata.",
            "microglia": "12,220 microglia nuclei in GSE264648 microglia object (all regions); MTG only: about 3,000 over 15 donors (see per_donor).",
            "processed_subset": "Yes: GSE264648_counts_microglia.rds.gz (42 MB) + metadata_microglia.rds.gz (1.5 MB; downloaded, tabulated, deleted).",
            "file_sizes": "42 MB microglia counts; 1.4 GB all cell types",
            "raw_counts_on_36601_axis": False,
            "caveats": ["Donors overlap between GSE160936 and GSE264648 (e.g. A127/11, A053/11): count each donor once.",
                        "Controls have nonzero regional amyloid area (0.05-0.97%), so 'control' is not A0; Not AD cannot be assigned without Thal/CERAD."],
        },
        {
            "id": "morabito_2021",
            "rank": 5,
            "paper": "Morabito et al. Nat Genet 2021 doi:10.1038/s41588-021-00894-z",
            "accession": "GEO GSE174367 (also Synapse syn22079621)",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174367",
            "access": "open (GEO): single merged filtered matrix h5 (261 MB) + cell metadata (425 KB)",
            "terms": GEO_TERMS,
            "region": "prefrontal cortex (UCI MIND ADRC)",
            "technology": "10x Chromium 3' v3 (snRNA) + snATAC; Cell Ranger 3.0.2 pre-mRNA",
            "donors": {"total_snRNA": len(mor), "by_diagnosis": tally(mor, "diagnosis"), "per_donor": mor},
            "per_donor_variables_open": "diagnosis, age, sex, PMI, RIN, Braak tangle stage, Braak amyloid (plaque) stage None/A/B/C, batch (cell metadata). No Thal, CERAD or APOE.",
            "microglia": f"{sum(int(r['n_microglia']) for r in mor)} microglia nuclei (author labels 'MG'), 141-372 per donor",
            "processed_subset": "cell-type labels in GSE174367_snRNA-seq_cell_meta.csv.gz",
            "raw_counts_on_36601_axis": False,
            "caveats": ["Controls: Braak 1-2; plaque stage None (2 donors), A (3), B (2): only the 2 'None' donors are A0-like."],
        },
        {
            "id": "leng_2021",
            "rank": 7,
            "paper": "Leng et al. Nat Neurosci 2021 doi:10.1038/s41593-020-00764-7",
            "accession": "GEO GSE147528; CELLxGENE collection 180bff9c (EC and SFG microglia subsets 9f1049ac, bdacc907)",
            "access": "open (GEO; CELLxGENE CC BY 4.0). GEO summary: 'Detailed neuropathological, clinical and genetic data are available under request.'",
            "terms": GEO_TERMS + " " + CXG_TERMS,
            "region": "caudal entorhinal cortex and superior frontal gyrus; 10 male donors",
            "technology": "10x Chromium 3' v2, custom GRCh38 pre-mRNA",
            "donors": {"total": 10, "by_braak": tally(leng_ec["donors"], "BraakStage")},
            "per_donor_variables_open": "Braak stage (0/2/6), age, sex. Nothing on amyloid.",
            "microglia": f"EC {leng_ec['n_microglia']}, SFG {leng_sfg['n_microglia']} microglia (CELLxGENE subsets; normalized X only, no raw layer in the subset files)",
            "per_donor": {"EC": leng_ec["donors"], "SFG": leng_sfg["donors"]},
            "raw_counts_on_36601_axis": False,
            "caveats": ["Braak 0 donors are 50-71 y.", "v2 chemistry."],
        },
        {
            "id": "lau_2020",
            "rank": 6,
            "paper": "Lau et al. PNAS 2020 doi:10.1073/pnas.2008762117",
            "accession": "GEO GSE157827",
            "access": "open (GEO): 'Anonymized snRNA-seq sequencing data have been deposited in GEO (accession no. GSE157827).'",
            "terms": GEO_TERMS,
            "region": "prefrontal cortex (BA6/8/9), South West Dementia Brain Bank",
            "technology": "10x Chromium 3' v3; Cell Ranger hg38 pre-mRNA",
            "gene_axis": "VERIFIED 33,538 genes (3.0.0 reference)",
            "donors": {"total": 21, "AD (Braak >= 4)": 12, "NC (Braak <= 2)": 9},
            "per_donor_variables_open": "GEO: diagnosis only. Paper: 'We categorized samples according to Braak stage (i.e., AD: >=4 and NC: <=2)'; per-donor Braak/age/sex in SI Table S1 (not opened). No amyloid stage.",
            "microglia": "about 4.7% of 169,496 nuclei = about 8,000 (paper; per-donor not verified)",
            "raw_counts_on_36601_axis": False,
        },
        {
            "id": "mayo_apoe_2025",
            "rank": 4,
            "paper": "Liu/Zhao et al. Neuron 2025 'APOE genotype determines cell-type-specific pathological landscape of AD' PMID 40112813",
            "accession": "GEO GSE237718 (56 per-sample tar.gz, RAW.tar 5.1 GB)",
            "access": "open (GEO): 'Count matrices, raw sequencing data, and the metadata associated with each sample identifier ... are available in the National Center for Biotechnology Information Gene Expression Omnibus under accession number GSE237718.'",
            "terms": GEO_TERMS,
            "region": "temporal cortex (Mayo Clinic brain bank) - closest open region match to MTG together with ITG above",
            "technology": "10x Chromium 3' v3; Cell Ranger 3.1.0, GRCh38-1.2.0 pre-mRNA",
            "donors": {"total": 56, "control": 27, "AD": 29, "apoe_by_group": "controls: 9 e2 carriers, 10 e3/e3, 8 e4 carriers"},
            "per_donor_variables_open": "GEO: APOE genotype, diagnosis, sex. Paper Table S1 'Information for all human samples' (Braak and CERAD reported at group level; columns NOT verified; Thal not mentioned).",
            "microglia": "5.4% of 343,632 nuclei = about 18,500 (about 330 per donor; derived, not verified)",
            "raw_counts_on_36601_axis": False,
            "caveats": ["Control definition 'not affected by AD neuropathology' - thresholds not verified.", "Older 1.2.0 reference (Ensembl 84)."],
        },
        {
            "id": "cellxgene_layer4_resilience_2026",
            "rank": 8,
            "paper": "'Molecular Signatures of Resilience to AD in Neocortical Layer 4 Neurons' Nat Commun 2026 doi:10.1038/s41467-026-68920-4",
            "accession": "CELLxGENE collection 0d35c0fd, dataset cff99df2 (6.47 GB); GEO GSE263468 (27 GB h5ad)",
            "access": "open, CC BY 4.0",
            "terms": CXG_TERMS,
            "region": "BA9, BA7, BA17",
            "technology": "10x 3' v2/v3 + Drop-seq",
            "donors": {"total": 46, "disease_labels": "Alzheimer disease / normal"},
            "per_donor_variables_open": "VERIFIED obs columns: Braak.stage, Amyloid, APOE_class, NP.Diagonis, Disease.Group, Age, PMI, RIN, Race, SORT (a sort column: some libraries may be sorted). Per-donor tally NOT done (byte budget).",
            "microglia": "not counted",
            "raw_counts_on_36601_axis": False,
        },
        {
            "id": "cellxgene_fresh_myeloid_2026",
            "rank": 9,
            "paper": "'Plasticity of Human Microglia and BAMs in Aging and AD' Nat Genet 2026 doi:10.1038/s41588-026-02716-6",
            "accession": "CELLxGENE collection f6ea32cb, dataset 440da0a0 (5.25 GB, 543,012 cells, 137 donors)",
            "access": "open, CC BY 4.0",
            "terms": CXG_TERMS,
            "region": "dorsolateral prefrontal cortex",
            "technology": "fresh ex vivo sorted myeloid cells, single-CELL 10x 3' v2/v3 (not nuclei: whole-cell transcriptome, large modality gap to snRNA)",
            "donors": {"total": 137, "disease_labels": "Alzheimer disease / CNS disorder / normal"},
            "per_donor_variables_open": "VERIFIED obs columns: Braak, CERAD, dementia, dx, PMI, Source, Type. Per-donor tally NOT done (two obs reads hit the byte budget).",
            "microglia": "most of 543,012 cells",
            "raw_counts_on_36601_axis": False,
        },
        {
            "id": "cellxgene_psychad",
            "rank": 10,
            "paper": "PsychAD (preprint doi:10.1101/2024.10.31.24316513; myeloid Nat Genet 2026)",
            "accession": "CELLxGENE 09642770 (myeloid, 3.0 GB) and cohort objects (MSSM 36 GB, RADC 6.3 GB, HBCC 14 GB, Aging 12 GB)",
            "access": "expression open (CC BY 4.0); pathology (CERAD, Braak) controlled at AD Knowledge Portal syn52160016",
            "region": "DLPFC", "technology": "10x 3' v3 nuclei",
            "donors": {"total": 1469, "aged_controls_100plus_myeloid": 141, "aged_AD_100plus_myeloid": 404,
                       "source": "DATA_EXPANSION.md census (remote_cxg_psychad_myeloid_census.json)"},
            "per_donor_variables_open": "diagnosis flags only (AD, Control, ...). 'Control' = no clinical neuropsychiatric diagnosis, pathology unknown in open data.",
            "raw_counts_on_36601_axis": False,
        },
        {
            "id": "cellxgene_small_open",
            "rank": 11,
            "items": [
                {"name": "Rexach 2024 cross-dementia (c53573b2/ac0c6561)", "donors": 40, "by_disease": tally(rex["donors"], "disease"), "microglia": rex["n_microglia"],
                 "regions": "BA4, insula, V1", "pathology": "disease label only; controls 56-79 y", "tech": "10x v2/v3"},
                {"name": "Brain vascular multi-omics (433700dc/203025fe, Neuron 2025)", "donors": 30, "by_disease": tally(vasc["donors"], "disease"), "microglia": vasc["n_microglia"],
                 "regions": "DLPFC", "pathology": "disease label only (likely ROSMAP-derived)", "tech": "10x multiome"},
                {"name": "CSF1R-related disorder (7c4552fd/fe2eecbc)", "donors": 8, "by_disease": tally(csf1r["donors"], "disease"), "microglia": csf1r["n_microglia"],
                 "pathology": "Braak + CERAD-like 'Amyloid' C0/C1 + APOE; 5 normals with C0 but 17-180 microglia each", "per_donor": csf1r["donors"]},
                {"name": "Olah 2020 live microglia (fcb3d1c1/a1b9c51e)", "donors": 17, "by_disease": tally(olah["donors"], "disease"), "microglia": olah["n_microglia"],
                 "pathology": "AD vs temporal lobe epilepsy surgical; no staging", "tech": "fresh cells 10x v2"},
                {"name": "UPenn GSE243292 (DLPFC)", "donors": 15, "by_group": "normal 2, pathological aging 5, AD 8 (A/T score, APOE, TREM2)", "microglia": 3982,
                 "tech": "10x, HiSeq 2500; 26,423 filtered genes; Microglia_GEO.h5ad 7.2 MB"},
                {"name": "Anderson 2023 GSE214979 (DLPFC multiome)", "donors": "about 20 (control 22 / AD 18 libraries incl. ATAC)", "pathology": "disease state, age (controls 49-77)"},
                {"name": "Grubman 2019 GSE138852 (EC)", "donors": "12 (6 AD, 6 control)", "microglia": 449, "tech": "10x v2", "note": "too few microglia"},
            ],
        },
    ]
    controlled = [
        {"id": "rosmap_mathys2023_sun2023_green2024", "access": "AD Knowledge Portal DUC (syn52293417 clickwrap + ManagedACT; syn31512863 ManagedACT) for pathology; open Broad files without licence (see DATA_EXPANSION.md 4.2)", "donors": "427 / 424 / 437 (one cohort)"},
        {"id": "prater_2023_pu1", "access": "VERIFIED Synapse syn51272688: processed objects and raw data carry a ManagedACTAccessRequirement 'PU.1 Data Use Statement'; only supplementary tables are open", "donors": "22 (10 ADNC 0-1)"},
        {"id": "zhou_2020_human", "access": "Synapse syn21125841 (ROSMAP-derived; children not listable anonymously) - controlled", "donors": "32 DLPFC"},
        {"id": "brase_2023_knight_adrc", "access": "NIAGADS NG00108 (data access request) + DIAN restricted: 'available under restricted access to maintain individual and family confidentiality'", "donors": "Knight ADRC parietal"},
        {"id": "olah_2020_fastq", "access": "syn21438358 ADKP clickwrap + ManagedACT (processed counts open on CELLxGENE)"},
        {"id": "seaad_fastq", "access": "syn26223298 ADKP clickwrap (processed SEA-AD data open on S3/CELLxGENE)"},
    ]
    census = {
        "route": "CELLxGENE Census (tiledbsoma) = all primary-data CELLxGENE datasets on one Ensembl gene axis with raw counts",
        "verified": ["CELLxGENE index (2,271 public datasets, 2026-11-11): 211 human datasets list microglia; AD-relevant ones listed in cxg/cxg_microglia_datasets.txt",
                     "Standard donor fields are donor_id, disease (MONDO), development_stage, sex, self_reported_ethnicity, tissue, assay. Braak/CERAD/Thal are NOT schema fields; they exist only as author columns in some datasets (Layer4, fresh myeloid, Leng, CSF1R) and are not exposed through Census queries.",
                     "Census keeps is_primary_data=True cells: PsychAD myeloid (09642770) and SEA-AD Microglia-PVM subsets are non-primary duplicates; their cells reach Census through the cohort / whole-taxonomy objects."],
        "approximate_yield_microglia_donors_by_open_label": {
            "SEA-AD MTG + DLPFC": "89 / 83 donors, disease = dementia or normal (cognitive, not ADNC); ADNC is in SEA-AD's own open metadata, not in Census",
            "PsychAD (MSSM + RADC + HBCC)": "about 1,494 donors; myeloid census: Control 470, AD 560 (aged controls with >=100 myeloid nuclei: 141)",
            "Rexach 2024": "10 normal, 10 AD, 10 Pick, 10 PSP",
            "Brain vascular 2025": "10 normal, 10 cognitive disorder, 10 AD",
            "PD multi-region atlas 2026": "101 donors (PD / normal), MTG among regions, no AD staging",
            "others": "Olah 17, Leng 10, CSF1R 8, Layer4 46, fresh myeloid 137, HVS 61 young neurosurgical",
        },
        "verdict": "Census is a good harmonised route for 'normal vs AD' labels and a common gene axis, but NOT for neuropathological stage: none of the Census-standard fields carries Braak, CERAD, Thal or ADNC, and the large contributors (PsychAD, SEA-AD) label disease clinically. Not installed or queried directly (cellxgene_census not present; would need a new package); counts come from per-dataset obs reads and the CELLxGENE curation index.",
    }
    inv = {
        "component": "improve_20261109 round 3, R8 open-data inventory",
        "date_checked": "2026-11-11",
        "binding_limit": "dev37 has 4 Not AD donors (inner-fold training 2-3)",
        "method": "GEO series matrices and file lists, NCBI E-utilities search (254 human AD single-cell/nucleus series screened by title), CELLxGENE curation index + HTTPS range reads of obs only, Europe PMC full texts, Springer supplementary tables, Synapse REST access requirements. No expression matrix was read.",
        "download_budget_note": "About 95 MB were transferred in total (CELLxGENE index 9.9 MB, obs range reads about 55 MB of which about 35 MB were wasted on three reads that hit their byte cap, one 19 MB Springer workbook to extract Gerrits Table S1, GEO metadata about 4 MB, papers about 2 MB). This exceeds the about-50 MB guidance; nothing larger than 20 MB was a single file, and only small tables were kept (smoke_r3/data_inventory is about 2 MB).",
        "feasibility": {
            "open_not_ad_donors_exact_adnc": {"Serrano-Pozo/Wachter MADRC (A0 from published ABC)": 8, "Gerrits 2021 (A0)": 2,
                                              "total": 10, "vs_dev37": 4},
            "open_not_ad_like_probable": {"Gerrits donor 5 (CTR, no detectable amyloid, Braak 0-I)": 1,
                                          "Morabito controls with plaque stage None": 2},
            "open_controls_without_amyloid_stage": {"Imperial GSE264648 (Braak 0-2)": 8, "Lau 2020 (Braak <= 2)": 9,
                                                    "Mayo GSE237718 (control; staging in unverified Table S1)": 27,
                                                    "Leng 2021 (Braak 0, 50-71 y)": 3},
            "label_mapping": "Only the MADRC set (and 9 Gerrits donors) publish the Thal-based A score, so only they map exactly onto the 4 ADNC levels. Every other open set gives Braak (+/- CERAD or a Braak amyloid stage): from SEA-AD's own cross-tab, Not AD predicted from Braak+CERAD is right for only ~56% of donors.",
            "batch_problem": "Different region, sorting (FANS NeuN-/OLIG2- vs unsorted), chemistry (v2/v3), reference (3.0.0 vs 2020-A vs pre-mRNA builds), lab and age. No published number was found that directly compares cross-dataset microglia pseudobulk differences with stage effects; in this study the within-SEA-AD High-minus-Not-AD contrast is not even reproducible between donor halves (split-half 0.135), so any cohort offset larger than that would swamp it. This was not measured here (no counts read).",
            "cheapest_real_measurement": "Download GSE264648_counts_microglia.rds.gz (42 MB; MTG for 15 donors) and compare its MTG microglia pseudobulk to dev37 TRAIN donors' pseudobulk on the shared Ensembl axis: cohort offset vs stage contrast magnitude, before any training decision.",
        },
        "recommendation": {
            "can_do_now_open": [
                "1. MADRC (GSE268599): ITG libraries of the 31 donors that have ITG (7 Not AD, 5 Low, 6 Intermediate, 13 High by published ABC). Same 36,601 gene axis as SEA-AD. Re-annotate microglia with the study's train-only type classifier; add a dataset/region indicator to the condition vector; use as additional TRAINING donors only, keep SEA-AD MTG held-out donors as the only evaluation. About 2-3 GB transient download, needs a protocol amendment (it adds non-SEA-AD counts).",
                "2. Before that, measure the cohort offset cheaply on GSE264648 microglia (42 MB, MTG) as above; if the offset dwarfs the stage contrast, use external donors only for external validation of stage direction (e.g. MADRC ITG High-minus-NotAD vs the generator's MTG contrast), not for training.",
                "3. Gerrits 2021 (2-3 Not AD-like, 147k microglia, occipital) as a second external validation set; lower priority because of region and age.",
            ],
            "needs_agreement": [
                "ROSMAP (Mathys/Sun/Green; ~189 NIA-Reagan non-AD) and PsychAD pathology: AD Knowledge Portal DUC signed by an institutional official.",
                "Prater 2023 PU.1 microglia (10 ADNC 0-1 controls, same UW staging scale): Synapse 'PU.1 Data Use Statement'.",
                "Brase 2023 (NIAGADS NG00108 / DIAN).",
            ],
        },
        "datasets_open": datasets,
        "datasets_controlled": controlled,
        "cellxgene_census": census,
    }
    OUT.write_text(json.dumps(inv, indent=1, default=str))
    return inv


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        inv = build()
        print("wrote", OUT, len(inv["datasets_open"]), "open records")
        sys.exit(0)
    if cmd == "peek":
        a = sys.argv[2:]
        r = peek(a[0], float(a[1]) if len(a) > 1 else 12.0,
                 a[2] if len(a) > 2 else "donor_id", a[3] if len(a) > 3 else "cell_type",
                 a[4] if len(a) > 4 else r"microglia|microglial",
                 set(a[5].split(",")) if len(a) > 5 and a[5] else None)
        print(json.dumps(r, indent=1, default=str))
    elif cmd == "cxg":
        print(json.dumps(cxg(sys.argv[2]), indent=1))
    else:
        raise SystemExit(__doc__)
