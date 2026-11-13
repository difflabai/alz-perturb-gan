"""Fetch the open external donors (round 4, external-donor workstream).

Sources (only these two GEO series are allowed by r4/PROTOCOL_EXT.md):
  GSE268599  MGH-AbbVie MADRC cohort (Serrano-Pozo 2024 Nat Neurosci; Wachter
             2024 Acta Neuropathol). Per-library 10x filtered matrices; only the
             inferior temporal gyrus (ITG, "BA20") libraries are downloaded.
  GSE264648  Imperial/UK DRI microglia object (counts_microglia.rds.gz, 42 MB)
             plus its metadata object; cohort-offset check only.
Donor-level tables come from the papers' open supplementary workbooks
(Serrano-Pozo 2024 MOESM3: Supplementary Table 1 + "Quality Control Data";
Wachter 2024 MOESM2: Table S1 only - the 131 MB workbook is deleted after the
table is extracted).

Subcommands (python3 -B ext_fetch.py <cmd>):
  meta        download metadata, verify access terms, build libraries.json and
              donors_meta.json (ADNC from A, B, C with the NIA-AA 2012 table)
  libs        for every donor with an ITG library: download, select microglia
              (ext_select_microglia.select_library), write counts/{donor}.npz,
              {donor}_names.npy, cellmeta/{donor}.npz and receipts/{donor}.json,
              verify, delete the raw matrix. Rerunnable: donors whose receipt
              verifies against the files on disk are skipped.
  gse264648   parse the two .rds objects with the pure-Python RDS reader below
              (R and rpy2 are not installed) and write ext/GSE264648/.
  donors      (re)write ext/GSE268599/donors.json from donors_meta.json plus
              receipts (done automatically at the end of libs).
  status      print progress.
  terms-eutils  (review finding D4) scriptable re-check of public status through
              NCBI E-utilities esummary (db=gds; eutils.ncbi.nlm.nih.gov is not
              behind the reCAPTCHA that www.ncbi.nlm.nih.gov shows scripts) plus
              the anonymous FTP listing of each series' suppl/ folder; merged into
              ext/meta/access_terms.json under 'eutils_check'. The GEO disclaimer
              wording stays NOT VERIFIED (needs a browser).

Network: plain HTTPS with a generic User-Agent; no personal details are sent.
Disk: one library (about 30-120 MB gz) is on disk at a time under
ext/GSE268599/raw/{GSM}/ and is deleted after its outputs verify.
CPU: thread pools capped at 3 (environment variables set before numpy import).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "3"
import argparse
import gzip
import hashlib
import html
import io
import json
import re
import shutil
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
EXT = Path("/ml2/alz/runs/improve_20261109/ext")
OUT1 = EXT / "GSE268599"
OUT2 = EXT / "GSE264648"
META = EXT / "meta"
BENCH = Path("/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark")
REBUILT = BENCH / "rebuilt"
UA = "generic-research-client/1.0"
GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo"

URLS = {
    "gse268599_matrix": f"{GEO_FTP}/series/GSE268nnn/GSE268599/matrix/GSE268599_series_matrix.txt.gz",
    "gse268599_filelist": f"{GEO_FTP}/series/GSE268nnn/GSE268599/suppl/filelist.txt",
    "gse268599_page": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268599",
    "gse264648_matrix": f"{GEO_FTP}/series/GSE264nnn/GSE264648/matrix/GSE264648_series_matrix.txt.gz",
    "gse264648_page": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264648",
    "gse264648_counts": f"{GEO_FTP}/series/GSE264nnn/GSE264648/suppl/GSE264648_counts_microglia.rds.gz",
    "gse264648_meta": f"{GEO_FTP}/series/GSE264nnn/GSE264648/suppl/GSE264648_metadata_microglia.rds.gz",
    "geo_disclaimer": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
    "serrano_article": "https://www.nature.com/articles/s41593-024-01791-4",
    "serrano_supp": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-024-01791-4/MediaObjects/41593_2024_1791_MOESM3_ESM.xlsx",
    "wachter_article": "https://link.springer.com/article/10.1007/s00401-024-02704-2",
    "wachter_supp": "https://static-content.springer.com/esm/art%3A10.1007%2Fs00401-024-02704-2/MediaObjects/401_2024_2704_MOESM2_ESM.xlsx",
}
STAGES = ["Not AD", "Low", "Intermediate", "High"]
BRAAK = {"0": 0, "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
CERAD = {"none": 0, "sparse": 1, "moderate": 2, "frequent": 3}
# NIA-AA A score -> Thal phase range (Montine 2012): A0=Thal 0, A1=1-2, A2=3, A3=4-5
A_TO_THAL = {0: (0, 0), 1: (1, 2), 2: (3, 3), 3: (4, 5)}


# ----------------------------------------------------------------------------- helpers
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def http_get(url, dest=None, expect_size=None, tries=4):
    """GET with a generic UA. Streams to dest (atomic rename) or returns bytes."""
    import requests
    last = None
    for attempt in range(tries):
        try:
            with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=120) as r:
                r.raise_for_status()
                if dest is None:
                    return r.content
                dest = Path(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".partial")
                n = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        n += len(chunk)
                if expect_size is not None and n != expect_size:
                    raise IOError(f"size mismatch for {url}: {n} != {expect_size}")
                tmp.replace(dest)
                return n
        except Exception as e:  # retry transient failures
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}: {last}")


def page_text(raw):
    t = raw.decode("utf8", "replace")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S)
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t)))


def find_sentence(text, key, before=0, after=600):
    i = text.find(key)
    return None if i < 0 else text[max(0, i - before):i + after].strip()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    tmp.replace(path)


def parse_series_matrix(path):
    """Return (series dict of lists, samples list of dicts) from a GEO series matrix."""
    series, cols = {}, {}
    with gzip.open(path, "rt", encoding="utf8") as f:
        for line in f:
            if not line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            key, vals = parts[0], [v.strip('"') for v in parts[1:]]
            if key.startswith("!Series_"):
                series.setdefault(key[8:], []).append(" ".join(vals))
            elif key.startswith("!Sample_"):
                cols.setdefault(key[8:], []).append(vals)
    n = len(cols["geo_accession"][0])
    samples = []
    for j in range(n):
        s = {}
        for k, rows in cols.items():
            for r in rows:
                v = r[j] if j < len(r) else ""
                if k == "characteristics_ch1" and ": " in v:
                    a, b = v.split(": ", 1)
                    s.setdefault("char", {})[a.strip().lower()] = b
                else:
                    s.setdefault(k, []).append(v)
        samples.append({k: (v if k == "char" else (v[0] if len(v) == 1 else v)) for k, v in s.items()})
    return series, samples


# ----------------------------------------------------------------------------- ADNC
def adnc_2012(a, b, c):
    """NIA-AA 2012 (Montine et al. 2012, Table 4) ABC -> ADNC level name.

    A0 -> Not AD whatever B and C. A1 -> Low. A2: B0/B1 -> Low, B2/B3 ->
    Intermediate. A3: B0/B1 -> Low if C0/C1 else Intermediate; B2 ->
    Intermediate; B3 -> Intermediate if C0/C1 else High.
    """
    a, b, c = int(a), int(b), int(c)
    if not (0 <= a <= 3 and 0 <= b <= 3 and 0 <= c <= 3):
        raise ValueError((a, b, c))
    if a == 0:
        return "Not AD"
    if a == 1:
        return "Low"
    if a == 2:
        return "Low" if b <= 1 else "Intermediate"
    if b <= 1:
        return "Low" if c <= 1 else "Intermediate"
    if b == 2:
        return "Intermediate"
    return "Intermediate" if c <= 1 else "High"


def braak_to_b(braak):
    """NIA-AA B score from Braak stage: 0 -> B0, I-II -> B1, III-IV -> B2, V-VI -> B3."""
    s = BRAAK[braak]
    return 0 if s == 0 else 1 if s <= 2 else 2 if s <= 4 else 3


def cerad_to_c(cerad):
    return CERAD[cerad]


# ----------------------------------------------------------------------------- metadata
def seaad_gene_ids():
    """Ensembl IDs of the SEA-AD 36,601-gene axis, range-read from the var table of
    one DEVELOPMENT donor's public h5ad (var only, about 2 MB; no counts read)."""
    out = META / "seaad_gene_ids.npy"
    if out.exists():
        return np.load(out, allow_pickle=False)
    sys.path.insert(0, str(BENCH))
    from range_h5 import RangeFile  # frozen helper, imported read-only
    import h5py
    dev = json.loads((STUDY / "folds.json").read_text())["development_donors"]
    inv = json.loads((BENCH / "donor_object_inventory.json").read_text())["objects"]
    donor = sorted(dev)[0]
    obj = next(o for o in inv if f"/{donor}_SEAAD" in o["key"])
    with RangeFile("https://sea-ad-single-cell-profiling.s3.amazonaws.com/" + obj["key"], obj["size"], obj["etag"]) as r:
        with h5py.File(r, "r") as f:
            ids = np.array([x.decode() for x in f["var"]["gene_ids"][:]])
            sym = np.array([x.decode() for x in f["var"]["index"][:]])
    genes = np.load(REBUILT / "counts/genes.npy", allow_pickle=False)
    if not np.array_equal(sym, genes):
        raise RuntimeError("SEA-AD var index differs from rebuilt/counts/genes.npy")
    META.mkdir(parents=True, exist_ok=True)
    np.save(out, ids.astype("<U15"))
    write_json(META / "seaad_gene_ids.receipt.json", {"donor_object": obj["key"], "etag": obj["etag"],
               "n": len(ids), "symbols_equal_rebuilt_genes_npy": True, "sha256": sha256(out)})
    return ids


def read_xlsx_sheet(path, sheet):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    hdr = [str(h) for h in rows[0]]
    return [dict(zip(hdr, r)) for r in rows[1:] if any(v is not None for v in r)]


def cmd_meta(args):
    META.mkdir(parents=True, exist_ok=True)
    got = {}
    for k in ["gse268599_matrix", "gse268599_filelist", "gse264648_matrix"]:
        dest = META / Path(URLS[k]).name.replace("filelist.txt", "GSE268599_filelist.txt")
        if not dest.exists():
            http_get(URLS[k], dest)
        got[k] = {"url": URLS[k], "file": str(dest), "sha256": sha256(dest)}
    # Access terms, quoted from the source pages (pages are parsed, not kept).
    terms = {"checked_unix": time.time()}
    for k in ["gse268599_page", "gse264648_page", "geo_disclaimer"]:
        t = page_text(http_get(URLS[k]))
        blocked = "Checking your browser" in t
        terms[k] = {"url": URLS[k], "reachable_by_script": not blocked,
                    "text": ("NOT VERIFIED: www.ncbi.nlm.nih.gov served a reCAPTCHA browser-check interstitial to "
                             "scripted clients (curl and a second fetcher) on the check date") if blocked else
                    (find_sentence(t, "NCBI places no restrictions", 0, 500) or find_sentence(t, "cannot provide comment", 300, 500)
                     or find_sentence(t, "Status", 0, 300))}
    # Terms that ARE verifiable from the GEO FTP series matrices.
    for gse, fn in [("GSE268599", "GSE268599_series_matrix.txt.gz"), ("GSE264648", "GSE264648_series_matrix.txt.gz")]:
        ser, sam = parse_series_matrix(META / fn)
        terms[gse + "_series_matrix"] = {
            "status": ser.get("status"), "overall_design_quotes": [x for x in ser.get("overall_design", []) if x.strip("* ")],
            "sample_status": sorted({s.get("status") for s in sam}),
            "sample_consent_values": sorted({s.get("char", {}).get("consent", "absent") for s in sam}),
            "licence_field": "none present in the series matrix"}
    ts = page_text(http_get(URLS["serrano_article"]))
    terms["serrano_pozo_2024"] = {"url": URLS["serrano_article"],
                                  "data_availability": find_sentence(ts, "Data availability The snRNA-seq", 0, 700),
                                  "licence": find_sentence(ts, "Open Access This article is licensed", 0, 420)}
    tw = page_text(http_get(URLS["wachter_article"]))
    terms["wachter_2024"] = {"url": URLS["wachter_article"],
                             "data_availability": find_sentence(tw, "Data availability Raw data", 0, 260),
                             "licence": find_sentence(tw, "Open Access This article is licensed", 0, 420),
                             "methods_qc": find_sentence(tw, "Nuclei were quality checked", 0, 900)}
    write_json(META / "access_terms.json", terms)
    # Supplementary tables.
    sp_x = META / "serrano_pozo2024_MOESM3.xlsx"
    if not sp_x.exists():
        http_get(URLS["serrano_supp"], sp_x)
    got["serrano_supp"] = {"url": URLS["serrano_supp"], "sha256": sha256(sp_x)}
    s1 = read_xlsx_sheet(sp_x, "Supplementary Table 1")
    qc = read_xlsx_sheet(sp_x, "Quality Control Data")
    w1p = META / "wachter2024_tableS1.json"
    if not w1p.exists():
        tmp = META / "wachter2024_MOESM2.xlsx"
        http_get(URLS["wachter_supp"], tmp)
        h = sha256(tmp)
        w1 = read_xlsx_sheet(tmp, "TableS1")
        tmp.unlink()
        write_json(w1p, {"url": URLS["wachter_supp"], "sha256_of_deleted_workbook": h, "rows": w1})
    w1 = json.loads(w1p.read_text())["rows"]
    seaad_ids = seaad_gene_ids()

    # Libraries: GEO samples x QC sheet (library code 6289-MW-xxxx -> Donor.ID, region).
    series, samples = parse_series_matrix(META / "GSE268599_series_matrix.txt.gz")
    filelist = {}
    for line in (META / "GSE268599_filelist.txt").read_text().splitlines()[1:]:
        p = line.split("\t")
        if p[0] == "File":
            filelist[p[1]] = int(p[3])
    qc_by_code = {}
    for r in qc:
        qc_by_code[str(r["SampleName"])] = r
    s1_by = {int(r["Donor.ID"]): r for r in s1}
    w1_by = {int(r["Donor.ID"]): r for r in w1}
    libs, problems = [], []
    for s in samples:
        gsm = s["geo_accession"]
        files = [Path(u).name for k, u in s.items() if k.startswith("supplementary_file") and isinstance(u, str) and u]
        code = re.search(r"(6289-MW-\d+)", files[0]).group(1)
        q = qc_by_code.get(code)
        if q is None:
            problems.append(f"{gsm} {code}: not in QC sheet")
            continue
        donor_id = int(q["Donor.ID"])
        region = s["source_name_ch1"]
        if q["region"] != region:
            problems.append(f"{gsm}: region {region} vs QC sheet {q['region']}")
        d1 = s1_by[donor_id]
        sex_geo = s["char"].get("sex", "")[:1].upper()
        age_geo = s["char"].get("age", "")
        if sex_geo != d1["Sex"] or str(age_geo) != str(d1["Age"]):
            problems.append(f"{gsm}: GEO sex/age {sex_geo}/{age_geo} vs Table S1 {d1['Sex']}/{d1['Age']} (donor {donor_id})")
        libs.append({"gsm": gsm, "title": s["title"], "isolate": s["char"].get("isolate"), "library_code": code,
                     "donor_id": donor_id, "region": region, "geo_age": age_geo, "geo_sex": sex_geo,
                     "files": {f.split(code + "_")[1]: {"name": f, "size": filelist.get(f),
                               "url": f"{GEO_FTP}/samples/{gsm[:-3]}nnn/{gsm}/suppl/{f}"} for f in files},
                     "qc_sheet": {k: q[k] for k in ["sample_name", "number_nuclei_sorted", "RIN", "average UMI/cell",
                                                     "average genes-detected/cell", "average percent.mito (%)",
                                                     "Number_of_astrocytes"]}})
    isolates = {}
    for l in libs:
        isolates.setdefault(l["donor_id"], set()).add(l["isolate"])
    for d, iso in isolates.items():
        if len(iso) != 1:
            problems.append(f"donor {d}: several GEO isolates {iso}")
    # Donor table.
    donors = {}
    for did, r in sorted(s1_by.items()):
        a, b, c = int(r["A"]), int(r["B"]), int(r["C"])
        adnc = adnc_2012(a, b, c)
        if braak_to_b(str(r["Braak"])) != b:
            problems.append(f"donor {did}: B={b} inconsistent with Braak {r['Braak']}")
        if cerad_to_c(r["CERAD"]) != c:
            problems.append(f"donor {did}: C={c} inconsistent with CERAD {r['CERAD']}")
        w = w1_by.get(did, {})
        for k in ["A", "B", "C", "Braak", "CERAD", "APOE", "Sex", "Age"]:
            if str(w.get(k)) != str(r[k]):
                problems.append(f"donor {did}: Wachter Table S1 {k}={w.get(k)} vs Serrano-Pozo {r[k]}")
        age_raw = str(r["Age"])
        apoe = str(r["APOE"])
        itg = [l for l in libs if l["donor_id"] == did and l["region"] == "BA20"]
        donors[f"MADRC-{did:02d}"] = {
            "cohort": "MADRC (GSE268599)", "cohort_code": 1, "donor_id_table_s1": did,
            "geo_isolate": sorted(isolates.get(did, {None}), key=str)[0],
            "region": "ITG (BA20)", "A": a, "B": b, "C": c, "ADNC": adnc, "ADNC_index": STAGES.index(adnc),
            "thal_range_from_A": list(A_TO_THAL[a]), "Braak": str(r["Braak"]), "Braak_index": BRAAK[str(r["Braak"])],
            "CERAD": r["CERAD"], "CERAD_index": CERAD[r["CERAD"]], "pathology_stage_paper": r["Pathology Stage"],
            "APOE": apoe, "APOE4_dose": apoe.count("4"), "sex": r["Sex"], "male": int(r["Sex"] == "M"),
            "age_raw": age_raw, "age_90plus_censored": age_raw == "90+",
            "TREM2_variant": w.get("TREM2"), "TDP43_pathology": w.get("TDP-43 pathology"),
            "alpha_synuclein_pathology": w.get("Alpha Synuclein pathology"),
            "regions_profiled_table_s1": [k for k in ["EC", "ITG", "PFC", "V2", "V1"] if r[k] == "yes"],
            "itg_libraries": [l["gsm"] for l in itg],
            "itg_table_s1": r["ITG"] == "yes",
        }
        if (r["ITG"] == "yes") != bool(itg):
            problems.append(f"donor {did}: Table S1 ITG={r['ITG']} but GEO BA20 libraries {len(itg)}")
    # Age coding for "90+": SEA-AD reports exact ages (development donors aged >= 90:
    # 90..102). Code "90+" as the mean age of SEA-AD development donors aged >= 90.
    mg = json.loads((REBUILT / "mg_meta.json").read_text())
    dev = set(json.loads((STUDY / "folds.json").read_text())["development_donors"])
    ci = mg["cols"].index("Age at Death")
    ages = {row[0]: float(row[ci]) for row in mg["rows"].values() if row[0] in dev}
    old = np.array([v for v in ages.values() if v >= 90])
    age90 = round(float(old.mean()), 2)
    for d in donors.values():
        d["age"] = age90 if d["age_90plus_censored"] else float(d["age_raw"])
    counts = {s: sum(d["ADNC"] == s for d in donors.values()) for s in STAGES}
    counts_itg = {s: sum(d["ADNC"] == s and bool(d["itg_libraries"]) for d in donors.values()) for s in STAGES}
    expect = {"Not AD": 8, "Low": 5, "Intermediate": 6, "High": 13}
    cross = {"computed_all": counts, "computed_with_itg": counts_itg, "inventory_all": expect,
             "inventory_not_ad_with_itg": 7, "match_all": counts == expect,
             "match_not_ad_itg": counts_itg["Not AD"] == 7}
    age_coding = {"rule": "'90+' -> mean age at death of the SEA-AD development donors aged >= 90 (SEA-AD codes exact ages, "
                          "90-102 among the 37 development donors); flag age_90plus_censored kept so a cache builder can "
                          "recompute the value from the train donors of a fold",
                  "value": age90, "n_seaad_dev_donors_ge90": int(len(old)), "seaad_dev_ages_ge90": sorted(old.tolist()),
                  "n_madrc_90plus": sum(d["age_90plus_censored"] for d in donors.values()),
                  "note": "MADRC Table S1 caps most ages at '90+' but lists donors 24 (91) and 28 (92) exactly; GEO "
                          "characteristics carry the same strings"}
    write_json(META / "libraries.json", {"series_title": series.get("title"), "n_samples": len(samples),
                                         "libraries": libs, "problems": problems})
    write_json(META / "donors_meta.json", {"donors": donors, "adnc_crosscheck": cross, "age_coding": age_coding,
                                           "downloads": got, "problems": problems,
                                           "seaad_gene_ids_n": int(len(seaad_ids))})
    print(json.dumps({"libraries": len(libs), "itg": sum(l["region"] == "BA20" for l in libs),
                      "adnc": cross, "age_coding": age_coding["value"], "problems": problems}, indent=1))


# ----------------------------------------------------------------------------- 10x reading
def read_10x(dirpath, prefix):
    """Read a 10x filtered matrix (features x barcodes MatrixMarket) into cells x genes CSR int32."""
    import pandas as pd
    with gzip.open(dirpath / f"{prefix}matrix.mtx.gz", "rt") as f:
        header = f.readline()
        if not header.startswith("%%MatrixMarket matrix coordinate integer general"):
            raise ValueError(f"unexpected MatrixMarket header: {header!r}")
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        G, C, N = map(int, line.split())
        df = pd.read_csv(f, sep=" ", header=None, dtype=np.int64, engine="c")
    v = df.values
    if v.shape != (N, 3):
        raise ValueError(f"truncated matrix: {v.shape} vs {N} entries")
    if (v[:, 2] <= 0).any() or v[:, 2].max() > np.iinfo(np.int32).max:
        raise ValueError("non-positive or overflowing counts")
    M = sp.csr_matrix((v[:, 2].astype(np.int32), (v[:, 1] - 1, v[:, 0] - 1)), shape=(C, G))
    M.sum_duplicates()
    M.sort_indices()
    feat = pd.read_csv(dirpath / f"{prefix}features.tsv.gz", sep="\t", header=None, dtype=str)
    bc = pd.read_csv(dirpath / f"{prefix}barcodes.tsv.gz", header=None, dtype=str)[0].values
    if len(feat) != G or len(bc) != C:
        raise ValueError("features/barcodes do not match matrix shape")
    return M, feat[0].values.astype(str), feat[1].values.astype(str), bc.astype(str)


# ----------------------------------------------------------------------------- per-donor store
def verify_donor(donor, rec):
    f = OUT1 / "counts" / f"{donor}.npz"
    n = OUT1 / "counts" / f"{donor}_names.npy"
    cm = OUT1 / "cellmeta" / f"{donor}.npz"
    if not (f.exists() and n.exists() and cm.exists()):
        return False
    return (sha256(f) == rec["counts_sha256"] and sha256(n) == rec["names_sha256"]
            and sha256(cm) == rec["cellmeta_sha256"])


def process_donor(donor, dmeta, libs, ids_seaad, sym_seaad, keep_raw=False):
    import ext_select_microglia as sel
    recp = OUT1 / "receipts" / f"{donor}.json"
    if recp.exists():
        rec = json.loads(recp.read_text())
        if verify_donor(donor, rec):
            print(json.dumps({"donor": donor, "status": "skip, verified"}), flush=True)
            return rec
        raise RuntimeError(f"{donor}: receipt exists but files do not verify; inspect before rerun")
    t0 = time.time()
    mats, names, metas, lib_receipts = [], [], [], []
    for lib in libs:
        raw = OUT1 / "raw" / lib["gsm"]
        raw.mkdir(parents=True, exist_ok=True)
        files = {}
        for kind in ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]:
            info = lib["files"][kind]
            dest = raw / info["name"]
            if not (dest.exists() and dest.stat().st_size == info["size"]):
                http_get(info["url"], dest, expect_size=info["size"])
            files[kind] = {"name": info["name"], "bytes": info["size"], "sha256": sha256(dest), "url": info["url"]}
        prefix = f"{lib['gsm']}_{lib['library_code']}_"
        M, ens, sym, bc = read_10x(raw, prefix)
        if not np.array_equal(ens, ids_seaad):
            raise RuntimeError(f"{lib['gsm']}: Ensembl axis differs from SEA-AD")
        sym_equal = bool(np.array_equal(sym, sym_seaad))
        res = sel.select_library(M, sym_seaad, bc, library=lib["gsm"])
        keep = res["keep"]
        sub = M[keep].astype(np.int32)
        sub.sort_indices()
        mats.append(sub)
        names.append(np.array([f"{b}-{lib['gsm']}" for b in bc[keep]]))
        metas.append(res["cellmeta"])
        r = res["receipt"]
        r.update({"gsm": lib["gsm"], "library_code": lib["library_code"], "files": files,
                  "ensembl_axis_equal_seaad": True, "symbol_axis_equal_seaad": sym_equal,
                  "author_qc_sheet": lib["qc_sheet"]})
        lib_receipts.append(r)
        if not keep_raw:
            shutil.rmtree(raw)
    X = sp.vstack(mats).tocsr()
    X.sort_indices()
    nm = np.concatenate(names).astype("<U64")
    cm = {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}
    (OUT1 / "counts").mkdir(parents=True, exist_ok=True)
    (OUT1 / "cellmeta").mkdir(parents=True, exist_ok=True)
    (OUT1 / "receipts").mkdir(parents=True, exist_ok=True)
    f = OUT1 / "counts" / f"{donor}.npz"
    n = OUT1 / "counts" / f"{donor}_names.npy"
    c = OUT1 / "cellmeta" / f"{donor}.npz"
    tmp = f.with_name(f.stem + ".partial.npz")
    sp.save_npz(tmp, X)
    tmp.replace(f)
    np.save(n, nm)
    tmpc = c.with_name(c.stem + ".partial.npz")
    np.savez_compressed(tmpc, **cm)
    tmpc.replace(c)
    # Verify by reloading.
    Y = sp.load_npz(f).tocsr()
    if Y.shape != X.shape or (Y != X).nnz or Y.dtype != np.int32:
        raise RuntimeError(f"{donor}: reload mismatch")
    if len(np.load(n)) != X.shape[0] or len(np.unique(nm)) != len(nm):
        raise RuntimeError(f"{donor}: names mismatch")
    umi = np.asarray(X.sum(1)).ravel()
    ng = np.diff(X.indptr)
    rec = {"donor": donor, "cell_count": int(X.shape[0]), "gene_count": int(X.shape[1]), "nnz": int(X.nnz),
           "median_umis": float(np.median(umi)), "median_genes": float(np.median(ng)),
           "counts_sha256": sha256(f), "names_sha256": sha256(n), "cellmeta_sha256": sha256(c),
           "genes_file": "counts/genes.npy", "libraries": lib_receipts, "seconds": round(time.time() - t0, 1),
           "selection_version": sel.SELECTION_VERSION}
    write_json(recp, rec)
    print(json.dumps({"donor": donor, "ADNC": dmeta["ADNC"], "cells": rec["cell_count"],
                      "median_umis": rec["median_umis"], "sec": rec["seconds"]}), flush=True)
    return rec


def cmd_libs(args):
    meta = json.loads((META / "donors_meta.json").read_text())
    libs = json.loads((META / "libraries.json").read_text())["libraries"]
    ids = np.load(META / "seaad_gene_ids.npy", allow_pickle=False)
    sym = np.load(REBUILT / "counts/genes.npy", allow_pickle=False)
    (OUT1 / "counts").mkdir(parents=True, exist_ok=True)
    gfile = OUT1 / "counts" / "genes.npy"
    if not gfile.exists():
        np.save(gfile, sym)
        np.save(OUT1 / "counts" / "gene_ids.npy", ids)
    if not np.array_equal(np.load(gfile), sym):
        raise RuntimeError("store genes.npy differs from SEA-AD rebuilt genes.npy")
    donors = sorted(meta["donors"])
    if args.donor:
        donors = [d for d in donors if d in args.donor]
    for d in donors:
        dm = meta["donors"][d]
        dl = [l for l in libs if l["gsm"] in dm["itg_libraries"]]
        if not dl:
            continue
        process_donor(d, dm, dl, ids, sym, keep_raw=args.keep_raw)
    cmd_donors(args)


def cmd_donors(args):
    import ext_select_microglia as sel
    meta = json.loads((META / "donors_meta.json").read_text())
    terms = json.loads((META / "access_terms.json").read_text()) if (META / "access_terms.json").exists() else {}
    out = {}
    for d, dm in sorted(meta["donors"].items()):
        recp = OUT1 / "receipts" / f"{d}.json"
        if not recp.exists():
            continue
        rec = json.loads(recp.read_text())
        if not verify_donor(d, rec):
            raise RuntimeError(f"{d}: files do not verify")
        e = dict(dm)
        e.update({"cells_kept": rec["cell_count"], "median_umis_per_cell": rec["median_umis"],
                  "median_genes_per_cell": rec["median_genes"], "counts_file": f"counts/{d}.npz",
                  "names_file": f"counts/{d}_names.npy", "cellmeta_file": f"cellmeta/{d}.npz",
                  "counts_sha256": rec["counts_sha256"], "names_sha256": rec["names_sha256"],
                  "selection_receipt": {"selection_version": rec["selection_version"],
                                        "libraries": [{k: v for k, v in l.items() if k != "files"} |
                                                      {"raw_files": l["files"]} for l in rec["libraries"]]}})
        out[d] = e
    stage_tab = {s: {"donors": sum(v["ADNC"] == s for v in out.values()),
                     "cells": sum(v["cells_kept"] for v in out.values() if v["ADNC"] == s)} for s in STAGES}
    doc = {"dataset": "GSE268599 MADRC ITG microglia (external TRAINING donors only)",
           "layout": "counts/{donor}.npz scipy CSR int32 cells x 36,601 (gene order = counts/genes.npy = SEA-AD "
                     "rebuilt/counts/genes.npy; Ensembl IDs in counts/gene_ids.npy); counts/{donor}_names.npy cell names "
                     "'{barcode}-{GSM}'; cellmeta/{donor}.npz per-cell selection scores",
           "selection": sel.SELECTION_SUMMARY, "selection_version": sel.SELECTION_VERSION,
           "age_coding": meta["age_coding"], "adnc_crosscheck": meta["adnc_crosscheck"],
           "access_terms": terms, "metadata_problems": meta["problems"],
           "stage_table": stage_tab, "donors": out}
    write_json(OUT1 / "donors.json", doc)
    print(json.dumps(stage_tab))


# ----------------------------------------------------------------------------- RDS reader
class RDSReader:
    """Minimal pure-Python reader for R serialization format 2/3 (XDR binary).

    Supports what dgCMatrix objects and data.frames need: vectors, strings,
    pairlists/attributes, S4 objects, symbols, references, environments and the
    common ALTREP classes (compact_intseq, compact_realseq, deferred_string,
    wrap_*). Returns plain Python/numpy values; attributes in RObj.attrs.
    """

    class RObj:
        def __init__(self, rtype, value=None, attrs=None):
            self.rtype, self.value, self.attrs = rtype, value, attrs or {}

        def __repr__(self):
            return f"RObj({self.rtype}, attrs={list(self.attrs)})"

    def __init__(self, data):
        self.b = data
        self.p = 0
        self.refs = []

    def _int(self):
        v = struct.unpack_from(">i", self.b, self.p)[0]
        self.p += 4
        return v

    def _len(self):
        n = self._int()
        if n == -1:
            hi, lo = self._int(), self._int()
            n = (hi << 32) + lo
        return n

    def _arr(self, dt, n):
        a = np.frombuffer(self.b, dtype=dt, count=n, offset=self.p)
        self.p += a.nbytes
        return a.astype(dt.newbyteorder("=")) if n else a

    def read(self):
        if self.b[:2] != b"X\n":
            raise ValueError("only XDR RDS supported")
        self.p = 2
        version = self._int()
        self._int()
        self._int()
        if version == 3:
            n = self._int()
            self.p += n
        return self.item()

    def _attrs(self, pl):
        out = {}
        if isinstance(pl, list):
            for tag, val in pl:
                out[tag] = val
        return out

    def _pairlist(self, flags):
        items = []
        while True:
            t = flags & 0xFF
            has_attr, has_tag = flags & (1 << 9), flags & (1 << 10)
            if has_attr:
                self.item()
            tag = self.item() if has_tag else None
            car = self.item()
            items.append((tag, car))
            flags = self._int()
            t = flags & 0xFF
            if t == 254:  # NILVALUE terminates
                return items
            if t not in (2, 239):  # cdr is not another pairlist cell
                self.p -= 4
                self.item()
                return items

    def item(self):
        flags = self._int()
        t = flags & 0xFF
        has_attr = flags & (1 << 9)
        if t == 254:
            return None
        if t in (253, 252, 251, 250, 242, 241):
            return self.RObj("special%d" % t)
        if t == 255:
            i = flags >> 8
            if i == 0:
                i = self._int()
            return self.refs[i - 1]
        if t in (247, 248, 249):
            self._int()
            n = self._int()
            v = [self.item() for _ in range(n)]
            o = self.RObj("persist", v)
            self.refs.append(o)
            return o
        if t == 1:  # SYMSXP
            o = self.item()
            self.refs.append(o)
            return o
        if t == 9:  # CHARSXP
            n = self._int()
            if n == -1:
                return None
            s = self.b[self.p:self.p + n].decode("utf8", "replace")
            self.p += n
            return s
        if t == 4:  # ENVSXP
            o = self.RObj("env")
            self.refs.append(o)
            self._int()
            o.value = [self.item(), self.item(), self.item()]
            o.attrs = self._attrs(self.item())
            return o
        if t in (2, 6, 3, 5, 17, 239, 240):
            return self._pairlist(flags)
        if t == 238:  # ALTREP
            info = self.item()
            state = self.item()
            attr = self.item()
            cls = info[0][1] if isinstance(info, list) else None
            if cls == "compact_intseq":
                n, start, inc = np.asarray(state, dtype=float)[:3]
                v = (start + inc * np.arange(int(n))).astype(np.int32)
            elif cls == "compact_realseq":
                n, start, inc = np.asarray(state, dtype=float)[:3]
                v = start + inc * np.arange(int(n))
            elif cls == "deferred_string":
                arg = state[0][1] if isinstance(state, list) else state
                arg = arg.value if isinstance(arg, self.RObj) else arg
                v = np.array([str(int(x)) if float(x).is_integer() else repr(float(x)) for x in np.asarray(arg)], dtype=object)
            elif cls and cls.startswith("wrap_"):
                v = state.value[0] if isinstance(state, self.RObj) else state[0]
                if isinstance(v, self.RObj):
                    v = v.value
            else:
                raise ValueError(f"unsupported ALTREP class {cls}")
            return self.RObj("altrep", v, self._attrs(attr)) if attr else v
        if t in (10, 13):
            n = self._len()
            v = self._arr(np.dtype(">i4"), n)
        elif t == 14:
            n = self._len()
            v = self._arr(np.dtype(">f8"), n)
        elif t == 15:
            n = self._len()
            v = self._arr(np.dtype(">f8"), 2 * n)
        elif t == 24:
            n = self._len()
            v = self.b[self.p:self.p + n]
            self.p += n
        elif t == 16:
            n = self._len()
            v = np.array([self.item() for _ in range(n)], dtype=object)
        elif t in (19, 20):
            n = self._len()
            v = [self.item() for _ in range(n)]
        elif t == 25:
            v = None
        else:
            raise ValueError(f"unsupported SEXP type {t} at {self.p}")
        attrs = self._attrs(self.item()) if has_attr else {}
        if attrs or t == 25:
            return self.RObj({10: "lgl", 13: "int", 14: "real", 15: "cplx", 16: "str", 19: "list",
                              20: "expr", 24: "raw", 25: "S4"}[t], v, attrs)
        return v


def read_rds(path):
    data = Path(path).read_bytes()
    while data[:2] == b"\x1f\x8b":  # GEO wraps an already gzip-compressed RDS in another gzip layer
        data = gzip.decompress(data)
    return RDSReader(data).read()


def _val(x):
    return x.value if isinstance(x, RDSReader.RObj) else x


def dgc_to_csr(obj):
    """dgCMatrix (genes x cells, column-compressed) -> cells x genes CSR, plus dimnames."""
    a = obj.attrs
    cls = _val(a.get("class"))
    if cls is None or "dgCMatrix" not in list(cls):
        raise ValueError(f"not a dgCMatrix: {cls}")
    G, C = map(int, _val(a["Dim"]))
    M = sp.csc_matrix((np.asarray(_val(a["x"])), np.asarray(_val(a["i"])), np.asarray(_val(a["p"]))), shape=(G, C))
    dn = _val(a["Dimnames"])
    genes = np.array(_val(dn[0]), dtype=str)
    cells = np.array(_val(dn[1]), dtype=str)
    return M.T.tocsr(), genes, cells


def rdf_to_dict(obj):
    a = obj.attrs
    names = list(_val(a["names"]))
    cols = {}
    for nm, col in zip(names, obj.value):
        v = _val(col)
        ca = col.attrs if isinstance(col, RDSReader.RObj) else {}
        if "levels" in ca:  # factor
            lev = np.array(_val(ca["levels"]), dtype=object)
            codes = np.asarray(v)
            v = np.array([lev[c - 1] if c > 0 else None for c in codes], dtype=object)
        cols[nm] = np.asarray(v)
    rn = _val(a.get("row.names"))
    return cols, rn


def cmd_gse264648(args):
    raw = OUT2 / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    got = {}
    for k in ["gse264648_counts", "gse264648_meta"]:
        dest = raw / Path(URLS[k]).name
        if not dest.exists():
            http_get(URLS[k], dest)
        got[k] = {"url": URLS[k], "bytes": dest.stat().st_size, "sha256": sha256(dest)}
    t0 = time.time()
    cnt = read_rds(raw / "GSE264648_counts_microglia.rds.gz")
    print("counts object", cnt, round(time.time() - t0, 1), flush=True)
    M, genes, cells = dgc_to_csr(cnt)
    md = read_rds(raw / "GSE264648_metadata_microglia.rds.gz")
    cols, rn = rdf_to_dict(md)
    print("metadata columns", list(cols), flush=True)
    if not args.write:
        for k, v in cols.items():
            u = np.unique(v.astype(str))
            print(k, len(u), u[:8])
        print("genes", genes[:5], len(genes), "cells", cells[:3], len(cells), "x dtype", M.dtype,
              "integer", bool(np.all(M.data == np.round(M.data))))
        return
    import ext_select_microglia as sel
    sel.write_gse264648(M, genes, cells, cols, rn, got, OUT2)


def cmd_terms_eutils(args):
    """Review finding D4: verify public status through E-utilities and FTP."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    out = {"checked_unix": time.time(), "client": UA, "series": {}}
    for acc, sub in [("GSE268599", "GSE268nnn"), ("GSE264648", "GSE264nnn")]:
        q = f"{base}/esearch.fcgi?db=gds&term={acc}[ACCN]+AND+gse[ETYP]&retmode=json"
        ids = json.loads(http_get(q))["esearchresult"]["idlist"]
        time.sleep(0.5)
        su = f"{base}/esummary.fcgi?db=gds&id={','.join(ids)}&retmode=json"
        res = json.loads(http_get(su))["result"]
        recs = [res[i] for i in res.get("uids", []) if res[i].get("accession") == acc]
        r = recs[0] if recs else {}
        ftp = f"{GEO_FTP}/series/{sub}/{acc}/suppl/"
        listing = page_text(http_get(ftp))
        out["series"][acc] = {
            "esearch_url": q, "esummary_url": su, "uid": r.get("uid"),
            "fields": {k: r.get(k) for k in ["accession", "entrytype", "gdstype", "pdat", "n_samples",
                                             "suppfile", "bioproject", "taxon", "title"]},
            "ftp_suppl_url": ftp, "ftp_suppl_listing_excerpt": listing[:600],
            "public": bool(r.get("pdat")) and r.get("accession") == acc}
        time.sleep(0.5)
    out["interpretation"] = ("E-utilities list both series as public GEO series with the release dates above, and "
                             "their supplementary files are served by anonymous FTP without login. Neither series "
                             "carries a licence field. The GEO disclaimer page text was NOT re-verified (reCAPTCHA "
                             "for scripts); the round-3 inventory quotes the older wording ('NCBI places no "
                             "restrictions on the use or distribution of the GEO data', with the caveat that "
                             "submitters may claim IP rights).")
    out["redistribution_note"] = ("ext/meta/serrano_pozo2024_MOESM3.xlsx is the Serrano-Pozo 2024 supplementary "
                                  "workbook, licensed with the article under CC BY-NC-ND 4.0: keep it INTERNAL, do "
                                  "not redistribute it or a modified version. The GEO counts are not under that "
                                  "licence.")
    p = META / "access_terms.json"
    doc = json.loads(p.read_text())
    doc["eutils_check"] = out
    write_json(p, doc)
    print(json.dumps({a: v["fields"] | {"public": v["public"]} for a, v in out["series"].items()}, indent=1))


def cmd_status(args):
    rec = sorted((OUT1 / "receipts").glob("*.json")) if (OUT1 / "receipts").exists() else []
    print(len(rec), "donor receipts")
    for p in rec:
        r = json.loads(p.read_text())
        print(r["donor"], r["cell_count"], r["median_umis"])


def main():
    sys.path.insert(0, str(HERE))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["meta", "libs", "gse264648", "donors", "status", "terms-eutils"])
    p.add_argument("--donor", nargs="*", help="restrict libs to these donor names (e.g. MADRC-01)")
    p.add_argument("--keep-raw", action="store_true", help="do not delete raw matrices (debugging only)")
    p.add_argument("--write", action="store_true", help="gse264648: write the store (default: inspect only)")
    a = p.parse_args()
    {"meta": cmd_meta, "libs": cmd_libs, "gse264648": cmd_gse264648, "donors": cmd_donors, "status": cmd_status,
     "terms-eutils": cmd_terms_eutils}[a.cmd](a)


if __name__ == "__main__":
    main()
