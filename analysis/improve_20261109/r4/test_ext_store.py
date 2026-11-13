"""Tests for the external donor stores (round 4).

Run: /mnt/nvme/ai/conda/anaconda3/bin/python3 -B test_ext_store.py
(plain asserts; no pytest needed). Checks:
  - ADNC table logic (NIA-AA 2012) and the MADRC cross-check counts
  - GSE268599 store integrity: shapes, CSR int32, gene axis identical to SEA-AD
    (symbols and Ensembl IDs), integer non-negative counts, no empty cells,
    unique names, donors.json consistent with files and hashes
  - rerunnable fetch: process_donor on a completed donor skips without network
  - GSE264648 store integrity
  - RDS reader round trip on a hand-built XDR stream
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ext_fetch as F  # noqa: E402

REB = F.REBUILT


def test_adnc():
    a = F.adnc_2012
    assert a(0, 3, 3) == "Not AD"
    assert a(1, 3, 3) == "Low"
    assert a(2, 1, 3) == "Low" and a(2, 2, 0) == "Intermediate" and a(2, 3, 3) == "Intermediate"
    assert a(3, 1, 1) == "Low" and a(3, 1, 2) == "Intermediate" and a(3, 2, 3) == "Intermediate"
    assert a(3, 3, 1) == "Intermediate" and a(3, 3, 2) == "High"
    assert F.braak_to_b("0") == 0 and F.braak_to_b("II") == 1 and F.braak_to_b("IV") == 2 and F.braak_to_b("V") == 3
    m = json.loads((F.META / "donors_meta.json").read_text())
    cc = m["adnc_crosscheck"]
    assert cc["match_all"] and cc["match_not_ad_itg"], cc
    assert len(m["donors"]) == 32


def test_store():
    out = F.OUT1
    doc = json.loads((out / "donors.json").read_text())
    genes = np.load(out / "counts/genes.npy", allow_pickle=False)
    assert np.array_equal(genes, np.load(REB / "counts/genes.npy", allow_pickle=False))
    ids = np.load(out / "counts/gene_ids.npy", allow_pickle=False)
    assert len(ids) == 36601 and len(set(ids)) == 36601
    assert np.array_equal(ids, np.load(F.META / "seaad_gene_ids.npy", allow_pickle=False))
    meta = json.loads((F.META / "donors_meta.json").read_text())
    expected = sorted(d for d, v in meta["donors"].items() if v["itg_libraries"])
    assert sorted(doc["donors"]) == expected, (len(doc["donors"]), len(expected))
    files = {p.name for p in (out / "counts").glob("*.npz")}
    assert files == {f"{d}.npz" for d in expected}, files ^ {f"{d}.npz" for d in expected}
    allnames = set()
    total = 0
    for d, e in doc["donors"].items():
        M = sp.load_npz(out / e["counts_file"])
        assert M.format == "csr" and M.dtype == np.int32 and M.shape[1] == 36601
        M.check_format(full_check=True)
        assert (M.data > 0).all()
        assert np.diff(M.indptr).min() > 0, f"{d}: empty cell"
        n = np.load(out / e["names_file"], allow_pickle=False)
        assert len(n) == M.shape[0] == e["cells_kept"]
        assert len(set(n)) == len(n) and not (allnames & set(n))
        allnames |= set(n)
        assert F.sha256(out / e["counts_file"]) == e["counts_sha256"]
        assert F.sha256(out / e["names_file"]) == e["names_sha256"]
        cm = np.load(out / e["cellmeta_file"], allow_pickle=False)
        assert all(len(cm[k]) == M.shape[0] for k in cm.files)
        umi = np.asarray(M.sum(1)).ravel()
        assert np.allclose(cm["umi"], umi)
        assert abs(np.median(umi) - e["median_umis_per_cell"]) < 1e-6
        assert e["ADNC"] == F.adnc_2012(e["A"], e["B"], e["C"])
        assert e["APOE4_dose"] == e["APOE"].count("4")
        assert (e["age"] == meta["age_coding"]["value"]) == e["age_90plus_censored"]
        for lib in e["selection_receipt"]["libraries"]:
            assert lib["ensembl_axis_equal_seaad"]
        total += M.shape[0]
    st = doc["stage_table"]
    assert sum(v["cells"] for v in st.values()) == total
    assert sum(v["donors"] for v in st.values()) == len(doc["donors"])
    assert not (out / "raw").exists() or not any((out / "raw").iterdir()), "raw matrices left on disk"


def test_rerun_skips():
    doc = json.loads((F.OUT1 / "donors.json").read_text())
    d = sorted(doc["donors"])[0]
    rec = json.loads((F.OUT1 / "receipts" / f"{d}.json").read_text())

    def boom(*a, **k):
        raise AssertionError("network used for a completed donor")
    orig = F.http_get
    F.http_get = boom
    try:
        meta = json.loads((F.META / "donors_meta.json").read_text())
        libs = json.loads((F.META / "libraries.json").read_text())["libraries"]
        dl = [l for l in libs if l["gsm"] in meta["donors"][d]["itg_libraries"]]
        r = F.process_donor(d, meta["donors"][d], dl, None, None)
    finally:
        F.http_get = orig
    assert r == rec


def test_gse264648():
    out = F.OUT2
    doc = json.loads((out / "donors.json").read_text())
    genes = np.load(out / "counts/genes.npy", allow_pickle=False)
    assert np.array_equal(genes, np.load(REB / "counts/genes.npy", allow_pickle=False))
    present = np.load(out / "counts/genes_present.npy")
    assert present.sum() == doc["gene_axis"]["mapped_to_36601_by_symbol"]
    for d, e in doc["donors"].items():
        M = sp.load_npz(out / "counts" / f"{d}.npz")
        assert M.format == "csr" and M.dtype == np.int32 and M.shape == (e["cells_kept"], 36601)
        assert (M.data > 0).all() and np.diff(M.indptr).min() > 0
        assert M[:, ~present].nnz == 0
        assert len(np.load(out / "counts" / f"{d}_names.npy")) == M.shape[0]
    assert len(doc["donors"]) == 15


def test_rds_reader():
    # list(a = 1:3 (int), b = c(1.5, NA)) serialized as XDR v3 by hand.
    def i(x):
        return struct.pack(">i", x)
    b = b"X\n" + i(3) + i(0x040300) + i(0x030500) + i(5) + b"UTF-8"
    b += i(19 | (1 << 9)) + i(2)
    b += i(13) + i(3) + b"".join(i(v) for v in [1, 2, 3])
    b += i(14) + i(2) + struct.pack(">d", 1.5) + struct.pack(">d", float("nan"))
    b += i(2 | (1 << 10)) + i(1) + i(9) + i(5) + b"names"
    b += i(16) + i(2) + i(9) + i(1) + b"a" + i(9) + i(1) + b"b"
    b += i(254)
    o = F.RDSReader(b).read()
    assert list(o.value[0]) == [1, 2, 3] and o.value[1][0] == 1.5
    assert list(o.attrs["names"]) == ["a", "b"]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # report all
                fails += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
