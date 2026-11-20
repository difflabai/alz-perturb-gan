"""State tests for HypoAD extraction, the HypoAD loader and PsychAD receipts.

(g) extract: source states, a kill mid-write leaves nothing at --output,
    CORRUPT / STALE / STALE_UNKNOWN extracts, and a malformed extract that
    makes the readout FAIL instead of skipping
(h) PsychAD receipt ordering under crash injection, and the
    psychad_programs TOCTOU guard

The pure-state tests need neither anndata nor h5py; the rest importorskip.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

PW = Path(__file__).resolve().parent.parent
SCRIPTS = PW / "scripts"
if str(PW) not in sys.path:
    sys.path.insert(0, str(PW))

from alztest import data  # noqa: E402
from alztest.data import DataMalformed, DataUnavailable  # noqa: E402


def _script(name):
    spec = importlib.util.spec_from_file_location(f"_pw_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _receipt(path: Path) -> str:
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    Path(f"{path}.receipt.json").write_text(json.dumps(
        {"url": "u", "etag": '"v1"', "size": len(raw), "sha256": sha,
         "finished_at": "2026-11-20T00:00:00Z"}))
    return sha


@pytest.fixture
def hypoad_env(tmp_path, monkeypatch):
    dst = tmp_path / "hypoad_microglia.h5ad"
    src = tmp_path / "GSE342635_HypoAD.h5ad"
    monkeypatch.setenv("HYPOAD_H5AD", str(dst))
    monkeypatch.setenv("HYPOAD_SOURCE", str(src))
    monkeypatch.delenv("HYPOAD_STRICT", raising=False)
    return src, dst


# ------------------------------------------------------------------ pure state

def test_data_malformed_is_not_a_data_unavailable():
    assert not issubclass(DataMalformed, DataUnavailable)
    assert not issubclass(DataUnavailable, DataMalformed)


def test_download_suffixes_match_the_downloader():
    dl = _script("parallel_download")
    assert (dl.STATE_SUFFIX, dl.RECEIPT_SUFFIX) == (data.DOWNLOAD_STATE_SUFFIX,
                                                    data.DOWNLOAD_RECEIPT_SUFFIX)


def test_source_states(tmp_path):
    src = tmp_path / "obj.h5ad"
    assert data.hypoad_source_state(src) == ("NO_SOURCE", None)
    Path(f"{src}.partial").write_bytes(b"abc")
    assert data.hypoad_source_state(src)[0] == "SOURCE_INCOMPLETE"
    os.remove(f"{src}.partial")
    src.write_bytes(b"0123456789")
    assert data.hypoad_source_state(src)[0] == "SOURCE_INCOMPLETE"  # no receipt
    sha = _receipt(src)
    assert data.hypoad_source_state(src) == ("SOURCE_DONE", json.loads(
        Path(f"{src}.receipt.json").read_text()))
    assert data.download_receipt(src)["sha256"] == sha
    Path(f"{src}.state.json").write_text(json.dumps({"state": "ASSEMBLING"}))
    assert data.hypoad_source_state(src)[0] == "SOURCE_INCOMPLETE"  # state wins
    Path(f"{src}.state.json").write_text(json.dumps({"state": "DONE"}))
    assert data.hypoad_source_state(src)[0] == "SOURCE_DONE"
    src.write_bytes(b"01234")  # truncated after the receipt
    assert data.hypoad_source_state(src)[0] == "SOURCE_INCOMPLETE"


def test_extract_provenance_states():
    good = {"schema": 2, "params": dict(data.HYPOAD_EXPECTED_PARAMS), "input_sha256": "ab"}
    assert data.hypoad_extract_state(None)[0] == "STALE_UNKNOWN"
    assert data.hypoad_extract_state({"source": "x"})[0] == "STALE_UNKNOWN"
    assert data.hypoad_extract_state(good) == ("EXTRACTED", [])
    assert data.hypoad_extract_state(good, {"sha256": "ab"}) == ("EXTRACTED", [])
    assert data.hypoad_extract_state(good, {"sha256": "cd"})[0] == "STALE"
    other = dict(good, params=dict(good["params"], pattern="(?i)^micro"))
    assert data.hypoad_extract_state(other)[0] == "STALE"
    loose = dict(good, params=dict(good["params"], allow_noninteger=np.bool_(True)))
    assert data.hypoad_extract_state(loose)[0] == "STALE"


def test_loader_missing_skips_directory_fails(hypoad_env):
    src, dst = hypoad_env
    with pytest.raises(DataUnavailable, match="NO_SOURCE"):
        data.load_hypoad_microglia()
    Path(f"{src}.partial").write_bytes(b"x")
    with pytest.raises(DataUnavailable, match="SOURCE_INCOMPLETE"):
        data.load_hypoad_microglia()
    dst.mkdir()
    with pytest.raises(DataMalformed, match="not a regular file"):
        data.load_hypoad_microglia()
    dst.rmdir()
    dst.symlink_to(dst.parent / "nowhere.h5ad")
    with pytest.raises(DataMalformed, match="not a regular file"):
        data.load_hypoad_microglia()


def test_extractor_refuses_sources_that_are_not_done(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("pandas")
    ext = _script("extract_hypoad_microglia")
    src = tmp_path / "src.h5ad"
    with pytest.raises(SystemExit, match="NO_SOURCE"):
        ext.main(["--input", str(src), "--output", str(tmp_path / "o.h5ad")])
    src.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 100)  # a prefix still downloading
    Path(f"{src}.state.json").write_text(json.dumps({"state": "FETCHING"}))
    with pytest.raises(SystemExit, match="SOURCE_INCOMPLETE.*--input-unverified"):
        ext.main(["--input", str(src), "--output", str(tmp_path / "o.h5ad")])
    assert not (tmp_path / "o.h5ad").exists()


# ------------------------------------------------------------------ anndata fixtures

def _write_source(path: Path, n=120, genes=20, seed=0):
    ad = pytest.importorskip("anndata")
    import pandas as pd
    from scipy import sparse
    rng = np.random.default_rng(seed)
    labels = np.array(["Microglia", "Astrocyte", "Micro_2"])[rng.integers(0, 3, n)]
    counts = sparse.random(n, genes, density=0.3, random_state=seed, format="csr") * 10
    counts.data = np.ceil(counts.data)
    obs = pd.DataFrame({"cell_type": pd.Categorical(labels),
                        "donor_id": pd.Categorical(rng.choice(["AD_44", "AD_1", "C_1"], n)),
                        "diagnosis": pd.Categorical(rng.choice(["AD", "Control"], n))},
                       index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"G{i}" for i in range(genes)])
    a = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
    a.layers["counts"] = counts
    a.write_h5ad(path)


def _readout_extract(path: Path, *, symbols=True, condition=True, provenance=True):
    """A synthetic extract shaped like the real one (donors, groups, module genes)."""
    ad = pytest.importorskip("anndata")
    import pandas as pd
    from scipy import sparse
    from alztest import hypoad
    rng = np.random.default_rng(0)
    genes = sorted(hypoad.HOMEO | hypoad.DAM) + [f"G{i}" for i in range(10)]
    names = genes if symbols else [f"ENSG{i:011d}" for i in range(len(genes))]
    donors = [f"CTL_{i}" for i in range(7)] + [f"AD_{i}" for i in range(40, 47)]
    cells = [d for d in donors for _ in range(5)]
    obs = {"donor": cells, "cell_type": "Microglia"}
    if condition:
        obs["condition"] = ["Control" if d.startswith("CTL") else "AD" for d in cells]
    a = ad.AnnData(X=sparse.csr_matrix(rng.poisson(2.0, (len(cells), len(genes))).astype(np.float32)),
                   obs=pd.DataFrame(obs, index=[f"c{i}" for i in range(len(cells))]),
                   var=pd.DataFrame(index=names))
    if provenance:
        a.uns["hypoad_extract"] = {"schema": 2, "params": dict(data.HYPOAD_EXPECTED_PARAMS),
                                   "input_sha256": "", "input_unverified": "fixture"}
    a.write_h5ad(path)


def _readout_module():
    pytest.importorskip("particlegan")
    spec = importlib.util.spec_from_file_location(
        "_hypoad_readout_state", PW / "tests" / "test_hypoad_microglia_readout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ (g) extract

def test_extract_killed_mid_write_leaves_no_output(hypoad_env, tmp_path):
    pytest.importorskip("anndata")
    src, dst = hypoad_env
    _write_source(src)
    _receipt(src)
    code = f"""
import os, sys
sys.path.insert(0, {str(SCRIPTS)!r})
import anndata
def dying_write(self, path, *a, **k):
    with open(path, "wb") as fh:
        fh.write(b"\\x89HDF\\r\\n\\x1a\\n half of an h5ad")
    os._exit(137)
anndata.AnnData.write_h5ad = dying_write
import extract_hypoad_microglia as ext
ext.main(["--input", {str(src)!r}, "--output", {str(dst)!r}])
"""
    rc = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert rc.returncode == 137, rc.stderr
    assert not dst.exists()
    leftovers = list(tmp_path.glob(".hypoad_microglia.h5ad.*.partial"))
    assert len(leftovers) == 1  # hidden, pid-named, swept by age
    with pytest.raises(DataUnavailable, match="SOURCE_DONE"):
        data.load_hypoad_microglia()  # the readout skips: nothing was extracted


def test_extract_records_provenance_and_is_idempotent(hypoad_env, capsys):
    pytest.importorskip("anndata")
    ext = _script("extract_hypoad_microglia")
    src, dst = hypoad_env
    _write_source(src)
    sha = _receipt(src)
    args = ["--input", str(src), "--output", str(dst)]
    assert ext.main(args) == 0
    adata = data.load_hypoad_microglia()
    info = adata.uns["hypoad_extract"]
    assert info["schema"] == 2 and info["input_sha256"] == sha
    stamp = dst.stat().st_mtime_ns
    capsys.readouterr()
    assert ext.main(args) == 0
    assert "already EXTRACTED" in capsys.readouterr().out and dst.stat().st_mtime_ns == stamp
    # A new download (different receipt) makes the old extract STALE.
    _write_source(src, seed=1)
    _receipt(src)
    with pytest.raises(DataMalformed, match="STALE.*input_sha256"):
        data.load_hypoad_microglia()
    assert ext.main(args) == 0  # re-extracts
    assert data.load_hypoad_microglia().n_obs > 0
    # Non-default parameters are recorded and the loader calls them STALE.
    assert ext.main(args + ["--pattern", r"(?i)^microglia$"]) == 0
    with pytest.raises(DataMalformed, match="STALE.*pattern"):
        data.load_hypoad_microglia()


def test_extract_unverified_input_records_the_reason(hypoad_env):
    ad = pytest.importorskip("anndata")
    ext = _script("extract_hypoad_microglia")
    src, dst = hypoad_env
    _write_source(src)
    assert ext.main(["--input", str(src), "--output", str(dst),
                     "--input-unverified", "copied from a colleague"]) == 0
    info = ad.read_h5ad(dst).uns["hypoad_extract"]
    assert info["input_unverified"] == "copied from a colleague" and info["input_sha256"] == ""
    assert data.load_hypoad_microglia().n_obs > 0


def test_corrupt_and_unknown_provenance_extracts(hypoad_env):
    pytest.importorskip("anndata")
    _, dst = hypoad_env
    dst.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 64)  # a torn write from an old extractor
    with pytest.raises(DataMalformed, match="CORRUPT"):
        data.load_hypoad_microglia()
    _readout_extract(dst, provenance=False)  # an extract from before schema 2
    with pytest.warns(UserWarning, match="schema-2"):
        assert data.load_hypoad_microglia().n_obs > 0
    os.environ["HYPOAD_STRICT"] = "1"
    try:
        with pytest.raises(DataMalformed, match="STALE_UNKNOWN"):
            data.load_hypoad_microglia()
    finally:
        del os.environ["HYPOAD_STRICT"]


@pytest.mark.parametrize("problem", ["gene_column", "condition_column"])
def test_malformed_extract_fails_the_readout_instead_of_skipping(hypoad_env, problem):
    pytest.importorskip("anndata")
    _, dst = hypoad_env
    _readout_extract(dst, symbols=problem != "gene_column",
                     condition=problem != "condition_column")
    mod = _readout_module()
    with pytest.raises(DataMalformed):
        mod._prepare()
    # The test itself must raise (fail), not call pytest.skip.
    with pytest.raises(DataMalformed):
        mod.test_hypoad_microglia_readout()


def test_missing_extract_still_skips_the_readout(hypoad_env):
    mod = _readout_module()
    with pytest.raises(pytest.skip.Exception):
        mod.test_hypoad_microglia_readout()


# ------------------------------------------------------------------ (h) PsychAD receipts

def _psychad_matrix(seed: int, n=30):
    ad = pytest.importorskip("anndata")
    import pandas as pd
    from scipy import sparse
    rng = np.random.default_rng(seed)
    a = ad.AnnData(X=sparse.csr_matrix(rng.poisson(1.0, (n, 8)).astype(np.float32)),
                   obs=pd.DataFrame({"donor_id": [f"d{i % 5}" for i in range(n)],
                                     "diagnosis": ["control"] * n},
                                    index=[f"s{seed}c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"ENSG{i:011d}" for i in range(8)]))
    return a


def _psychad_write(path: Path, seed: int, n: int, fault: str | None = None) -> int:
    code = f"""
import sys
sys.path.insert(0, {str(PW)!r})
sys.path.insert(0, {str(PW / 'tests')!r})
import importlib.util
spec = importlib.util.spec_from_file_location("fp", {str(SCRIPTS / 'fetch_psychad.py')!r})
fp = importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
from test_extract_state import _psychad_matrix
from pathlib import Path
fp.write_with_receipt(_psychad_matrix({seed}, {n}), Path({str(path)!r}),
                      {{"source": "fixture", "query": "q", "groups": {{"a b || c": "x"}}}})
"""
    env = {**os.environ}
    env.pop("FETCH_PSYCHAD_FAULT", None)
    if fault:
        env["FETCH_PSYCHAD_FAULT"] = fault
    return subprocess.run([sys.executable, "-B", "-c", code], env=env,
                          capture_output=True, text=True).returncode


@pytest.mark.parametrize("fault", [None, "after_partial", "after_pending", "after_h5ad_replace"])
@pytest.mark.parametrize("first", [True, False])
def test_psychad_receipt_ordering_under_crashes(tmp_path, fault, first):
    pytest.importorskip("anndata")
    path = tmp_path / "psychad-mssm-microglia.h5ad"
    if not first:
        assert _psychad_write(path, seed=1, n=30) == 0
    rc = _psychad_write(path, seed=2, n=40, fault=fault)
    assert rc == (0 if fault is None else 137)
    if first and fault in ("after_partial", "after_pending"):
        with pytest.raises(DataUnavailable):
            data.load_psychad(path, verify=True, full_sha=True)
        return
    adata = data.load_psychad(path, verify=True, full_sha=True)  # never a mismatched pair
    new = fault in (None, "after_h5ad_replace")
    assert adata.n_obs == (40 if new else 30)
    assert adata.obs_names[0] == ("s2c0" if new else "s1c0")
    receipt, pending = data.psychad_receipt_paths(path)
    assert pending.exists() == (fault in ("after_pending", "after_h5ad_replace"))
    inner = adata.uns["psychad_receipt"]
    assert json.loads(inner["groups"]) == {"a b || c": "x"}
    assert inner["content_sha256"] == data.psychad_content_sha256(adata)


def test_psychad_tampered_file_and_legacy_receipt(tmp_path):
    ad = pytest.importorskip("anndata")
    path = tmp_path / "psychad-mssm-microglia.h5ad"
    assert _psychad_write(path, seed=1, n=30) == 0
    # Replaced behind the receipt's back: the stamp no longer matches.
    _psychad_matrix(3, 30).write_h5ad(path)
    with pytest.raises(DataMalformed, match="does not match its receipt"):
        data.load_psychad(path, verify=True)
    assert data.load_psychad(path).n_obs == 30  # verify is opt-in
    # Legacy layout: no uns receipt, sidecar with sha256 and shape only.
    receipt, pending = data.psychad_receipt_paths(path)
    receipt.write_text(json.dumps({"sha256": data.file_sha256(path), "shape": [30, 8]}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert data.load_psychad(path, verify=True).n_obs == 30
    assert any("stamp" in str(w.message) for w in caught)
    assert data.load_psychad(path, verify=True, full_sha=True).n_obs == 30
    receipt.write_text(json.dumps({"sha256": "0" * 64, "shape": [30, 8]}))
    with pytest.raises(DataMalformed):
        data.load_psychad(path, verify=True, full_sha=True)
    receipt.unlink()
    with pytest.raises(DataMalformed, match="no receipt"):
        data.load_psychad(path, verify=True)
    del ad


def test_psychad_programs_refuses_an_input_that_changes_while_loading(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    pytest.importorskip("particlegan")
    from alztest import psychad_programs as pp
    path = tmp_path / "p.h5ad"
    _psychad_matrix(1).write_h5ad(path)
    _, sha = pp.load_pinned(path)
    assert sha == data.file_sha256(path)
    real = pp.load_psychad

    def swapping_load(p):
        adata = real(p)
        _psychad_matrix(2).write_h5ad(p)  # replaced while (just after) it was read
        return adata
    monkeypatch.setattr(pp, "load_psychad", swapping_load)
    with pytest.raises(SystemExit, match="changed while it was being loaded"):
        pp.load_pinned(path)
