"""Data-free checks for the HypoAD readout pipeline (synthetic fixtures only)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from paper_watch.alztest import hypoad

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _script(name):
    spec = importlib.util.spec_from_file_location(f"_hypoad_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ donor selection

def test_ad44_and_young_donors_are_excluded_before_grouping():
    donors = ["AD_43", "AD_44", "CTL_1", "Y_1", "CTL_2"]
    cond = ["AD", "AD", "Control", "Young control", "Control"]
    keep, group = hypoad.select_donors(donors, cond)
    assert keep.tolist() == [True, False, True, False, True]
    assert group[keep].tolist() == ["AD", "CTRL", "CTRL"]


def test_exclusion_matches_punctuation_variants_and_fails_when_missing():
    keep, _ = hypoad.select_donors(["ad-44", "AD_1", "C_1"], ["AD", "AD", "Control"])
    assert keep.tolist() == [False, True, True]
    with pytest.raises(ValueError, match="AD_44"):
        hypoad.select_donors(["AD_1", "C_1"], ["AD", "Control"])


def test_condition_labels():
    got = hypoad.classify_condition(["AD", "sporadic_AD", "Control", "CTL", "non-AD",
                                     "Young", "ADNC_high"])
    assert got.tolist() == ["AD", "AD", "CTRL", "CTRL", "CTRL", "OTHER", "OTHER"]
    got = hypoad.classify_condition(["HC", "NC", "Healthy", "normal", "Cognitively normal"])
    assert got.tolist() == ["CTRL"] * 5


def test_too_few_donors_fails_with_the_raw_labels():
    donors = ["A1", "A2", "C1", "C2", "C3"]
    cond = ["AD", "AD", "Unaffected", "Unaffected", "Unaffected"]
    group = hypoad.classify_condition(cond)
    with pytest.raises(ValueError, match=r"0 CTRL.*'Unaffected'->OTHER.: 3"):
        hypoad.check_groups(donors, group, cond)
    hypoad.check_groups(donors, hypoad.classify_condition(["AD", "AD", "Control", "HC", "NC"]),
                        cond)


# ------------------------------------------------------------ fit-only basis

def _synthetic(seed=0, n=300, genes=60):
    rng = np.random.default_rng(seed)
    X = rng.poisson(rng.gamma(2.0, 1.0, size=genes), size=(n, genes)).astype(float)
    fit = np.zeros(n, dtype=bool)
    fit[:120] = True
    return X, fit


def test_transform_is_fit_on_fit_cells_only():
    X, fit = _synthetic()
    Xl = hypoad.log_normalize(X)
    mod = [0, 1, 2, 3]
    t = hypoad.fit_feature_transform(Xl, fit, mod, n_top=20, n_comp=5)

    rng = np.random.default_rng(1)
    X2 = X.copy()
    X2[~fit] = rng.poisson(50.0, size=X2[~fit].shape) * (rng.random(X2[~fit].shape) < 0.5)
    X2[~fit, 40:] *= 20  # make different genes the most variable among held-out cells
    Xl2 = hypoad.log_normalize(X2)
    t2 = hypoad.fit_feature_transform(Xl2, fit, mod, n_top=20, n_comp=5)
    for name in ("cols", "mod_pos", "gene_mu", "gene_sd", "pca_mean", "comps",
                 "proj_mu", "proj_sd"):
        np.testing.assert_array_equal(getattr(t, name), getattr(t2, name), err_msg=name)

    Xp, Z = hypoad.apply_feature_transform(t, Xl)
    Xp2, Z2 = hypoad.apply_feature_transform(t2, Xl2)
    np.testing.assert_array_equal(Xp[fit], Xp2[fit])
    assert set(Z) == set(mod)
    np.testing.assert_allclose(Xp[fit].mean(0), 0, atol=1e-8)
    np.testing.assert_allclose(Xp[fit].std(0), 1, atol=1e-6)
    assert not np.allclose(Xp[~fit], Xp2[~fit])


def test_transform_keeps_module_genes_and_n_top_others():
    X, fit = _synthetic()
    t = hypoad.fit_feature_transform(hypoad.log_normalize(X), fit, [5, 3], n_top=10, n_comp=4)
    assert len(t.cols) == 12 and {3, 5} <= set(t.cols.tolist())
    assert t.comps.shape == (4, 12)


# ------------------------------------------------------------ donor statistics

def test_exact_permutation_p_and_bootstrap_ci():
    scores = [1, 2, 3, 4, 5, 6, 7]
    labels = [0, 0, 0, 1, 1, 1, 1]
    p, n, exact = hypoad.permutation_p(scores, labels)
    assert exact and n == 35 and p == pytest.approx(1 / 35)
    p_null, _, _ = hypoad.permutation_p([1, 1, 1, 1, 1, 1, 1], labels)
    assert p_null == 1.0
    lo, hi = hypoad.donor_bootstrap_ci(scores, labels, n_boot=200)
    assert lo == hi == 1.0
    lo, hi = hypoad.donor_bootstrap_ci([1, 5, 3, 4, 2, 6, 7], labels, n_boot=500)
    assert 0 <= lo < hi <= 1


# ------------------------------------------------------------ extractor

def _receipt(path):
    """The DONE receipt parallel_download.py leaves beside a finished download."""
    data = Path(path).read_bytes()
    Path(f"{path}.receipt.json").write_text(json.dumps({
        "url": "file://fixture", "etag": None, "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "finished_at": "2026-11-20T00:00:00Z"}))
    return hashlib.sha256(data).hexdigest()


def _write_source(path, *, source="counts"):
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(0)
    n, genes = 200, 25
    labels = np.array(["Microglia", "Astrocyte", "Oligodendrocyte", "Micro_2",
                       "Microvascular", "Micro-PVM"])[rng.integers(0, 6, n)]
    counts = sparse.random(n, genes, density=0.3, random_state=1, format="csr") * 10
    counts.data = np.ceil(counts.data)
    obs = pd.DataFrame({
        "cell_type": pd.Categorical(labels),
        "donor_id": pd.Categorical(rng.choice(["AD_44", "AD_1", "C_1"], n)),
        "diagnosis": pd.Categorical(rng.choice(["AD", "Control"], n)),
        "n_genes": rng.integers(100, 900, n),
    }, index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame({"feature_name": [f"G{i}" for i in range(genes)]},
                       index=[f"ENSG{i:011d}" for i in range(genes)])
    src = ad.AnnData(X=sparse.csr_matrix(np.log1p(counts.toarray())), obs=obs, var=var)
    if source == "counts":
        src.layers["counts"] = counts
    elif source == "raw":
        src.raw = ad.AnnData(X=counts, obs=obs, var=var)
    src.write_h5ad(path)
    return labels, counts


@pytest.mark.parametrize("source", ["counts", "raw", "X"])
def test_extractor_writes_only_microglia_rows(tmp_path, source):
    ad = pytest.importorskip("anndata")
    ext = _script("extract_hypoad_microglia")
    labels, counts = _write_source(tmp_path / "src.h5ad", source=source)
    sha = _receipt(tmp_path / "src.h5ad")
    dst = tmp_path / "out" / "hypoad_microglia.h5ad"
    args = ["--input", str(tmp_path / "src.h5ad"), "--output", str(dst)]
    if source == "X":  # X holds log1p values, not counts
        with pytest.raises(SystemExit, match="not integer-valued"):
            ext.main(args)
        assert not dst.exists()
        args.append("--allow-noninteger")
    rc = ext.main(args)
    assert rc == 0
    out = ad.read_h5ad(dst)
    want = np.flatnonzero(np.isin(labels, ["Microglia", "Micro_2"]))
    assert out.n_obs == len(want)
    assert set(out.obs["cell_type"]) == {"Microglia", "Micro_2"}
    assert list(out.obs.columns) == ["donor", "condition", "cell_type"]
    assert out.obs_names.tolist() == [f"c{i}" for i in want]
    assert out.var_names.tolist()[:2] == ["G0", "G1"]
    expected = counts[want] if source != "X" else np.log1p(counts[want].toarray())
    np.testing.assert_allclose(out.X.toarray(), np.asarray(
        expected.toarray() if sparse.issparse(expected) else expected), rtol=1e-6)
    info = out.uns["hypoad_extract"]
    assert info["matrix"] == {"counts": "layers/counts", "raw": "raw/X", "X": "X"}[source]
    assert list(info["labels_excluded"]) == ["Micro-PVM"]
    assert bool(info["integer_counts"]) == (source != "X")
    assert sum(info["condition_cells"]) == len(want)
    assert info["schema"] == 2 and info["input_sha256"] == sha and info["input_unverified"] == ""
    assert bool(info["params"]["allow_noninteger"]) == (source == "X")
    assert not list(dst.parent.glob(".*partial"))


def test_microglia_label_guard():
    ext = _script("extract_hypoad_microglia")
    labels = pd.Series(["Microglia", "Micro_2", "MG", "Micro-PVM", "Microglia/Macrophage",
                        "Micro/Macro", "Micro_BAM", "Microvascular", "Monocyte", "Astro"])
    keep = ext.microglia_mask(labels, ext.MICROGLIA, ext.NOT_MICROGLIA)
    assert labels[keep].tolist() == ["Microglia", "Micro_2", "MG"]


def test_extractor_defaults_resolve_from_file_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ext = _script("extract_hypoad_microglia")
    assert ext.DEFAULT_OUTPUT == SCRIPTS.parent / "data" / "hypoad_microglia.h5ad"


def test_row_window_reader_matches_scipy(tmp_path):
    h5py = pytest.importorskip("h5py")
    ext = _script("extract_hypoad_microglia")
    m = sparse.random(97, 13, density=0.2, random_state=3, format="csr")
    with h5py.File(tmp_path / "m.h5", "w") as f:
        g = f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["shape"] = m.shape
        for k in ("data", "indices", "indptr"):
            g.create_dataset(k, data=getattr(m, k))
        f.create_dataset("D", data=m.toarray())
    rows = np.array([0, 4, 5, 30, 31, 32, 96])
    with h5py.File(tmp_path / "m.h5", "r") as f:
        np.testing.assert_array_equal(ext.read_rows(f["X"], rows, window=7).toarray(),
                                      m[rows].toarray())
        np.testing.assert_array_equal(ext.read_rows(f["D"], rows, window=2).toarray(),
                                      m[rows].toarray())


# ------------------------------------------------------------ download tools

PAYLOAD = bytes(range(256)) * 8


ETAG = '"v1"'


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("ETag", ETAG)
        super().end_headers()

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/error"):
            lo, hi = map(int, self.headers["Range"].split("=")[1].split("-"))
            body = b"x" * (hi - lo + 1)  # error page exactly the expected size
            self.send_response(403)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        rng = self.headers.get("Range")
        if self.path.startswith("/norange") or self.path.startswith("/ignorerange") or not rng:
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)
            return
        lo, hi = map(int, rng.split("=")[1].split("-"))
        if self.path.startswith("/wrongstart"):
            lo = lo + 1  # a shifted range: right length class, wrong offset
        body = PAYLOAD[lo:hi + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(PAYLOAD)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.path.startswith("/truncate") and hi not in _TRUNCATED and len(body) > 1:
            _TRUNCATED.add(hi)  # the first request for each part drops half-way
            self.wfile.write(body[:len(body) // 2])
            self.close_connection = True
            return
        self.wfile.write(body)


_TRUNCATED: set = set()


@pytest.fixture
def server(monkeypatch):
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


CURL_FAST = ["curl", "--fail", "--silent", "--show-error", "--location"]


def test_parallel_download_assembles_ranges(tmp_path, server):
    dl = _script("parallel_download")
    out = tmp_path / "f.bin"
    out.write_bytes(PAYLOAD[:100])  # resume from an existing prefix (old-tool layout)
    argv = [f"{server}/f", str(out), str(len(PAYLOAD)), "5"]
    assert dl.main(argv, curl=CURL_FAST) == 1  # legacy bytes need --sha256 or --allow-unverified
    rc = dl.main(argv + ["--sha256", hashlib.sha256(PAYLOAD).hexdigest()], curl=CURL_FAST)
    assert rc == 0 and out.read_bytes() == PAYLOAD
    assert not list(tmp_path.glob("f.bin.part*"))


def test_parallel_download_refuses_http_error_body(tmp_path, server):
    dl = _script("parallel_download")
    assert "--fail" in dl.CURL and "--retry" in dl.CURL
    out = tmp_path / "f.bin"
    rc = dl.main([f"{server}/error", str(out), str(len(PAYLOAD)), "4"], curl=CURL_FAST)
    assert rc == 1
    assert not out.exists()
    assert not list(tmp_path.glob("f.bin.part*"))


@pytest.mark.parametrize("path", ["ignorerange", "wrongstart"])
def test_parallel_download_refuses_responses_that_ignore_the_range(tmp_path, server, path):
    dl = _script("parallel_download")
    out = tmp_path / "f.bin"
    rc = dl.main([f"{server}/{path}", str(out), str(len(PAYLOAD)), "4"], curl=CURL_FAST)
    assert rc == 1
    assert not out.exists()
    assert not list(tmp_path.glob("f.bin.part*"))


def test_parallel_download_resumes_a_truncated_part(tmp_path, server):
    dl = _script("parallel_download")
    _TRUNCATED.clear()
    out = tmp_path / "f.bin"
    rc = dl.main([f"{server}/truncate", str(out), str(len(PAYLOAD)), "3"], curl=CURL_FAST)
    assert rc == 0 and out.read_bytes() == PAYLOAD
    assert len(_TRUNCATED) == 3  # each part was cut once and then resumed
    left = sorted(p.name for p in tmp_path.glob("f.bin.*"))
    assert left == ["f.bin.lock", "f.bin.receipt.json", "f.bin.state.json",
                    "f.bin.state.json.journal.jsonl", "f.bin.state.json.lock"]
    receipt = json.loads((tmp_path / "f.bin.receipt.json").read_text())
    assert receipt["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    assert receipt["etag"] == ETAG and receipt["size"] == len(PAYLOAD)
    # A rerun of a finished download is a no-op that succeeds.
    assert dl.main([f"{server}/truncate", str(out), str(len(PAYLOAD)), "3"], curl=CURL_FAST) == 0


def test_parallel_download_refuses_a_changed_layout(tmp_path, server):
    dl = _script("parallel_download")
    out = tmp_path / "f.bin"
    (tmp_path / "f.bin.plan").write_text('{"url": "x", "total": 1, "jobs": []}')
    rc = dl.main([f"{server}/f", str(out), str(len(PAYLOAD)), "4"], curl=CURL_FAST)
    assert rc == 1 and not out.exists()


def test_rangefile_reads_and_rejects_missing_content_range(server):
    pytest.importorskip("requests")
    rf_mod = _script("rangefile")
    rf = rf_mod.RangeFile(f"{server}/f", block=64)
    assert rf.size == len(PAYLOAD)
    rf.seek(100)
    assert rf.read(300) == PAYLOAD[100:400]
    rf.close()
    with pytest.raises(IOError, match="Content-Range"):
        rf_mod.RangeFile(f"{server}/norange")


# ------------------------------------------------------------ end-to-end smoke

def test_readout_runs_on_a_synthetic_extract(tmp_path, monkeypatch):
    ad = pytest.importorskip("anndata")
    pytest.importorskip("particlegan")
    rng = np.random.default_rng(0)
    genes = sorted(hypoad.HOMEO | hypoad.DAM) + [f"G{i}" for i in range(40)]
    donors = ([f"CTL_{i}" for i in range(7)] + [f"AD_{i}" for i in range(40, 47)]
              + ["Y_1", "Y_2"])
    assert "AD_44" in donors
    cond = {d: ("Control" if d.startswith("CTL") else "Young" if d.startswith("Y") else "AD")
            for d in donors}
    cells = [d for d in donors for _ in range(50)]
    counts = rng.poisson(2.0, size=(len(cells), len(genes))).astype(np.float32)
    obs = pd.DataFrame({"donor": cells, "condition": [cond[d] for d in cells],
                        "cell_type": "Microglia"}, index=[f"c{i}" for i in range(len(cells))])
    path = tmp_path / "hypoad_microglia.h5ad"
    ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(path)
    monkeypatch.setenv("HYPOAD_H5AD", str(path))
    monkeypatch.syspath_prepend(str(SCRIPTS.parent))
    spec = importlib.util.spec_from_file_location(
        "_hypoad_readout", SCRIPTS.parent / "tests" / "test_hypoad_microglia_readout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    data = mod._prepare()
    _, group, kept_donors, *_ = data
    assert "AD_44" not in set(kept_donors) and not any(d.startswith("Y") for d in kept_donors)
    assert sorted(set(group)) == ["AD", "CTRL"]
    r = mod.run_once(seed=0, steps=3, data=data)
    assert "AD_44" not in r["scored_donors"]
    assert (r["n_fit_ctrl_donors"], r["n_ad"], r["n_ctrl_heldout"]) == (3, 6, 4)
    assert r["min_attainable_p"] == pytest.approx(1 / 210)
    assert r["model_p_exact"] and 0 < r["model_p"] <= 1
    lo, hi = r["model_ci95"]
    assert 0 <= lo <= hi <= 1
