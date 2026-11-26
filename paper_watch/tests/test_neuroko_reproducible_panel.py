import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "neuroko_reproducible_panel.py"
spec = importlib.util.spec_from_file_location("neuroko_reproducible_panel", SCRIPT)
stage_a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage_a)


def _write_screen(path, real=("G01", "G02", "G03", "G04", "G05"), null=12, cells=140, genes=600, seed=0):
    rng = np.random.default_rng(seed)
    gene_names = [f"G{i:02d}" for i in range(1, 30)] + [f"X{i}" for i in range(genes - 29)]
    targets = ["INT", *real, *[f"G{i:02d}" for i in range(6, 6 + null)]]
    base = rng.gamma(2.0, 1.0, genes)
    effects = {}
    for t in real:
        eff = np.ones(genes)
        idx = rng.choice(np.arange(40, genes), 60, replace=False)
        eff[idx] = rng.choice([0.25, 4.0], 60)
        eff[gene_names.index(t)] = 0.1
        effects[t] = eff
    rows, obs = [], []
    for group in ("Screen1_1", "Screen1_2", "Screen1_3", "Screen1_4"):
        for t in targets:
            rate = base * effects.get(t, np.ones(genes))
            counts = rng.poisson(rate, size=(cells, genes)).astype(np.float32)
            if group == "Screen1_2":
                counts[:] = -1  # reading a reserved row must fail loudly
            rows.append(counts)
            obs += [{"perturbation": t, "target_vector": f"{t}_v1", "Replicate": group}] * cells
    x = sparse.csr_matrix(np.vstack(rows))
    adata = anndata.AnnData(X=x, obs=pd.DataFrame(obs), var=pd.DataFrame(index=gene_names))
    adata.obs.index = adata.obs.index.astype(str)
    adata.write_h5ad(path / "iNeurons.h5ad")
    return set(real)


def test_stage_a_selects_reproducible_knockouts_without_reading_reserved_rows(tmp_path):
    real = _write_screen(tmp_path)
    out = tmp_path / "out"
    stage_a.main(["--data", str(tmp_path), "--out", str(out), "--window-cells", "100000"])
    result = json.loads((out / "stageA_panels.json").read_text())
    assert result["reserved_groups_dropped"] == ["Screen1_2"]
    assert result["stop"] is False
    for fold in result["folds"]:
        assert set(fold["panel"]) == real
        assert fold["heldout"] not in fold["training"]
        for t in fold["panel"]:
            assert fold["after_selection_heldout_check"][t]["heldout_cosine"] > 0.5
        base = fold["baselines"]
        assert base["empirical"]["mean_energy_u"] < base["int_only"]["mean_energy_u"]
        assert base["mean_shift"]["ko_gain_u"] > 0
        assert (out / fold["arrays"]["path"]).is_file()


def test_stage_a_stops_when_no_knockout_repeats(tmp_path):
    _write_screen(tmp_path, real=(), null=14)
    out = tmp_path / "out"
    stage_a.main(["--data", str(tmp_path), "--out", str(out), "--window-cells", "100000"])
    result = json.loads((out / "stageA_panels.json").read_text())
    assert result["stop"] is True
    assert all(len(f["panel"]) < 4 for f in result["folds"])
    assert all("baselines" not in f for f in result["folds"])


def test_energy_estimators_differ_by_within_sample_terms():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(50, 3)), rng.normal(size=(40, 3))
    v, u = stage_a.energy_pair(a, b)
    assert v > u
    # The U-statistic of two samples from one distribution is centered near zero.
    assert abs(u) < 0.2
