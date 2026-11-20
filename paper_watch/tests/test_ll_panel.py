"""Data-free regression for the lipid–lysosome gene panel diagnostic."""
import numpy as np

from paper_watch.alztest.ll_panel import LIPID, LYSO, panel_coverage


def test_panel_coverage_reports_partial_and_complete_modules(tmp_path):
    path = tmp_path / "space.npz"
    np.savez(path, genes=np.array(["OTHER", "PLIN2"]))
    result = panel_coverage(path)
    assert result["panel_size"] == 2
    assert not result["full_module_coverage"]
    assert result["lipid"]["present"] == ["PLIN2"]
    assert result["lysosome"]["missing"] == list(LYSO)

    np.savez(path, genes=np.array(LIPID + LYSO))
    result = panel_coverage(path)
    assert result["full_module_coverage"]
    assert result["lipid"]["missing"] == result["lysosome"]["missing"] == []
