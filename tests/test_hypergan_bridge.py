from alz_perturb_gan.data import make_synthetic
from alz_perturb_gan.hypergan_bridge import (
    DEVELOP_COMMIT,
    RC2_FINDINGS,
    describe_hypergan,
    train_condition_gan,
)


def test_rc2_findings_are_explicit():
    assert RC2_FINDINGS["git_tag"] == "1.0.0-rc2"
    assert RC2_FINDINGS["pypi_version"] == "1.0.6"
    assert RC2_FINDINGS["importable_here"] is False
    assert any("hyperchamber" in item for item in RC2_FINDINGS["errors"])
    assert any("torchvision" in item for item in RC2_FINDINGS["errors"])
    assert RC2_FINDINGS["develop_commit"] == DEVELOP_COMMIT


def test_develop_branch_trains_or_failure_is_structured(tmp_path):
    info = describe_hypergan()
    data = make_synthetic(seed=9, cells_per=2, n_genes=8, n_single=4, n_combo=2, holdout_singles=1, holdout_combos=1)
    if info["status"] != "develop":
        assert info["status"] in {"unavailable", "unexpected"}
        assert "rc2" in info
        return
    result = train_condition_gan(data, tmp_path / "run", steps=2, batch_size=4)
    assert result["status"] == "complete"
    assert result["steps"] == 2
    assert result["hypergan"].startswith("2.")
