import json

from alz_perturb_gan.smoke import main


def test_smoke_main_writes_a_comparison(tmp_path):
    output = tmp_path / "smoke.json"
    run_dir = tmp_path / "run"
    code = main([
        "--steps", "2",
        "--hypergan-steps", "2",
        "--output", str(output),
        "--run-dir", str(run_dir),
    ])
    assert code == 0
    report = json.loads(output.read_text())
    names = [row["model"] for row in report["models"]]
    assert names[:4] == ["no_change", "mean_effect", "additive", "linear"]
    assert "gan_generator" in names
    assert "decision_noul" in names
    assert report["cell_eval"]["available"] in (True, False)
    train = report["hypergan"].get("train") or {}
    if report["hypergan"]["probe"]["status"] == "develop":
        assert train.get("status") == "complete"
