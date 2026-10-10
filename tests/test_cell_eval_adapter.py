from alz_perturb_gan.cell_eval_adapter import cell_eval_status, score_anndata


def test_cell_eval_is_optional():
    status = cell_eval_status()
    if status["available"]:
        assert "origin" in status
        return
    scored = score_anndata(None, None)
    assert scored["scored"] is False
    assert "cell_eval" in scored["reason"]
    assert "cell-eval run" in scored["cli"]
