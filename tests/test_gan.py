import torch

from alz_perturb_gan.data import make_synthetic
from alz_perturb_gan.gan import train_gan_decision
from alz_perturb_gan.harness import compare_all
from alz_perturb_gan.baselines import fit_baselines
from alz_perturb_gan.judgments import MOVES_TOWARD


def test_gan_decision_smoke_step_is_finite():
    data = make_synthetic(seed=7, cells_per=2)
    model = train_gan_decision(data, steps=3, batch_size=8, n_particles=32, hidden=32, seed=7)
    for key, values in model.history.items():
        assert len(values) == 3
        assert all(value == value and abs(value) < 1e6 for value in values)
    names = [item.name for item in data.named("eval")]
    pred = model.predict_profiles(data, names, n_draws=2)
    assert pred.shape == (len(names), data.n_genes)
    assert torch.isfinite(pred).all()
    query = data.named("eval")[0]
    judgment = model.decision.judge(data.control_mean, query.embedding, query.profile)
    assert judgment.question == MOVES_TOWARD
    assert judgment.as_dict()["type"] == "noul"
    assert 0.0 < judgment.noul < 1.0
    scores = model.score_moves_toward(
        data.control_mean,
        torch.stack([item.embedding for item in data.named()]),
        query.profile,
    )
    assert scores.shape == (len(data.perturbations),)
    assert torch.isfinite(scores).all()


def test_harness_compares_baselines_and_decision_model():
    data = make_synthetic(seed=8, cells_per=2)
    gan = train_gan_decision(data, steps=2, batch_size=8, n_particles=32, hidden=32, seed=8)
    report = compare_all(data, fit_baselines(data), gan, n_draws=2)
    names = [row["model"] for row in report["models"]]
    assert names == ["no_change", "mean_effect", "additive", "linear", "gan_generator", "decision_noul"]
    assert report["models"][-1]["expression"] is None
    assert report["decision_train_check"]["n"] > 0
