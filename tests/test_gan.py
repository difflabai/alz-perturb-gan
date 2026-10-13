import torch

from alz_perturb_gan.data import make_synthetic
from alz_perturb_gan.gan import CandidateSampler, train_gan_decision
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
    judgment = model.decision.judge(data.control_mean, pred[0], query.profile)
    assert judgment.question == MOVES_TOWARD
    assert judgment.as_dict()["type"] == "noul"
    assert 0.0 < judgment.noul < 1.0
    scores = model.score_candidates(
        data.control_mean,
        model.predict_profiles(data, [item.name for item in data.named()], n_draws=2),
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
    assert report["decision_train_check"]["n_targets"] > 0


def test_decision_labels_have_both_classes_with_nontrivial_positive_rate():
    data = make_synthetic(seed=9, cells_per=3)
    sampler = CandidateSampler(data, positive_rate=0.5)
    rng = torch.Generator().manual_seed(0)
    target, candidate, label = sampler.sample(4096, rng)
    assert set(label.unique().tolist()) == {0.0, 1.0}
    assert 0.4 < float(label.mean()) < 0.6
    assert torch.equal(label > 0, candidate == target)
    cells = sampler.cells_for(candidate, 5, rng)
    names = [data.cell_perturbation[i] for i in cells.reshape(-1).tolist()]
    expected = [sampler.names[c] for c in candidate.tolist() for _ in range(5)]
    assert names == expected

    model = train_gan_decision(data, steps=20, batch_size=16, n_particles=32, hidden=32, seed=9)
    rates = model.history["real_positive_rate"]
    assert 0.25 < sum(rates) / len(rates) < 0.75
    assert min(rates) < 1.0 and max(rates) > 0.0

    report = compare_all(data, fit_baselines(data), model, n_draws=2)["decision_train_check"]
    assert report["balanced_positive_rate"] == 0.5
    assert 0.0 < report["full_positive_rate"] < 1.0
    assert report["full_auroc"] == report["full_auroc"]


def test_decision_learns_identity_on_easy_synthetic():
    data = make_synthetic(seed=3, cells_per=6, noise=0.01)
    model = train_gan_decision(data, steps=300, batch_size=32, n_particles=32, hidden=64, seed=3, lr=1e-3)
    report = compare_all(data, fit_baselines(data), model, n_draws=4)["decision_train_check"]
    assert report["full_auroc"] > 0.7
