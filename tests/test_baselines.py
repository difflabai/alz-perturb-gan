import torch

from alz_perturb_gan.baselines import Additive, LinearModel, MeanEffect, NoChange, fit_baselines
from alz_perturb_gan.data import make_synthetic


def test_no_change_is_the_control():
    data = make_synthetic(noise=0.0, seed=1)
    model = NoChange().fit(data)
    names = [item.name for item in data.named("eval")]
    pred = model.predict(data, names)
    assert torch.allclose(pred, data.control_mean.expand_as(pred))


def test_mean_effect_matches_mean_training_profile():
    data = make_synthetic(noise=0.0, seed=2)
    model = MeanEffect().fit(data)
    _names, profiles = data.train_profiles()
    pred = model.predict(data, [data.named("eval")[0].name])
    assert torch.allclose(pred[0], profiles.mean(dim=0))


def test_additive_combinations_match_the_paper_formula():
    data = make_synthetic(noise=0.0, seed=3)
    model = Additive().fit(data)
    lookup = data.by_name()
    combos = [item for item in data.named("eval") if item.combination]
    assert combos
    pred = model.predict(data, [item.name for item in combos])
    for index, item in enumerate(combos):
        members = [lookup[name] for name in item.members]
        assert all(member.split == "train" for member in members)
        expected = members[0].profile + members[1].profile - data.control_mean
        assert torch.allclose(pred[index], expected, atol=1e-5)
        assert torch.allclose(pred[index], item.profile, atol=1e-5)


def test_linear_beats_no_change_on_held_out_singles():
    data = make_synthetic(noise=0.0, seed=4)
    models = fit_baselines(data, ridge=0.1)
    singles = [item.name for item in data.named("eval") if not item.combination]
    assert singles
    true = torch.stack([data.by_name()[name].profile for name in singles])
    linear = models["linear"].predict(data, singles)
    unchanged = models["no_change"].predict(data, singles)
    linear_mae = (linear - true).abs().mean()
    base_mae = (unchanged - true).abs().mean()
    assert linear_mae < 0.5 * base_mae


def test_fit_does_not_require_eval_profiles():
    data = make_synthetic(noise=0.0, seed=5)
    for item in data.named("eval"):
        item.profile = torch.full_like(item.profile, 99.0)
    LinearModel().fit(data)
    Additive().fit(data)
