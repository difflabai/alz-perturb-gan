"""CPU regression checks for the paper-watch harness."""
import numpy as np
from particlegan import get_recipe
import pytest

from paper_watch.alztest.harness import BATCH, ConditionalPGAN, TrainingSettings
from paper_watch.alztest.neuroko_duration import optimizer_rates


@pytest.mark.parametrize("use_recipe_defaults", (False, True))
def test_fit_keeps_condition_ids_and_uses_recipe_batch_size(monkeypatch, use_recipe_defaults):
    model = ConditionalPGAN(data_dim=2, n_cond=3, seed=7, use_recipe_defaults=use_recipe_defaults)
    expected_batch_size = BATCH
    if use_recipe_defaults:
        published = get_recipe("empf")
        assert model.trainer.recipe == published
        assert model.prior.num_particles == published.num_particles
        assert model.prior.z_dim == published.z_dim
        expected_batch_size = published.batch_size
    labels = np.repeat([0, 1, 2], [8, 33, BATCH])
    values = np.broadcast_to(labels[:, None], (len(labels), 2))
    seen = []

    def record(batch):
        expected = model.G.cond.argmax().item()
        assert model.D.cond.argmax().item() == expected
        assert batch.shape == (expected_batch_size, 2)
        assert (batch == expected).all()
        seen.append(expected)

    monkeypatch.setattr(model.trainer, "step", record)
    model.fit(values, labels, steps=4)

    assert sorted(seen) == [1, 1, 2, 2]


def test_fit_rejects_an_empty_set_of_supported_conditions():
    model = ConditionalPGAN(data_dim=2, n_cond=2)
    with pytest.raises(ValueError, match="at least 32 rows"):
        model.fit(np.zeros((5, 2)), np.zeros(5, dtype=int), steps=1)


def test_released_empf_trains_and_draws_on_cpu():
    model = ConditionalPGAN(data_dim=2, n_cond=2)
    labels = np.repeat([0, 1], BATCH)
    values = np.random.default_rng(0).normal(size=(len(labels), 2))
    model.fit(values, labels, steps=2)
    draws = model.sample(0, 8)
    assert draws.shape == (8, 2)
    assert np.isfinite(draws).all()


def test_restored_fit_replays_condition_order_and_row_draws():
    labels = np.repeat([0, 1, 2], [36, 71, 53])
    values = np.random.default_rng(42).normal(size=(len(labels), 3))
    original = ConditionalPGAN(data_dim=3, n_cond=3, seed=7)
    saved = {}

    def save(step):
        if step == 5:
            saved.update(original.trainer.state_dict())

    original.fit(values, labels, steps=12, after_step=save)
    restored = ConditionalPGAN(data_dim=3, n_cond=3, seed=7)
    restored.trainer.load_state_dict(saved)
    restored.fit(values, labels, steps=7, replay_batches=restored.trainer.completed_steps)
    assert restored.trainer.completed_steps == 12
    for label in range(3):
        np.testing.assert_array_equal(restored.sample(label, 32), original.sample(label, 32))


def test_independent_rates_and_output_initialization():
    settings = TrainingSettings(d_lr_mult=0.3, prior_lr_mult=4.0, generator_output_init_gain=2)
    tuned = ConditionalPGAN(3, 2, seed=5, learning_rate=0.01, settings=settings)
    reference = ConditionalPGAN(3, 2, seed=5, learning_rate=0.01)
    assert optimizer_rates(tuned) == pytest.approx({"generator": 0.01, "critic": 0.003, "particles": 0.04})
    for (name, param), reference_param in zip(tuned.G.named_parameters(), reference.G.parameters()):
        factor = 2 if name == "net.4.weight" else 1
        np.testing.assert_array_equal(param.detach(), reference_param.detach() * factor)
    assert tuned.trainer.recipe.optimizer_family == reference.trainer.recipe.optimizer_family
    assert tuned.trainer.recipe.prior_regularizer == reference.trainer.recipe.prior_regularizer


def test_stratified_prior_uses_every_particle_during_training_and_preserves_evaluation(monkeypatch):
    import torch

    priors = pytest.importorskip("particlegan.priors")
    if not hasattr(priors, "StratifiedSamplingManipulator"):
        pytest.skip("the installed ParticleGAN release predates stratified prior sampling")

    def make(stratified):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            return ConditionalPGAN(3, 2, seed=7, recipe_name="empf",
                                   settings=TrainingSettings(stratified_prior=stratified))

    plain, stratified = make(False), make(True)
    torch.testing.assert_close(stratified.prior.z, plain.prior.z, rtol=0, atol=0)
    np.testing.assert_array_equal(stratified.sample(1, 256, seed=12), plain.sample(1, 256, seed=12))
    sampler = next(m for m in stratified.prior.manipulators if isinstance(m, priors.StratifiedSamplingManipulator))
    draw = sampler.sample_indices
    observed = []

    def record(base, batch_size, **kwargs):
        ids = draw(base, batch_size, **kwargs)
        observed.append((sampler.training, len(ids), ids.unique().numel()))
        return ids

    monkeypatch.setattr(sampler, "sample_indices", record)
    labels = np.repeat([0, 1], BATCH)
    stratified.fit(np.random.default_rng(4).normal(size=(len(labels), 3)), labels, steps=2)
    assert observed and all(row == (True, BATCH, BATCH) for row in observed)
    stratified.sample(0, 256, seed=12)
    assert observed[-1][0:2] == (False, 256)


@pytest.mark.parametrize("sigma", [0., .15])
def test_particle_panel_uses_the_same_rows_with_independent_native_jitter(sigma):
    import torch
    from paper_watch.alztest.neuroko_tune import particle_panel

    model = ConditionalPGAN(3, 3, seed=7, recipe_name="empf", settings=TrainingSettings(prior_sigma=sigma))
    observed = [np.arange(36).reshape(12, 3) / 11.] * 3
    original = model.sample(0, 16, seed=27)
    panel = particle_panel(model, ["ctrl", "a", "b"], observed, repeats=4, seed=73219)
    assert model.G.cond.argmax().item() == model.D.cond.argmax().item() == 0
    np.testing.assert_array_equal(model.sample(0, 16, seed=27), original)
    generator = torch.Generator().manual_seed(73220)
    model._set_cond(1)
    outputs = []
    with torch.no_grad():
        for _ in range(4):
            latent, indices = model.prior.sample(panel["particles"], fixed_first_n=True, generator=generator)
            assert torch.equal(indices, torch.arange(panel["particles"]))
            outputs.append(model.G(latent).double().numpy())
    values = np.stack(outputs)
    result = panel["targets"]["a"]
    assert result["total"] == pytest.approx(values.reshape(-1, 3).var(0).sum())
    assert result["between"] == pytest.approx(values.mean(0).var(0).sum())
    assert result["within"] == pytest.approx(values.var(0).sum(-1).mean())
    assert (result["within"] > 0) == (sigma > 0)


@pytest.mark.parametrize("field", ["lr_floor", "network_lr_floor"])
@pytest.mark.parametrize("horizon", [1200, 7000])
def test_fixed_rate_benchmark_rejects_decay_in_either_role(field, horizon):
    from paper_watch.alztest.neuroko_tune import verify_fixed_schedule

    fixed = {"lr_floor": 1., "network_lr_floor": 1., "total_steps": horizon}
    verify_fixed_schedule(get_recipe("empf", **fixed), 1200)
    with pytest.raises(ValueError, match="changes network or particle LR"):
        verify_fixed_schedule(get_recipe("empf", **(fixed | {field: .1})), 1200)


def test_fixed_rate_benchmark_rejects_delayed_staircase_decay():
    from paper_watch.alztest.neuroko_tune import verify_fixed_schedule

    recipe = get_recipe("empf", lr_schedule="exponential", lr_decay_staircase=True,
                        lr_decay_rate=0.96, lr_decay_steps=50000)
    with pytest.raises(ValueError, match="changes network or particle LR at update 50000"):
        verify_fixed_schedule(recipe, 1200)


def test_v12_gain_diagnostic_reads_the_released_optimizer_state():
    from paper_watch.alztest.neuroko_duration import function_space_gains

    pytest.importorskip("particlegan._unified_v12")
    model = ConditionalPGAN(3, 2, seed=4, recipe_name="unified_v12")
    labels = np.repeat([0, 1], BATCH)
    model.fit(np.random.default_rng(9).normal(size=(len(labels), 3)), labels, steps=2)
    gains = function_space_gains(model.trainer)
    for role, opt in (("generator", model.trainer.opt_g), ("critic", model.trainer.opt_d)):
        expected = opt.state_dict()["unified_v12_equil"]
        assert gains[role] == {"ema_func_gain": expected["ema_func_gain"],
                               "last_step_gain": expected["last_step_gain"],
                               "probe_consumptions": opt.probe_consumptions}


def test_sampling_noise_diagnostic_restores_native_draws(monkeypatch):
    from paper_watch.alztest import neuroko_tune

    pytest.importorskip("particlegan._unified_v12")
    model = ConditionalPGAN(3, 2, seed=4, recipe_name="unified_v12", settings=TrainingSettings(prior_sigma=.025))
    observed = np.random.default_rng(22).normal(size=(2, 32, 3))
    conditions = ["control", "target"]
    original = model.sample(1, 64, seed=25)
    native = neuroko_tune.score_draw(model, 24, observed, conditions)
    alternate = neuroko_tune.score_alternate_noise(model, .025, .15, [24], observed, conditions)
    assert alternate["24"] != native
    np.testing.assert_array_equal(model.sample(1, 64, seed=25), original)

    def interrupt(*args):
        raise RuntimeError("sampling interrupted")

    monkeypatch.setattr(neuroko_tune, "score_draw", interrupt)
    with pytest.raises(RuntimeError, match="sampling interrupted"):
        neuroko_tune.score_alternate_noise(model, .025, .15, [24], observed, conditions)
    np.testing.assert_array_equal(model.sample(1, 64, seed=25), original)
