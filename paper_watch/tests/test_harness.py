"""CPU regression checks for the paper-watch harness."""
import numpy as np
from particlegan import get_recipe
import pytest

from paper_watch.alztest.harness import BATCH, ConditionalPGAN


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
