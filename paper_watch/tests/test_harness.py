"""CPU regression checks for the paper-watch harness."""
from types import SimpleNamespace

import numpy as np
import pytest

from paper_watch.alztest.harness import BATCH, ConditionalPGAN


def test_fit_keeps_condition_ids_and_fixed_batch_size():
    model = ConditionalPGAN(data_dim=2, n_cond=3, seed=7)
    labels = np.repeat([0, 1, 2], [8, 33, BATCH])
    values = np.broadcast_to(labels[:, None], (len(labels), 2))
    seen = []

    def record(batch):
        expected = model.G.cond.argmax().item()
        assert model.D.cond.argmax().item() == expected
        assert batch.shape == (BATCH, 2)
        assert (batch == expected).all()
        seen.append(expected)

    model.trainer = SimpleNamespace(step=record)
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
