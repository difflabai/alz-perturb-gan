"""Regression checks for the two-program readout and checkpoint sampling."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

from paper_watch.alztest.psychad_programs import (
    GROUPS, GBA1_ENSEMBL, LIPID, LYSOSOME, fit_checkpoints, marker_expression, program_scores,
)


def test_gba_marker_uses_the_ensembl_id():
    markers = [*LIPID, *LYSOSOME, "GBA1"]
    codes = [f"gene-{i}" for i in range(len(markers))]
    gba_index = markers.index("GBA1")
    codes[gba_index] = GBA1_ENSEMBL
    markers[gba_index] = "GBA"
    counts = np.ones((3, len(markers)))
    counts[:, gba_index] = 3
    data = SimpleNamespace(
        X=sparse.csr_matrix(counts),
        obs={"donor_id": ["c", "d", "l"], "diagnosis": GROUPS},
        var=pd.DataFrame({"feature_name": markers, "feature_id": codes}),
    )
    expression, groups = marker_expression(data)
    assert expression.shape == (3, 15)
    np.testing.assert_array_equal(groups, GROUPS)
    np.testing.assert_allclose(expression[:, gba_index], np.log1p(3 * 1e4 / counts.sum(1)))


def test_program_standardization_uses_training_donors():
    expression = np.random.default_rng(0).normal(size=(20, len(LIPID) + len(LYSOSOME)))
    train = np.arange(15)
    scores = program_scores(expression, train)
    expression[15:] += 100
    changed = program_scores(expression, train)

    np.testing.assert_array_equal(scores[train], changed[train])
    assert np.all(changed[15:] > scores[15:])


def test_checkpoint_sampling_preserves_fit_and_audits_regularizer():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(72, 2))
    labels = np.repeat(np.arange(len(GROUPS)), 24)
    observed = {group: rng.normal(size=(8, 2)) for group in GROUPS}

    earlier = fit_checkpoints(train, labels, observed, seed=3, rate=0.01, checkpoints=(2, 4))
    terminal = fit_checkpoints(train, labels, observed, seed=3, rate=0.01, checkpoints=(4,))

    assert earlier["peak_rates"] == {"generator": 0.01, "critic": 0.015, "prior": 0.025}
    assert [row["regularizer_calls"] for row in earlier["checkpoints"]] == [2, 4]
    assert earlier["checkpoints"][-1] == terminal["checkpoints"][-1]
