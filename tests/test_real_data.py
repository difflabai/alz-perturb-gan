import numpy as np
import pandas as pd
import pytest
import torch
import anndata as ad
from scipy import sparse

from alz_perturb_gan.real_data import load_real_data, transform_expression


def test_streamed_means_and_seeded_identity_holdout(tmp_path):
    labels = ["non-targeting"] * 5 + [name for name in ["A", "B", "C", "D"] for _ in range(7)]
    values = np.random.default_rng(7).poisson(2, (len(labels), 5)).astype(np.float32)
    path = tmp_path / "counts.h5ad"
    ad.AnnData(sparse.csr_matrix(values), obs=pd.DataFrame(
        {"target_gene": labels}, index=[str(i) for i in range(len(labels))])).write_h5ad(path)
    data, real = load_real_data(path, eval_fraction=.5, split_seed=42, cell_seed=43,
                               train_cells=3, eval_cells=4, chunk_size=6)
    again, _ = load_real_data(path, eval_fraction=.5, split_seed=42, cell_seed=43,
                             train_cells=3, eval_cells=4, chunk_size=11)
    assert data.metadata["eval_names"] == again.metadata["eval_names"]
    assert set(data.cell_perturbation).isdisjoint(data.metadata["eval_names"])
    assert set(real.obs.target_gene) == set(data.metadata["eval_names"]) | {"non-targeting"}
    assert data.cell_outcome.shape[0] == 6
    expected = transform_expression(values, "counts")
    for item in data.named():
        mask = np.asarray(labels) == item.name
        assert np.allclose(item.profile.numpy(), expected[mask].mean(0), atol=1e-6)
        assert torch.allclose(item.profile, again.by_name()[item.name].profile)
    assert torch.allclose(data.cell_outcome, again.cell_outcome)
    assert np.allclose(data.control_mean, expected[:5].mean(0))


def test_expression_scale_is_explicit():
    with pytest.raises(ValueError, match="fractional"):
        transform_expression(np.array([[.3, 2.]], dtype=np.float32), "counts")
    log = np.array([[.3, 2.]], dtype=np.float32)
    assert np.array_equal(transform_expression(log, "log1p"), log)
    zero = np.zeros((1, 2), dtype=np.float32)
    assert np.array_equal(transform_expression(zero, "counts"), zero)
