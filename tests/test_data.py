import torch

from alz_perturb_gan.data import load_virtual_cell_h5ad, make_synthetic, perturbation_set_from_groups


def test_synthetic_holdout_keeps_combo_members_in_train():
    data = make_synthetic(seed=0, noise=0.0)
    lookup = data.by_name()
    assert data.named("train") and data.named("eval")
    for item in data.named("eval"):
        if item.combination:
            assert all(lookup[name].split == "train" for name in item.members)
    train_names = set(data.cell_perturbation)
    assert train_names.isdisjoint({item.name for item in data.named("eval")})


def test_second_dataset_is_the_same_interface():
    groups = perturbation_set_from_groups(
        name="hand",
        source="hand",
        gene_names=["a", "b"],
        control_mean=torch.zeros(2),
        profiles={"p": torch.tensor([1.0, 0.0]), "q": torch.tensor([0.0, 1.0])},
        train_names=["p"],
        eval_names=["q"],
    )
    synthetic = make_synthetic(seed=1)
    for data in (groups, synthetic):
        assert data.n_genes == data.control_mean.shape[0]
        assert data.cell_outcome.shape[0] == len(data.cell_perturbation)
        assert data.named("train")


def test_h5ad_loader_excludes_controls(tmp_path):
    anndata = __import__("anndata")
    import numpy as np
    import pandas as pd

    genes = 3
    labels = ["non-targeting"] * 4 + ["A"] * 5 + ["B"] * 5
    matrix = np.zeros((len(labels), genes), dtype=np.float32)
    for row, label in enumerate(labels):
        if label == "A":
            matrix[row] = (1.0, 0.0, 0.0)
        elif label == "B":
            matrix[row] = (0.0, 1.0, 0.0)
        else:
            matrix[row] = (0.2, 0.2, 0.2)
    obs = pd.DataFrame({"target_gene": labels})
    path = tmp_path / "tiny.h5ad"
    anndata.AnnData(matrix, obs=obs).write_h5ad(path)
    data = load_virtual_cell_h5ad(str(path), eval_fraction=0.5, max_cells_per_perturbation=2, seed=0)
    assert data.source == "virtual_cell_h5ad"
    assert {item.name for item in data.perturbations} == {"A", "B"}
    assert torch.allclose(data.control_mean, torch.tensor([0.2, 0.2, 0.2]))
    assert data.cell_control.shape[0] <= 2
    assert not data.paired_cells
