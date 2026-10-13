"""Simple perturbation baselines.

These follow Ahlmann-Eltze, Huber, and Anders, Nature Methods 2025
(https://doi.org/10.1038/s41592-025-02772-6):

* no change predicts the control profile
* the mean baseline predicts the average training expression, which is the
  control plus the mean perturbation effect when every profile shares that control
* additive predicts a combination as y_A + y_B - y_control
* the linear model is Y ≈ G W P^T + b, with G the leading gene singular vectors
  of the training matrix and P the perturbation embeddings

None of them see eval profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import PerturbationSet


@dataclass
class Prediction:
    name: str
    expression: torch.Tensor  # [N, G] aligned with the requested perturbation names


class Baseline:
    name = "baseline"

    def fit(self, data: PerturbationSet) -> "Baseline":
        return self

    def predict(self, data: PerturbationSet, names: list[str]) -> torch.Tensor:
        raise NotImplementedError


class NoChange(Baseline):
    name = "no_change"

    def predict(self, data: PerturbationSet, names: list[str]) -> torch.Tensor:
        return data.control_mean.unsqueeze(0).expand(len(names), -1).clone()


class MeanEffect(Baseline):
    """Control plus the mean training delta.

    With a shared control this is the mean of the training expression profiles,
    the 'mean' baseline in the Nature Methods comparison.
    """

    name = "mean_effect"

    def fit(self, data: PerturbationSet) -> "MeanEffect":
        _names, profiles = data.train_profiles()
        self.prediction = profiles.mean(dim=0)
        return self

    def predict(self, data: PerturbationSet, names: list[str]) -> torch.Tensor:
        return self.prediction.unsqueeze(0).expand(len(names), -1).clone()


class Additive(Baseline):
    """y_A + y_B - y_control for combinations of training singles.

    A single training perturbation predicts its own training profile. An unseen
    single, or a combination with an unseen member, uses the mean training
    delta for the missing piece. Eval combination profiles are never read.
    """

    name = "additive"

    def fit(self, data: PerturbationSet) -> "Additive":
        names, profiles = data.train_profiles()
        self.control = data.control_mean
        self.mean_delta = profiles.mean(dim=0) - self.control
        self.delta = {name: profile - self.control for name, profile in zip(names, profiles)}
        self.lookup = data.by_name()
        return self

    def predict(self, data: PerturbationSet, names: list[str]) -> torch.Tensor:
        rows = []
        for name in names:
            item = self.lookup[name]
            if not item.combination:
                rows.append(self.control + self.delta.get(name, self.mean_delta))
                continue
            pieces = [self.delta.get(member, self.mean_delta) for member in item.members]
            rows.append(self.control + torch.stack(pieces).sum(dim=0))
        return torch.stack(rows)


class LinearModel(Baseline):
    """Ridge linear map from perturbation embeddings through gene singular vectors.

    G is the top-K left singular vectors of the centered training matrix
    (one row per gene). P is the training embedding matrix. W solves the
    normal equations in Ahlmann-Eltze et al. equation (3), with ridge λ.
    The paper uses λ = 0.1 on transcriptome-scale matrices. The same default
    is used here; pass `ridge` to change it.
    """

    name = "linear"

    def __init__(self, n_components: int = 4, ridge: float = 0.1, device: str = "cpu"):
        if n_components < 1:
            raise ValueError("n_components must be positive")
        if ridge < 0:
            raise ValueError("ridge must be nonnegative")
        self.n_components = n_components
        self.ridge = ridge
        self.device = device

    def fit(self, data: PerturbationSet) -> "LinearModel":
        names, profiles = data.train_profiles()
        y = profiles.T.to(self.device).double()  # [genes, perts]
        b = y.mean(dim=1)
        centered = y - b.unsqueeze(1)
        _u, _s, _vh = torch.linalg.svd(centered, full_matrices=False)
        k = min(self.n_components, _u.shape[1], centered.shape[1])
        g = _u[:, :k]
        p = torch.stack([data.by_name()[name].embedding for name in names]).to(self.device).double()
        gram_g = g.T @ g
        gram_p = p.T @ p
        eye_g = torch.eye(gram_g.shape[0], dtype=torch.float64, device=self.device)
        eye_p = torch.eye(gram_p.shape[0], dtype=torch.float64, device=self.device)
        middle = g.T @ centered @ p
        left = torch.linalg.solve(gram_g + self.ridge * eye_g, middle)
        weight = torch.linalg.solve(gram_p + self.ridge * eye_p, left.T).T
        self.gene_basis = g
        self.weight = weight
        self.bias = b
        return self

    def predict(self, data: PerturbationSet, names: list[str]) -> torch.Tensor:
        lookup = data.by_name()
        p = torch.stack([lookup[name].embedding for name in names]).to(self.device).double()
        y = self.gene_basis @ self.weight @ p.T + self.bias.unsqueeze(1)
        return y.T.float().cpu()


def fit_baselines(data: PerturbationSet, *, n_components: int = 4, ridge: float = 0.1) -> dict[str, Baseline]:
    models: list[Baseline] = [
        NoChange(),
        MeanEffect(),
        Additive(),
        LinearModel(n_components=n_components, ridge=ridge),
    ]
    return {model.name: model.fit(data) for model in models}
