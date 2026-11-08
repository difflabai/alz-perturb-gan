"""Compare every model on the same held-out profiles and phenocopy queries."""

from __future__ import annotations

import math

import torch

from .baselines import Baseline
from .data import PerturbationSet
from .gan import GANDecision
from .metrics import expression_report, mean_phenocopy, phenocopy_rank


def _eval_bundle(data: PerturbationSet) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    rows = data.named("eval")
    if not rows:
        raise ValueError("dataset has no eval perturbations")
    names = [item.name for item in rows]
    true = torch.stack([item.profile for item in rows])
    return names, true, data.control_mean


def evaluate_expression(data: PerturbationSet, pred: torch.Tensor, *, k: int | None = None) -> dict:
    names, true, control = _eval_bundle(data)
    if pred.shape != true.shape:
        raise ValueError(f"prediction shape {tuple(pred.shape)} != eval profiles {tuple(true.shape)}")
    width = max(1, min(10, data.n_genes // 2))
    return expression_report(pred, true, control, k=width if k is None else k)


def phenocopy_from_profiles(
    data: PerturbationSet,
    predicted_by_name: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Rank candidates by closeness of their predicted profile to the target.

    The target is the observed eval profile. Candidates are the whole library.
    The correct phenocopy is the perturbation that produced the target.
    """
    library = data.named()
    rows = []
    for query in data.named("eval"):
        target = query.profile
        scores = []
        true_index = None
        for index, candidate in enumerate(library):
            if candidate.name == query.name:
                true_index = index
            scores.append(-torch.linalg.vector_norm(predicted_by_name[candidate.name] - target))
        rows.append(phenocopy_rank(torch.stack(scores), true_index))
    return mean_phenocopy(rows)


def phenocopy_from_scores(data: PerturbationSet, score_fn) -> dict[str, float]:
    """Rank candidates with a caller-supplied score, higher is better."""
    library = data.named()
    rows = []
    for query in data.named("eval"):
        scores = score_fn(data.control_mean, query.profile, library)
        true_index = next(index for index, item in enumerate(library) if item.name == query.name)
        rows.append(phenocopy_rank(scores, true_index))
    return mean_phenocopy(rows)


def evaluate_baseline(data: PerturbationSet, model: Baseline) -> dict:
    library_names = [item.name for item in data.named()]
    predicted = model.predict(data, library_names)
    by_name = {name: predicted[index] for index, name in enumerate(library_names)}
    eval_names = [item.name for item in data.named("eval")]
    eval_pred = torch.stack([by_name[name] for name in eval_names])
    return {
        "model": model.name,
        "expression": evaluate_expression(data, eval_pred),
        "phenocopy": phenocopy_from_profiles(data, by_name),
    }


def evaluate_gan(data: PerturbationSet, model: GANDecision, *, n_draws: int = 8) -> dict:
    library_names = [item.name for item in data.named()]
    predicted = model.predict_profiles(data, library_names, n_draws=n_draws)
    by_name = {name: predicted[index] for index, name in enumerate(library_names)}
    eval_names = [item.name for item in data.named("eval")]
    eval_pred = torch.stack([by_name[name] for name in eval_names])

    def decision_scores(control, target, library):
        embedding = torch.stack([item.embedding for item in library])
        return model.score_moves_toward(control, embedding, target)

    return {
        "gan_generator": {
            "model": "gan_generator",
            "expression": evaluate_expression(data, eval_pred),
            "phenocopy": phenocopy_from_profiles(data, by_name),
        },
        "decision_noul": {
            "model": "decision_noul",
            "expression": None,
            "phenocopy": phenocopy_from_scores(data, decision_scores),
            "question": model.decision.question,
        },
    }


def decision_label_report(data: PerturbationSet, model: GANDecision, *, seed: int = 0) -> dict:
    """Accuracy of the noul on held-out cells is not available.

    Eval cells are not stored. This report scores training cells against a
    fresh draw of training targets, which is an in-sample calibration check,
    not a test-set claim.
    """
    if data.cell_control.shape[0] == 0:
        return {"n": 0}
    generator = torch.Generator().manual_seed(seed)
    names, profiles = data.train_profiles()
    index = torch.randint(data.cell_control.shape[0], (data.cell_control.shape[0],), generator=generator)
    target_index = torch.randint(len(names), (index.shape[0],), generator=generator)
    control = data.cell_control[index]
    outcome = data.cell_outcome[index]
    embedding = data.cell_embedding[index]
    target = profiles[target_index]
    label = ((outcome - target).norm(dim=-1) < (control - target).norm(dim=-1)).float()
    with torch.no_grad():
        noul = model.decision(control, embedding, target)
    accuracy = float(((noul >= 0.5) == (label >= 0.5)).float().mean())
    brier = float(((noul - label) ** 2).mean())
    return {
        "split": "train_cells_reshuffled_targets",
        "n": int(label.shape[0]),
        "accuracy": accuracy,
        "brier": brier,
        "mean_noul": float(noul.mean()),
        "positive_rate": float(label.mean()),
        "note": "In-sample check only. It is not evidence the probability is calibrated on held-out biology.",
    }


def compare_all(data: PerturbationSet, baselines: dict, gan: GANDecision | None = None, *, n_draws: int = 8) -> dict:
    rows = [evaluate_baseline(data, model) for model in baselines.values()]
    extra = {}
    if gan is not None:
        judged = evaluate_gan(data, gan, n_draws=n_draws)
        rows.append(judged["gan_generator"])
        rows.append(judged["decision_noul"])
        extra = {"decision_train_check": decision_label_report(data, gan)}
    return {"dataset": data.name, "source": data.source, "models": rows, **extra}


def finite_metric(value) -> bool:
    return isinstance(value, (int, float)) and value == value and not math.isinf(value)
