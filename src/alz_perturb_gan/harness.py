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
        profiles = torch.stack([by_name[item.name] for item in library])
        return model.score_candidates(control, profiles, target)

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


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    pos, neg = scores[labels > 0.5], scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    greater = (pos[:, None] > neg[None, :]).float().mean()
    ties = (pos[:, None] == neg[None, :]).float().mean()
    return float(greater + 0.5 * ties)


def decision_pair_report(
    data: PerturbationSet,
    model: GANDecision,
    targets: list,
    predicted_by_name: dict,
    *,
    seed: int = 0,
    split: str = "",
) -> dict:
    """Score the well-posed label for each target T against every library candidate P.

    label(P, T) = 1 iff P is T's perturbation. Candidate profiles are the
    generator's averaged predictions. Reports both the full-library pairs
    (positive rate 1/L, AUROC) and a balanced set of one positive plus one
    seeded random negative per target (positive rate 0.5, accuracy, Brier).
    """
    library = [item.name for item in data.named()]
    profiles = torch.stack([predicted_by_name[name] for name in library])
    rng = torch.Generator().manual_seed(seed)
    all_scores, all_labels, bal_scores, bal_labels = [], [], [], []
    for target in targets:
        scores = model.score_candidates(data.control_mean, profiles, target.profile)
        labels = torch.tensor([1.0 if name == target.name else 0.0 for name in library])
        all_scores.append(scores)
        all_labels.append(labels)
        true_index = library.index(target.name)
        offset = int(torch.randint(1, len(library), (1,), generator=rng))
        neg_index = (true_index + offset) % len(library)
        bal_scores.extend([scores[true_index], scores[neg_index]])
        bal_labels.extend([1.0, 0.0])
    if not targets:
        return {"split": split, "n_targets": 0}
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    bal_s = torch.stack(bal_scores)
    bal_l = torch.tensor(bal_labels)
    return {
        "split": split,
        "label": "candidate perturbation == target perturbation",
        "n_targets": len(targets),
        "n_candidates": len(library),
        "full_positive_rate": float(labels.mean()),
        "full_auroc": _auroc(scores, labels),
        "full_mean_noul_pos": float(scores[labels > 0.5].mean()),
        "full_mean_noul_neg": float(scores[labels <= 0.5].mean()),
        "balanced_positive_rate": float(bal_l.mean()),
        "balanced_accuracy": float(((bal_s >= 0.5) == (bal_l >= 0.5)).float().mean()),
        "balanced_brier": float(((bal_s - bal_l) ** 2).mean()),
        "balanced_auroc": _auroc(bal_s, bal_l),
    }


def decision_label_report(data: PerturbationSet, model: GANDecision, *, seed: int = 0, n_draws: int = 8) -> dict:
    """In-sample check on training targets; not a held-out claim."""
    library = [item.name for item in data.named()]
    predicted = model.predict_profiles(data, library, n_draws=n_draws)
    by_name = dict(zip(library, predicted))
    report = decision_pair_report(data, model, data.named("train"), by_name, seed=seed, split="train_targets")
    report["note"] = "In-sample check only. It is not evidence the probability is calibrated on held-out biology."
    return report


def compare_all(data: PerturbationSet, baselines: dict, gan: GANDecision | None = None, *, n_draws: int = 8) -> dict:
    rows = [evaluate_baseline(data, model) for model in baselines.values()]
    extra = {}
    if gan is not None:
        judged = evaluate_gan(data, gan, n_draws=n_draws)
        rows.append(judged["gan_generator"])
        rows.append(judged["decision_noul"])
        extra = {"decision_train_check": decision_label_report(data, gan, n_draws=n_draws)}
    return {"dataset": data.name, "source": data.source, "models": rows, **extra}


def finite_metric(value) -> bool:
    return isinstance(value, (int, float)) and value == value and not math.isinf(value)
