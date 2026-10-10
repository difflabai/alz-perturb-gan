"""Expression and phenocopy metrics that do not require Arc's cell-eval package.

DES-like overlap and PDS-like discrimination follow the quantities named in the
Virtual Cell Challenge (differential-expression agreement, and whether the
predicted effect is closer to the matching perturbation than to the others).
They are local definitions for this experiment, not a claim to reproduce
cell-eval's exact implementation.
"""

from __future__ import annotations

import math

import torch


def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float((pred - true).abs().mean())


def l2(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(pred - true))


def pearson_delta(pred: torch.Tensor, true: torch.Tensor, control: torch.Tensor) -> float:
    """Correlation of predicted and observed deviations from control."""
    left = pred - control
    right = true - control
    left = left - left.mean()
    right = right - right.mean()
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denom) < 1e-8:
        return math.nan
    return float((left * right).sum() / denom)


def des_overlap(pred: torch.Tensor, true: torch.Tensor, control: torch.Tensor, k: int = 10) -> float:
    """Jaccard overlap of the top-k genes by absolute deviation from control."""
    pred_delta = (pred - control).abs()
    true_delta = (true - control).abs()
    if float(pred_delta.max()) < 1e-8 or float(true_delta.max()) < 1e-8:
        return 0.0
    k = min(k, pred.shape[0])
    pred_top = set(torch.topk(pred_delta, k).indices.tolist())
    true_top = set(torch.topk(true_delta, k).indices.tolist())
    union = pred_top | true_top
    if not union:
        return 0.0
    return len(pred_top & true_top) / len(union)


def perturbation_discrimination(pred_delta: torch.Tensor, true_delta: torch.Tensor) -> float:
    """Mean normalized rank of the matching perturbation.

    For each prediction, rank true perturbation profiles by L2 distance.
    Score 1 means the true profile is the unique nearest. Score 0 means it is
    the unique farthest. Ties share the average of the ranks they occupy.
    """
    if pred_delta.shape != true_delta.shape or pred_delta.shape[0] < 2:
        return math.nan
    count = pred_delta.shape[0]
    scores = []
    for index in range(count):
        distance = torch.linalg.vector_norm(true_delta - pred_delta[index], dim=-1)
        true_distance = distance[index]
        better = int((distance < true_distance - 1e-8).sum())
        tied = int((torch.abs(distance - true_distance) <= 1e-8).sum())
        # Average 0-based rank inside the tie, then normalize by count-1.
        avg_rank = better + (tied - 1) / 2
        scores.append(1.0 - avg_rank / (count - 1))
    return float(sum(scores) / len(scores))


def expression_report(
    pred: torch.Tensor,
    true: torch.Tensor,
    control: torch.Tensor,
    *,
    k: int = 10,
) -> dict[str, float]:
    """Macro-average per-profile metrics. `pred` and `true` are [N, G]."""
    deltas_pred = []
    deltas_true = []
    maes = []
    l2s = []
    pears = []
    overlaps = []
    for index in range(pred.shape[0]):
        maes.append(mae(pred[index], true[index]))
        l2s.append(l2(pred[index], true[index]))
        pears.append(pearson_delta(pred[index], true[index], control))
        overlaps.append(des_overlap(pred[index], true[index], control, k=k))
        deltas_pred.append(pred[index] - control)
        deltas_true.append(true[index] - control)
    finite = [value for value in pears if value == value]
    return {
        "mae": float(sum(maes) / len(maes)),
        "l2": float(sum(l2s) / len(l2s)),
        "pearson_delta": float(sum(finite) / len(finite)) if finite else math.nan,
        "des_overlap": float(sum(overlaps) / len(overlaps)),
        "pds": perturbation_discrimination(torch.stack(deltas_pred), torch.stack(deltas_true)),
        "n_profiles": float(pred.shape[0]),
    }


def phenocopy_rank(scores: torch.Tensor, true_index: int) -> dict[str, float]:
    """Rank a target perturbation among candidates. Higher score is better.

    Top-1 requires a strictly unique best score on the true index. MRR uses
    the average rank inside a tie, so an all-tied score is not counted as a hit.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be a vector")
    true_score = scores[true_index]
    better = int((scores > true_score + 1e-8).sum())
    tied = int((torch.abs(scores - true_score) <= 1e-8).sum())
    avg_rank = better + (tied + 1) / 2  # 1-based average rank
    return {
        "top1": 1.0 if better == 0 and tied == 1 else 0.0,
        "mrr": 1.0 / avg_rank,
        "recall_at_3": 1.0 if avg_rank <= 3 else 0.0,
        "rank": float(avg_rank),
    }


def mean_phenocopy(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"top1": math.nan, "mrr": math.nan, "recall_at_3": math.nan, "n_queries": 0.0}
    keys = ("top1", "mrr", "recall_at_3")
    summary = {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}
    summary["n_queries"] = float(len(rows))
    return summary
