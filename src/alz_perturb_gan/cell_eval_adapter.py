"""Optional hook for Arc Institute's cell-eval.

The smoke run and the unit tests do not need it. If `cell_eval` is installed
and exports `MetricsEvaluator`, `score_anndata` calls that class. Otherwise
the function returns a structured reason and the local metrics in `metrics.py`
remain the ones this experiment reports.
"""

from __future__ import annotations

import importlib
from importlib.util import find_spec


def cell_eval_status() -> dict:
    spec = find_spec("cell_eval")
    if spec is None:
        return {
            "available": False,
            "reason": "cell_eval is not installed in this environment.",
            "install": "pip install cell-eval",
            "cli": "cell-eval run -ap pred.h5ad -ar real.h5ad --profile full",
        }
    return {"available": True, "origin": spec.origin}


def score_anndata(pred, real, **kwargs) -> dict:
    """Score two AnnData objects if cell-eval's Python class is importable.

    `pred` and `real` are passed through. This function does not write files
    and does not download the Virtual Cell data.
    """
    status = cell_eval_status()
    if not status["available"]:
        return {**status, "scored": False}
    module = importlib.import_module("cell_eval")
    evaluator_cls = getattr(module, "MetricsEvaluator", None)
    if evaluator_cls is None:
        return {
            **status,
            "scored": False,
            "reason": (
                "cell_eval imported but does not export MetricsEvaluator. "
                "Use the cell-eval CLI on h5ad files. This package does not shell out."
            ),
        }
    try:
        evaluator = evaluator_cls(adata_pred=pred, adata_real=real, **kwargs)
        results = evaluator.compute() if hasattr(evaluator, "compute") else evaluator.evaluate()
    except Exception as exc:  # the optional API has moved before; keep the local harness running
        return {
            **status,
            "scored": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {**status, "scored": True, "results": results}
