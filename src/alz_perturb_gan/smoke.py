"""CPU smoke run: baselines, the ParticleGAN decision model, and HyperGAN develop.

Uses the synthetic dataset unless `--data` points at a local h5ad file.
Nothing is downloaded.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import __version__
from .baselines import fit_baselines
from .cell_eval_adapter import cell_eval_status
from .data import load_virtual_cell_h5ad, make_synthetic
from .gan import train_gan_decision
from .harness import compare_all
from .hypergan_bridge import RC2_FINDINGS, describe_hypergan, predict_conditions, train_condition_gan
from .metrics import expression_report


def _jsonable(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _hypergan_expression(data, run_dir: Path, *, draws: int) -> dict | None:
    names = [item.name for item in data.named("eval")]
    if not names:
        return None
    lookup = data.by_name()
    condition = []
    true = []
    for name in names:
        item = lookup[name]
        condition.append(torch_cat(data.control_mean, item.embedding))
        true.append(item.profile)
    import torch

    pred = predict_conditions(run_dir, torch.stack(condition), draws=draws)
    return expression_report(pred, torch.stack(true), data.control_mean, k=max(1, min(10, data.n_genes // 2)))


def torch_cat(control, embedding):
    import torch

    return torch.cat([control, embedding], dim=-1)


def run_smoke(
    *,
    steps: int = 40,
    hypergan_steps: int = 80,
    seed: int = 0,
    data_path: str | None = None,
    run_dir: str | Path = "runs/hypergan-smoke",
) -> dict:
    if data_path:
        data = load_virtual_cell_h5ad(data_path, seed=seed)
    else:
        data = make_synthetic(seed=seed)
    baselines = fit_baselines(data)
    gan = train_gan_decision(data, steps=steps, seed=seed)
    report = compare_all(data, baselines, gan)
    report["package_version"] = __version__
    report["gan_steps"] = steps
    report["gan_final"] = {key: values[-1] for key, values in gan.history.items()}
    report["cell_eval"] = cell_eval_status()
    report["hypergan_rc2"] = RC2_FINDINGS
    info = describe_hypergan()
    report["hypergan"] = {"probe": info, "steps": hypergan_steps}
    if info["status"] == "develop":
        try:
            trained = train_condition_gan(data, run_dir, steps=hypergan_steps, batch_size=8)
            trained["expression"] = _jsonable(_hypergan_expression(data, Path(run_dir), draws=4))
            report["hypergan"]["train"] = trained
        except Exception as exc:
            report["hypergan"]["train"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    report["metadata"] = data.metadata
    report["paired_cells"] = data.paired_cells
    report["n_train"] = len(data.named("train"))
    report["n_eval"] = len(data.named("eval"))
    report["n_genes"] = data.n_genes
    return _jsonable(report)


def format_table(report: dict) -> str:
    header = f"{'model':<16} {'mae':>8} {'pearson':>8} {'des':>8} {'pds':>8} {'top1':>8} {'mrr':>8}"
    lines = [header, "-" * len(header)]
    for row in report["models"]:
        expr = row.get("expression") or {}
        pheno = row.get("phenocopy") or {}

        def cell(value, digits=3):
            if value is None:
                return f"{'n/a':>8}"
            return f"{value:8.{digits}f}"

        lines.append(
            f"{row['model']:<16} {cell(expr.get('mae'))} {cell(expr.get('pearson_delta'))} "
            f"{cell(expr.get('des_overlap'))} {cell(expr.get('pds'))} "
            f"{cell(pheno.get('top1'))} {cell(pheno.get('mrr'))}"
        )
    hyper = ((report.get("hypergan") or {}).get("train") or {}).get("expression")
    if hyper:
        lines.append(
            f"{'hypergan':<16} {cell(hyper.get('mae'))} {cell(hyper.get('pearson_delta'))} "
            f"{cell(hyper.get('des_overlap'))} {cell(hyper.get('pds'))} "
            f"{'n/a':>8} {'n/a':>8}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU smoke run on the synthetic perturbation set.")
    parser.add_argument("--steps", type=int, default=40, help="ParticleGAN decision-model updates")
    parser.add_argument("--hypergan-steps", type=int, default=80, help="HyperGAN develop updates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", default=None, help="Optional local h5ad. Default: synthetic data.")
    parser.add_argument("--output", default="results/smoke_results.json")
    parser.add_argument("--run-dir", default="runs/hypergan-smoke")
    args = parser.parse_args(argv)
    report = run_smoke(
        steps=args.steps,
        hypergan_steps=args.hypergan_steps,
        seed=args.seed,
        data_path=args.data,
        run_dir=args.run_dir,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("LEARNING EXPERIMENT. Not a medical or scientific validation.")
    print(format_table(report))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
