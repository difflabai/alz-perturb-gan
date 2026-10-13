"""Bounded-memory, perturbation-held-out benchmark on real Arc h5ad data.

No downloads are implicit. All genes and all cells contribute to pseudobulks;
only the single-cell training/evaluation samples are capped. Eval outcomes are
kept out of the tensors consumed by the trainer and baseline fits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .baselines import Additive, LinearModel, MeanEffect, NoChange
from .data import perturbation_set_from_groups
from .gan import train_gan_decision
from .harness import decision_pair_report, evaluate_baseline, evaluate_gan


def transform_expression(block, input_scale: str):
    """Normalize raw counts to 10,000 per cell then log1p, or keep log data."""
    from scipy import sparse

    block = block.astype(np.float32)
    values = block.data if sparse.issparse(block) else block
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("expression must be finite and nonnegative")
    if input_scale == "log1p":
        return block
    if input_scale != "counts":
        raise ValueError("input_scale must be counts or log1p")
    if not np.allclose(values, np.rint(values), atol=1e-4, rtol=0):
        raise ValueError("counts input contains fractional values; inspect the source scale")
    total = np.asarray(block.sum(axis=1)).reshape(-1)
    scale = np.divide(10000.0, total, out=np.zeros_like(total), where=total > 0)
    if sparse.issparse(block):
        block = block.multiply(scale[:, None]).tocsr()
        block.data = np.log1p(block.data)
    else:
        block = np.log1p(block * scale[:, None])
    return block


def load_real_data(
    path: str | Path,
    *,
    eval_fraction: float = 0.2,
    split_seed: int = 42,
    cell_seed: int = 43,
    train_cells: int = 128,
    eval_cells: int = 128,
    chunk_size: int = 1024,
    input_scale: str = "counts",
):
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be in (0, 1)")
    if min(train_cells, eval_cells, chunk_size) < 1:
        raise ValueError("cell caps and chunk size must be positive")
    source = ad.read_h5ad(path, backed="r")
    try:
        if "target_gene" not in source.obs or source.obs.target_gene.isna().any():
            raise ValueError("expected nonmissing obs['target_gene']")
        labels = source.obs.target_gene.astype(str).to_numpy()
        controls = np.isin(np.char.lower(labels.astype(str)),
                             ["non-targeting", "non-targeting control", "control"])
        if not controls.any():
            raise ValueError("no non-targeting controls")
        labels[controls] = "non-targeting"
        names = sorted(set(labels) - {"non-targeting"})
        if len(names) < 2:
            raise ValueError("need at least two perturbations")
        perm = np.random.default_rng(split_seed).permutation(names)
        n_eval = min(len(names) - 1, max(1, round(len(names) * eval_fraction)))
        eval_names = sorted(perm[:n_eval].tolist())
        train_names = sorted(set(names) - set(eval_names))
        order = ["non-targeting"] + names
        groups = {name: np.flatnonzero(labels == name) for name in order}
        rng = np.random.default_rng(cell_seed)
        wanted = np.zeros(source.n_obs, dtype=bool)
        selected = {}
        for name in order:
            cap = train_cells if name in train_names else eval_cells
            selected[name] = np.sort(rng.choice(groups[name], min(cap, len(groups[name])), replace=False))
            wanted[selected[name]] = True
        sums = {name: np.zeros(source.n_vars, dtype=np.float64) for name in order}
        samples = {name: [] for name in order}
        for start in range(0, source.n_obs, chunk_size):
            stop = min(start + chunk_size, source.n_obs)
            block = transform_expression(source.X[start:stop], input_scale)
            local_labels = labels[start:stop]
            for name in np.unique(local_labels):
                subset = local_labels == name
                sums[name] += np.asarray(block[subset].sum(axis=0, dtype=np.float64)).reshape(-1)
                keep = subset & wanted[start:stop]
                if keep.any():
                    sample = block[keep]
                    samples[name].append(sample.toarray() if sparse.issparse(sample) else sample)
            if start % (chunk_size * 50) == 0:
                print(f"aggregate {stop}/{source.n_obs} cells", flush=True)
        profiles = {name: torch.from_numpy((sums[name] / len(groups[name])).astype(np.float32)) for name in order}
        sampled = {name: np.concatenate(samples[name]) for name in order}
        data = perturbation_set_from_groups(
            name="arc-vcc-2025-train-local-holdout",
            source="virtual_cell_h5ad",
            gene_names=source.var_names.astype(str).tolist(),
            control_mean=profiles["non-targeting"],
            profiles={name: profiles[name] for name in names},
            train_names=train_names,
            eval_names=eval_names,
            cells={name: torch.from_numpy(sampled[name]) for name in train_names},
            notes="Seeded identity holdout of the public training release; all genes; unpaired cells.",
        )
        assert set(data.cell_perturbation).isdisjoint(eval_names)
        real_order = ["non-targeting"] + eval_names
        real = ad.AnnData(
            X=np.concatenate([sampled[name] for name in real_order]),
            obs=pd.DataFrame(
                {"target_gene": [name for name in real_order for _ in selected[name]]},
                index=[str(source.obs_names[i]) for name in real_order for i in selected[name]],
            ),
            var=pd.DataFrame(index=data.gene_names),
        )
        data.metadata.update({
            "split_seed": split_seed, "cell_seed": cell_seed,
            "train_names": train_names, "eval_names": eval_names,
            "n_cells_source": source.n_obs, "n_genes": source.n_vars,
            "counts_by_perturbation": {name: len(groups[name]) for name in order},
            "train_cells_sampled": len(data.cell_perturbation),
            "train_cells_cap": train_cells, "eval_cells_cap": eval_cells,
            "input_scale": input_scale,
            "expression_scale": "log1p(normalize_total(counts, 10000))" if input_scale == "counts" else "source X log1p, unchanged",
            "evaluation": "local pseudobulks use ALL held-out cells; cell-eval uses the fixed sampled cells",
            "embedding": "one-hot identity; no pretrained gene features; unseen columns receive no training signal",
            "control": "all non-targeting cells contribute to shared control mean; sampled controls are available to all models",
        })
        return data, real
    finally:
        source.file.close()


def select_device(requested: str) -> tuple[str, dict]:
    """Select greatest free memory before CUDA initialization; cap our allocator."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        status = {"nvidia_smi_returncode": result.returncode, "nvidia_smi": (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        status = {"nvidia_smi_error": str(exc)}
        result = None
    if requested == "cpu":
        status.update({"torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
                       "cuda_available": torch.cuda.is_available(), "cpu_threads": torch.get_num_threads()})
        return "cpu", status
    if result is None or result.returncode:
        raise RuntimeError("GPU selection failed: " + json.dumps(status))
    rows = [line.split(",") for line in result.stdout.strip().splitlines()]
    row = max(rows, key=lambda item: int(item[2].strip()))
    os.environ["CUDA_VISIBLE_DEVICES"] = row[0].strip()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch cannot initialize the selected GPU")
    props = torch.cuda.get_device_properties(0)
    cap = 19 * 1024**3
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / props.total_memory), 0)
    status.update({"selected_uuid": row[0].strip(), "selected_name": props.name,
                   "free_mib_before_run": int(row[2]), "allocator_limit_gib": 19})
    return "cuda:0", status


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_metrics(real, predicted, directory: Path):
    from cell_eval import MetricsEvaluator

    directory.mkdir(parents=True, exist_ok=True)
    evaluator = MetricsEvaluator(
        adata_pred=predicted, adata_real=real.copy(), pert_col="target_gene",
        control_pert="non-targeting", num_threads=4,
        outdir=str(directory), pdex_kwargs={"is_log1p": True},
    )
    results, aggregate = evaluator.compute(profile="vcc")
    results.write_csv(directory / "results.csv")
    aggregate.write_csv(directory / "agg_results.csv")
    return aggregate.to_dicts()


def prediction_anndata(data, real, profiles, gan=None, *, cells_per=128):
    import anndata as ad
    import pandas as pd

    controls = real[real.obs.target_gene == "non-targeting"].X.copy()
    blocks = [controls]
    labels = ["non-targeting"] * len(controls)
    lookup = data.by_name()
    for name in data.metadata["eval_names"]:
        if gan is None:
            block = profiles[name].numpy()[None].repeat(cells_per, axis=0)
        else:
            with torch.no_grad():
                control = data.control_mean.to(gan.device).expand(cells_per, -1)
                embedding = lookup[name].embedding.to(gan.device).expand(cells_per, -1)
                latent, _ = gan.prior.sample(cells_per)
                block = gan.generator(latent, control, embedding).cpu().numpy()
        blocks.append(np.maximum(block, 0))
        labels.extend([name] * len(block))
    return ad.AnnData(np.concatenate(blocks), obs=pd.DataFrame(
        {"target_gene": labels}, index=[f"pred-{i}" for i in range(len(labels))]),
        var=pd.DataFrame(index=data.gene_names))


def decision_heldout_report(data, gan, seed, *, n_draws=8):
    """Held-out pseudobulk targets vs every library candidate, scored on predicted profiles.

    Targets are eval perturbations' pseudobulks (all their cells); their
    identities and cells were never used in fitting.
    """
    library = [item.name for item in data.named()]
    torch.manual_seed(seed)
    predicted = gan.predict_profiles(data, library, n_draws=n_draws)
    report = decision_pair_report(data, gan, data.named("eval"), dict(zip(library, predicted)),
                                  seed=seed, split="eval_targets")
    report["note"] = "label = (candidate == target perturbation); candidates are generator-averaged profiles."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--input-scale", choices=["counts", "log1p"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--train-cells", type=int, default=128)
    parser.add_argument("--eval-cells", type=int, default=128)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--cell-eval", action="store_true")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    write_json(args.run_dir / "command.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    device, hardware = select_device(args.device)
    write_json(args.run_dir / "hardware.json", hardware)
    data, real = load_real_data(args.data, split_seed=args.seed, cell_seed=args.seed + 1,
                               train_cells=args.train_cells, eval_cells=args.eval_cells,
                               eval_fraction=args.eval_fraction, input_scale=args.input_scale)
    load_seconds = time.perf_counter() - started
    write_json(args.run_dir / "split.json", data.metadata)
    real.write_h5ad(args.run_dir / "real_eval.h5ad", compression="gzip")
    names = [item.name for item in data.named()]
    models = []
    predictions = {}
    runtime = {"load_seconds": load_seconds}
    for model in [NoChange(), MeanEffect(), Additive(), LinearModel(n_components=32, device=device)]:
        tick = time.perf_counter()
        model.fit(data)
        row = evaluate_baseline(data, model)
        pred = model.predict(data, names)
        predictions[model.name] = dict(zip(names, pred))
        runtime[model.name + "_seconds"] = time.perf_counter() - tick
        models.append(row)
        print(json.dumps(clean_json(row)), flush=True)
    write_json(args.run_dir / "baseline_results.json", models)

    def progress(step, losses):
        if not all(math.isfinite(value) for value in losses.values()):
            raise RuntimeError(f"nonfinite training loss at step {step}: {losses}")
        with (args.run_dir / "training.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": step, **losses}) + "\n")
        if step == 1 or step % 100 == 0:
            print(f"step {step}/{args.steps} elapsed={time.perf_counter()-tick:.1f}s {losses}", flush=True)

    tick = time.perf_counter()
    gan = train_gan_decision(data, device=device, seed=args.seed, steps=args.steps,
                             batch_size=args.batch_size, hidden=args.hidden, lr=args.lr,
                             n_particles=256, z_dim=16, progress=progress)
    runtime["gan_train_seconds"] = time.perf_counter() - tick
    torch.save({"generator": gan.generator.state_dict(), "critic": gan.critic.state_dict(),
                "decision": gan.decision.state_dict(), "prior": gan.prior.state_dict(),
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}},
               args.run_dir / "checkpoint.pt")
    tick = time.perf_counter()
    torch.manual_seed(args.seed + 2)
    judged = evaluate_gan(data, gan, n_draws=args.draws)
    decision_report = decision_heldout_report(data, gan, args.seed + 3, n_draws=args.draws)
    models.extend(judged.values())
    predictions["gan_generator"] = dict(zip(names, gan.predict_profiles(data, names, n_draws=args.draws)))
    runtime["gan_eval_seconds"] = time.perf_counter() - tick
    write_json(args.run_dir / "local_results.json", models)
    torch.save({"names": names, "predictions": {name: torch.stack([pred[n] for n in names]) for name, pred in predictions.items()}}, args.run_dir / "predictions.pt")
    official = {}
    if args.cell_eval:
        for name, pred in predictions.items():
            tick = time.perf_counter()
            predicted = prediction_anndata(data, real, pred, gan if name == "gan_generator" else None, cells_per=args.eval_cells)
            predicted.write_h5ad(args.run_dir / (name + "_eval.h5ad"), compression="gzip")
            try:
                official[name] = official_metrics(real, predicted, args.run_dir / ("cell_eval_" + name))
            except Exception as exc:
                official[name] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"cell-eval {name}: {official[name]}", flush=True)
            runtime["cell_eval_" + name + "_seconds"] = time.perf_counter() - tick
            write_json(args.run_dir / "cell_eval_results.json", official)
    tick = time.perf_counter()
    data_hash = sha256(args.data)
    runtime["hash_seconds"] = time.perf_counter() - tick
    runtime["total_seconds"] = time.perf_counter() - started
    report = {"completed_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "hardware": hardware, "dataset": data.metadata, "models": models,
              "cell_eval": official, "runtime": runtime, "data_sha256": data_hash,
              "decision_heldout": decision_report,
              "training_summary": {key: {"mean": float(np.mean(values)), "last": values[-1]} for key, values in gan.history.items()},
              "versions": {pkg: importlib.metadata.version(pkg) for pkg in ["torch", "particlegan", "anndata", "numpy"]}}
    if args.cell_eval:
        report["versions"]["cell-eval"] = importlib.metadata.version("cell-eval")
    if device.startswith("cuda"):
        report["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        report["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    write_json(args.run_dir / "results.json", report)
    print(json.dumps(clean_json(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
