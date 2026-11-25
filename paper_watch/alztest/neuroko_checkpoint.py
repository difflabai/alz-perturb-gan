"""Sample a NeuroKO checkpoint or verify its endpoint against the published benchmark."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from .harness import ConditionalPGAN, TrainingSettings


REFERENCE = Path(__file__).resolve().parents[1] / "notes" / "neuroko_v12_shared_smoothing_40000.json"


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError(f"Unsupported checkpoint format {checkpoint['format_version']}")
    receipt = checkpoint["receipt"]
    if version("particlegan") != receipt["particlegan"]:
        raise ValueError(f"Install particlegan=={receipt['particlegan']} for this checkpoint")
    model = ConditionalPGAN(checkpoint["projection"]["scale"].numel(), len(checkpoint["conditions"]),
                            seed=receipt["seed"], learning_rate=receipt["lr"], recipe_name=receipt["recipe"],
                            settings=TrainingSettings(**receipt["settings"]), device=device)
    model.G.load_state_dict(checkpoint["served"]["generator"])
    model.D.load_state_dict(checkpoint["served"]["critic"])
    model.prior.load_state_dict(checkpoint["served"]["prior"])
    return model, checkpoint


def require_equal(actual, expected, field):
    if actual != expected:
        raise ValueError(f"Mismatch in {field}")


def verify(model, checkpoint, directory, window_cells, reference):
    from .neuroko import LAYOUT, load_comparison, open_screen, project_fold
    from .neuroko_duration import effective_rank
    from .neuroko_lr import DRAWS, score

    receipt = checkpoint["receipt"]
    with ExitStack() as stack:
        screens = [stack.enter_context(open_screen(directory, name, window_cells)) for name in LAYOUT]
        conditions, panels = load_comparison(screens, 100, 160, 6)
    require_equal(list(conditions), checkpoint["conditions"], "condition order")
    panel = panels["iNeurons"]
    fit = np.isin(panel.groups, receipt["fit_groups"])
    test = np.isin(panel.groups, receipt["score_groups"])
    x, y, projection = project_fold(panel.x[fit], panel.x[test], return_projection=True)
    saved = checkpoint["projection"]
    for key, value in projection.items():
        require_equal(np.array_equal(value, saved[key].numpy()), True, f"PCA {key}")
    require_equal(panel.features[projection["feature_indices"]].tolist(), saved["feature_names"], "genes")
    digest = sha256()
    for array in (x, panel.labels[fit], y, panel.labels[test]):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    require_equal(digest.hexdigest(), receipt["projected_data_sha256"], "projected data")
    seed = receipt["seed"]
    samples = [model.sample(i, DRAWS, seed=seed + 100 + i) for i in range(len(conditions))]
    observed = [y[panel.labels[test] == i] for i in range(len(conditions))]
    targets = score(samples, observed, conditions)
    ranks = {target: effective_rank(draws) for target, draws in zip(conditions[1:], samples[1:])}
    require_equal(targets, receipt["targets"], "rescored target metrics")
    require_equal(ranks, receipt["effective_rank"], "rescored effective rank")
    if reference is not None:
        cells = json.loads(reference.read_text())["cells"]
        matches = [row for row in cells if row["seed"] == seed and [row["group"]] == receipt["score_groups"]]
        if len(matches) != 1:
            raise ValueError("Expected one matching seed and group in the reference results")
        cell = matches[0]
        arm = cell["arms"]["v12"]
        require_equal(targets, arm["endpoint_targets"], "published target metrics")
        require_equal(ranks, arm["endpoint_effective_rank"], "published effective rank")
        require_equal(receipt["initial_rates"], arm["rates"], "published learning rates")
        require_equal(receipt["fixed_lr_optimizer_steps"], arm["verified_optimizer_steps"], "optimizer steps")
        require_equal(digest.hexdigest(), cell["fingerprints"]["data"], "published projected data")
        require_equal(receipt["initial_parameters_sha256"], cell["fingerprints"]["initialization"], "initialization")
        trace = receipt["trace"]
        hashes = [row["training_batches_sha256"] for row in trace]
        require_equal(hashes, cell["fingerprints"]["training_batches"], "minibatches")
        curve = [{key: value for key, value in row.items() if key != "training_batches_sha256"} for row in trace]
        require_equal(curve, arm["curve"], "published curve")
    return {"seed": seed, "updates": receipt["updates"], "targets": targets,
            "mean_energy_distance": mean(row["energy_distance"] for row in targets.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="write standardized principal component draws as a NumPy array")
    sample.add_argument("--condition", required=True)
    sample.add_argument("--draws", default=256, type=int)
    sample.add_argument("--seed", default=123, type=int)
    sample.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("verify", help="recompute endpoint metrics on the reference CUDA stack")
    check.add_argument("--data", type=Path, required=True)
    check.add_argument("--window-cells", type=int, help="use 10000 for the original Zenodo files")
    check.add_argument("--reference", type=Path, default=REFERENCE)
    for cmd, device in ((sample, "cpu"), (check, "cuda")):
        cmd.add_argument("checkpoint", type=Path)
        cmd.add_argument("--device", choices=("cpu", "cuda"), default=device)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    device = torch.device("cuda", torch.cuda.current_device()) if args.device == "cuda" else torch.device("cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    if args.command == "verify":
        print(json.dumps(verify(model, checkpoint, args.data, args.window_cells, args.reference), indent=2))
    else:
        conditions = checkpoint["conditions"]
        if args.condition not in conditions or args.draws < 1:
            parser.error(f"Provide a positive draw count and a condition from {conditions}")
        draws = model.sample(conditions.index(args.condition), args.draws, args.seed)
        with args.out.open("wb") as output:
            np.save(output, draws, allow_pickle=False)


if __name__ == "__main__":
    main()
