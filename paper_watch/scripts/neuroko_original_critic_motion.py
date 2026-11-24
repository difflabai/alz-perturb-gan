"""Compare released 0.11.1 v11 critic movement with the published v12 probes at three fixed rates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
from statistics import median

from compare_neuroko_v12 import GROUPS, write_frozen
from neuroko_critic_motion import measure, probe_folds
from neuroko_rate_noise import NOISY_PRIOR, REPLICATION_SEEDS, critic_motion_probe


def run_gpu(args, gpu):
    if version("particlegan") != "0.11.1":
        raise ValueError("The original v11 measurement requires ParticleGAN 0.11.1")
    references = {}
    for rate, suffix in ((.021, ""), (.0035, "_check"), (.0046, "_interpolated")):
        path = args.references / f"neuroko_v12_critic_motion{suffix}.json"
        critic_motion_probe(path, rate)
        source = json.loads(path.read_text())
        references[rate] = {(c["seed"], c["group"]): c for c in source["cells"]}

    cells = []
    for i, group, x, labels, probes, data_hash, probe_hash in probe_folds(args, gpu):
        for seed in REPLICATION_SEEDS:
            if (seed + i) % 2 != gpu:
                continue
            inputs = (x, labels, probes, seed, "unified_v11", probes.device, .021, NOISY_PRIOR["prior_reg"])
            original, _ = measure(*inputs)
            if seed == REPLICATION_SEEDS[0] and group == GROUPS[0]:
                plain, _ = measure(*inputs, monitor=False)
                if plain["final_parameters_sha256"] != original["final_parameters_sha256"]:
                    raise ValueError("Observing the original v11 critic changed its training trajectory")
            ratios = {}
            for rate, records in references.items():
                ref = records[seed, group]
                if (ref["projected_data_sha256"], ref["probe_indices_sha256"], ref["gpu"]) != (data_hash, probe_hash, gpu):
                    raise ValueError("Original v11 must use the same projected cells, probes and GPU as v12")
                v12 = ref["rows"]["unified_v12"]
                if any(original[k] != v12[k] for k in ("initial_parameters_sha256", "training_batches_sha256")):
                    raise ValueError("The original v11 and v12 fits must have matching initial parameters and batches")
                ratios[str(rate)] = median(v12["centered_rms"]) / median(original["centered_rms"])
            cell = dict(seed=seed, group=group, gpu=gpu, original_v11=original, projected_data_sha256=data_hash,
                        probe_indices_sha256=probe_hash, v12_to_original_by_critic_lr=ratios)
            write_frozen(args.output / f"s{seed}-{group}.json", cell)
            cells.append(cell)
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "references", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--window-cells", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "2"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(run_gpu, args, gpu) for gpu in (0, 1)]
        cells = sorted((row for job in jobs for row in job.result()), key=lambda row: (row["seed"], row["group"]))
    ratios = {rate: median(c["v12_to_original_by_critic_lr"][rate] for c in cells)
              for rate in cells[0]["v12_to_original_by_critic_lr"]}
    write_frozen(args.output / "result.json", dict(original_v11_release="0.11.1", v12_release="0.12.0",
                                                   settings=NOISY_PRIOR, original_v11_critic_lr=.021,
                                                   median_ratios=ratios, cells=cells))
    print("V12 to original v11, median centered critic movement", ratios, flush=True)


if __name__ == "__main__":
    main()
