"""Measure how much latent noise changes training versus sampling in the fixed-rate v12 benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import mean

from compare_neuroko_v12 import GROUPS, fit, receipt, reusable_fit, write_frozen
from neuroko_evaluation_draws import EVALUATION_SEEDS, METRICS
from neuroko_rate_noise import NOISY_PRIOR, REPLICATION_SEEDS
from retune_neuroko_v12 import BASE, RELEASE, V12


TRIALS = (dict(BASE, name="previous-regularized", prior_reg=0.1), NOISY_PRIOR)
SIGMAS = (0.025, 0.15)
HASHES = ("projected_data_sha256", "initial_parameters_sha256", "training_batches_sha256")


def replay(args, seed, group, gpu):
    draws = {}
    fingerprints = None
    for trial, sigma, alternate in zip(TRIALS, SIGMAS, reversed(SIGMAS), strict=True):
        arm = (RELEASE, V12, trial)
        reference = json.loads(receipt(args.reference, "validation", *arm, seed, group).read_text())
        path = receipt(args.output, "validation", *arm, seed, group)
        path = path.with_stem(f"{path.stem}-resampled-noise{alternate}")
        if not reusable_fit(path, require_constant_lr=True):
            fit(args, "validation", seed, group, gpu, [arm],
                options=["--evaluation-seeds", *map(str, EVALUATION_SEEDS), "--score-noise", str(alternate)])
        row = json.loads(path.read_text())
        for field in (*HASHES, "settings", "targets", "means", "seed", "score_groups", "fit_groups", "visible_devices",
                      "initial_rates", "final_rates", "recipe", "particlegan", "constant_lr_verified_updates"):
            if row[field] != reference[field]:
                raise ValueError(f"{field} differs from the original fit at {path}")
        if (row["constant_lr_verified_updates"] != 1200 or row["initial_rates"] != row["final_rates"]
                or row["settings"]["prior_sigma"] != sigma or row["score_noise"] != alternate
                or row["evaluation_seeds"] != EVALUATION_SEEDS or row["visible_devices"] != str(gpu)):
            raise ValueError(f"Incorrect rate, noise or sampling settings at {path}")
        hashes = {field: row[field] for field in HASHES}
        if fingerprints is not None and hashes != fingerprints:
            raise ValueError(f"Different training inputs across the paired fits at {path}")
        fingerprints = hashes
        canonical = str(seed + 100)
        original = {"mean_energy_distance": row["means"]["energy_distance"],
                    "spread_ratio": row["means"]["spread_ratio"], "effective_rank": mean(row["effective_rank"].values())}
        draws[str(sigma)] = {str(sigma): {canonical: original, **row["evaluation_draws"]},
                             str(alternate): row["alternate_noise_draws"]}
        expected = {canonical, *map(str, EVALUATION_SEEDS)}
        if any(set(records) != expected for records in draws[str(sigma)].values()):
            raise ValueError(f"Missing paired evaluation draw sets at {path}")
    return dict(seed=seed, group=group, visible_devices=str(gpu), hashes=fingerprints, draws=draws)


def summarize(cells, *, repeated):
    lo, hi = map(str, SIGMAS)
    contrasts = []
    matrices = []
    for cell in cells:
        seeds = EVALUATION_SEEDS if repeated else [cell["seed"] + 100]
        matrix = {t: {s: {m: mean(draws[str(seed)][m] for seed in seeds) for m in METRICS}
                      for s, draws in by_sample.items()} for t, by_sample in cell["draws"].items()}
        matrices.append(matrix)
        a, b, c, d = (matrix[t][s]["mean_energy_distance"] for t, s in ((lo, lo), (lo, hi), (hi, lo), (hi, hi)))
        training_low, training_high = c - a, d - b
        sampling_low, sampling_high = b - a, d - c
        contrasts.append(dict(seed=cell["seed"], group=cell["group"], native_change=d - a,
                              training_at_low=training_low, training_at_high=training_high,
                              sampling_after_low=sampling_low, sampling_after_high=sampling_high,
                              training_contribution=(training_low + training_high) / 2,
                              sampling_contribution=(sampling_low + sampling_high) / 2))
    means = {t: {s: {m: mean(x[t][s][m] for x in matrices) for m in METRICS} for s in (lo, hi)} for t in (lo, hi)}
    return dict(means_by_training_and_sampling_sigma=means, paired_energy_contrasts=contrasts,
                mean_energy_contrasts={k: mean(c[k] for c in contrasts) for k in contrasts[0] if k not in ("seed", "group")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "reference", "output"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--window-cells", type=int)
    args = parser.parse_args()
    args.require_constant_lr = True
    for field in ("data", "wheels", "reference", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(release=RELEASE, recipe=V12, trials=TRIALS, sigmas=SIGMAS, seeds=REPLICATION_SEEDS,
                    groups=GROUPS, additional_draw_seed_bases=EVALUATION_SEEDS, updates=1200, draws_per_target=256,
                    rule="Replay both release-rate v12 settings with prior weight 0.1 on all nine training pairs. "
                         "Change only Gaussian sampling noise in each fitted model. Compare the two training "
                         "widths at each sampling width, and the two sampling widths at each training width. "
                         "Average over the opposite factor for additive contributions to the diagonal difference. "
                         "Score original draws and eight additional common sets separately. Diagnostic only.")
    write_frozen(args.output / "protocol.json", protocol)
    with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
        jobs = []
        for seed in REPLICATION_SEEDS:
            for i, group in enumerate(GROUPS):
                gpu = (seed + i) % 2
                jobs.append((first, second)[gpu].submit(replay, args, seed, group, gpu))
        cells = [job.result() for job in jobs]
    result = dict(protocol=protocol, canonical=summarize(cells, repeated=False),
                  repeated=summarize(cells, repeated=True), cells=cells)
    write_frozen(args.output / "result.json", result)
    print("Repeated draw energy contrasts", result["repeated"]["mean_energy_contrasts"], flush=True)


if __name__ == "__main__":
    main()
