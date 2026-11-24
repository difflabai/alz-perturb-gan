"""Estimate a fixed prior weight from paired particle-gradient measurements at two critic rates."""
import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import median

from compare_neuroko_v12 import GROUPS, ROOT, write_frozen
from neuroko_rate_noise import NOISY_PRIOR, REPLICATION_SEEDS


def load(path):
    return json.loads(path.read_text())


def per_fit(row, reference):
    if {k: v for k, v in row.items() if k != "particle_loss_gradients"} != reference:
        raise ValueError("Measuring particle gradients changed the critic trace, training inputs or final weights")
    trace = row["particle_loss_gradients"]
    if len(trace) != 64:
        raise ValueError("Expected 64 pairs of loss gradients")
    ratios = []
    for step in trace:
        adv, prior = step["adversarial"], step["weighted_prior"]
        if not (isfinite(adv) and isfinite(prior) and adv > 0 and prior >= 0 and step["ratio"] == prior / adv):
            raise ValueError("Expected finite gradient norms and a matching ratio on every update")
        ratios.append(step["ratio"])
    return dict(median_adversarial_norm=median(r["adversarial"] for r in trace),
                median_weighted_prior_norm=median(r["weighted_prior"] for r in trace),
                median_ratio=median(ratios), ratios_by_update=ratios)


def compare(original, calibrated, reference_original, reference_calibrated):
    studies = (original, calibrated)
    for study, rate in zip(studies, (.021, .0046), strict=True):
        spec = study["protocol"]
        if (not spec.get("particle_gradients") or spec.get("v12_critic_lr", .021) != rate
                or spec.get("v12_prior_weight", .1) != .1 or spec["settings"] != NOISY_PRIOR or spec["steps"] != 64):
            raise ValueError("Unexpected critic rate, prior weight or training protocol")
    expected = [(seed, group) for seed in REPLICATION_SEEDS for group in GROUPS]
    cells = []
    for (seed, group), a, b, old_a, old_b in zip(expected, original["cells"], calibrated["cells"],
                                                  reference_original["cells"], reference_calibrated["cells"], strict=True):
        if any((r["seed"], r["group"]) != (seed, group) for r in (a, b, old_a, old_b)):
            raise ValueError("Inconsistent seed or group in particle-gradient measurements")
        for field in ("projected_data_sha256", "probe_indices_sha256", "gpu"):
            if any(r[field] != a[field] for r in (b, old_a, old_b)):
                raise ValueError(f"Unmatched {field} between gradient measurements and critic references")
        if a["rows"]["unified_v11"] != b["rows"]["unified_v11"]:
            raise ValueError("The fixed v11 control changed between the two measurements")
        u, v = (r["rows"]["unified_v12"] for r in (a, b))
        if any(u[k] != v[k] for k in ("initial_parameters_sha256", "training_batches_sha256")):
            raise ValueError("The v12 inputs differ at the two critic rates")
        ref = per_fit(a["rows"]["unified_v11"], old_a["rows"]["unified_v11"])
        base = per_fit(u, old_a["rows"]["unified_v12"])
        lower = per_fit(v, old_b["rows"]["unified_v12"])
        cells.append(dict(seed=seed, group=group, input_hash=a["projected_data_sha256"],
                          initial_parameters_sha256=u["initial_parameters_sha256"], batches_sha256=u["training_batches_sha256"],
                          v11_reference=ref, v12_original=base, v12_calibrated=lower,
                          relative_strength=lower["median_ratio"] / base["median_ratio"]))
    scale = median(c["relative_strength"] for c in cells)
    weight = round(NOISY_PRIOR["prior_reg"] / scale, 3)
    return dict(cells=cells, median_relative_strength=scale, pairs_increased=sum(c["relative_strength"] > 1 for c in cells),
                original_critic_rate=.021, calibrated_critic_rate=.0046, original_prior_weight=NOISY_PRIOR["prior_reg"],
                suggested_prior_weight=weight,
                rule="Use the per-fit median norm of the weighted prior gradient divided by the adversarial gradient "
                     "norm on the particle table. Divide weight 0.1 by the median paired increase in that ratio "
                     "when the fixed critic rate is reduced to 0.0046. Round to 0.001.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "calibrated", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    notes = ROOT / "paper_watch/notes"
    result = compare(load(args.original), load(args.calibrated), load(notes / "neuroko_v12_critic_motion.json"),
                     load(notes / "neuroko_v12_critic_motion_interpolated.json"))
    write_frozen(args.output, result)
    print("Relative particle prior gradient strength", result["median_relative_strength"],
          "estimated fixed prior weight", result["suggested_prior_weight"], flush=True)


if __name__ == "__main__":
    main()
