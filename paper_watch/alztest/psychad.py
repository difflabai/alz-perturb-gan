"""Benchmark Alzheimer and Lewy-body specificity on held-out microglia donors."""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from .data import load_psychad
from .harness import ConditionalPGAN, energy_distance, mean_shift_prediction, shift_cosine

GROUPS = ("control", "Alzheimer", "Lewy")
MIN_TRAIN_DONORS = 20
STEPS = 300
DRAWS = 256


def donor_profiles(counts, donors, diagnoses):
    """Return one log-normalized count profile per donor."""
    donors = np.asarray(donors, dtype=str)
    diagnoses = np.asarray(diagnoses, dtype=str)
    counts = sparse.csr_matrix(counts)
    if set(diagnoses) != set(GROUPS) or np.isin(donors, ("", "unknown", "nan")).any():
        raise ValueError("expected donor IDs and control, Alzheimer, and Lewy diagnosis labels")
    unique, first, inverse = np.unique(donors, return_index=True, return_inverse=True)
    groups = diagnoses[first]
    if np.any(diagnoses != groups[inverse]):
        raise ValueError("each donor must have a single diagnosis")
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("expected finite, nonnegative counts")

    membership = sparse.csr_matrix(
        (np.ones(len(donors)), (inverse, np.arange(len(donors)))),
        shape=(len(unique), len(donors)),
    )
    profiles = membership @ counts
    library = np.asarray(profiles.sum(axis=1)).ravel()
    if (library <= 0).any():
        raise ValueError("at least one donor has an empty expression profile")
    profiles = profiles.multiply((1e4 / library)[:, None]).tocsr()
    np.log1p(profiles.data, out=profiles.data)
    return profiles, groups


def split_donors(groups, seed):
    rng = np.random.default_rng(seed)
    train, held = [], []
    for group in GROUPS:
        rows = rng.permutation(np.flatnonzero(groups == group))
        cutoff = len(rows) * 3 // 4
        if cutoff < MIN_TRAIN_DONORS or len(rows) - cutoff < 2:
            raise ValueError(f"insufficient donors for a disjoint {group} evaluation")
        train.extend(rows[:cutoff])
        held.extend(rows[cutoff:])
    return np.asarray(train), np.asarray(held)


def project_profiles(profiles, train, *, max_genes=512, max_components=12):
    reference = profiles[train]
    mean = np.asarray(reference.mean(axis=0)).ravel()
    second = np.asarray(reference.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - mean ** 2, 0)
    expressed = np.flatnonzero(variance > 1e-10)
    if not len(expressed):
        raise ValueError("no varying genes in training donors")
    order = np.argsort(-variance[expressed], kind="stable")
    selected = expressed[order[:max_genes]]
    values = profiles[:, selected].toarray()
    standardized = (values - mean[selected]) / np.sqrt(variance[selected])
    _, singular, components = np.linalg.svd(standardized[train], full_matrices=False)
    n_components = min(max_components, np.count_nonzero(singular > 1e-8))
    return standardized @ components[:n_components].T


def score(predicted, real):
    def cosine(pred_group, real_group):
        return shift_cosine(
            predicted[pred_group], predicted["control"], real[real_group], real["control"]
        )

    specificity = (
        cosine("Alzheimer", "Alzheimer") - cosine("Alzheimer", "Lewy")
        + cosine("Lewy", "Lewy") - cosine("Lewy", "Alzheimer")
    )
    # Pinned to the V-statistic so the recorded results JSON reproduces.
    distances = {
        group: energy_distance(predicted[group], real[group], unbiased=False)
        for group in GROUPS
    }
    return {
        "specificity": specificity,
        "mean_energy_distance": float(np.mean(list(distances.values()))),
        "energy_distance_by_diagnosis": distances,
    }


def run_once(profiles, groups, seed):
    train, held = split_donors(groups, seed)
    coordinates = project_profiles(profiles, train)
    train_group = {group: coordinates[train][groups[train] == group] for group in GROUPS}
    real = {group: coordinates[held][groups[held] == group] for group in GROUPS}
    labels = np.array([GROUPS.index(group) for group in groups[train]])
    rng = np.random.default_rng(seed + 31)

    def draw(data):
        return data[rng.integers(len(data), size=DRAWS)]

    ctrl = train_group["control"]
    arms = {
        "mean_shift": {
            group: draw(mean_shift_prediction(ctrl, train_group[group])) for group in GROUPS
        },
        "empirical": {group: draw(train_group[group]) for group in GROUPS},
    }
    for arm, condition in (("empf", labels), ("shuffled", rng.permutation(labels))):
        model = ConditionalPGAN(coordinates.shape[1], len(GROUPS), seed=seed)
        model.fit(
            coordinates[train], condition, steps=STEPS, min_condition_rows=MIN_TRAIN_DONORS
        )
        arms[arm] = {
            group: model.sample(k, DRAWS, seed=seed + 100 + k) for k, group in enumerate(GROUPS)
        }

    scores = {arm: score(predicted, real) for arm, predicted in arms.items()}
    model = scores["empf"]
    passes = model["specificity"] > 0.2 and all(
        model["specificity"] > scores[arm]["specificity"]
        and model["mean_energy_distance"] < scores[arm]["mean_energy_distance"]
        for arm in ("mean_shift", "empirical", "shuffled")
    )
    return {
        "seed": seed,
        "train_donors": {group: len(train_group[group]) for group in GROUPS},
        "test_donors": {group: len(real[group]) for group in GROUPS},
        "scores": scores,
        "passes_screen": passes,
    }


def main():
    data = load_psychad()
    profiles, groups = donor_profiles(data.X, data.obs["donor_id"], data.obs["diagnosis"])
    runs = [run_once(profiles, groups, seed) for seed in (0, 1, 2)]
    print(json.dumps({
        "input_shape": list(data.shape),
        "regularizer_schedule": "released EMPF default (every step)",
        "updates_per_fit": STEPS,
        "runs": runs,
        "passes_screen": all(run["passes_screen"] for run in runs),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
