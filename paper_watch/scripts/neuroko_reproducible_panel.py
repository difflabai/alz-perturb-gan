"""Stage A of the NeuroKO sharpening spec: choose knockouts that repeat across replicates.

CPU only, no ParticleGAN fit. For each outer fold (one development group held out,
the other two used for training) the script:

1. lists knockouts with enough cells in both training groups,
2. measures each knockout's target-minus-INT pseudobulk shift separately in the
   two training groups and scores reproducibility as their cosine,
3. compares that cosine with the same target's cosines against every other
   target (wrong-target null), so a shared "any knockout" response does not count,
4. selects the panel by the pre-declared rule (training groups only),
5. reports, after selection, whether the selected shifts also appear in the
   held-out group, and
6. scores the simple baselines on the selected panel with both energy-distance
   estimators and exports the projected arrays so later ParticleGAN arms use
   identical inputs.

The reserved holdout group (Screen1_2 by default) is never read: its rows are
dropped from the annotation table before any expression row is loaded.

    python paper_watch/scripts/neuroko_reproducible_panel.py \
        --data paper_watch/data/neuroko --out runs/neuroko-sharpen/stageA
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_watch.alztest.neuroko import CONTROL, normalize_counts, open_screen  # noqa: E402

DEV_GROUPS = ("Screen1_1", "Screen1_3", "Screen1_4")
RESERVED = ("Screen1_2",)
RULE = {
    "min_cells_per_training_group": 100,
    "cap_cells_per_group": 256,
    "n_features": 512,
    "n_components": 30,
    "min_reproducibility_cosine": 0.20,
    "specificity": "reproducibility cosine above the 95th percentile of the target's wrong-target cosines",
    "require_target_rna_below_int_in_both_training_groups": True,
    "panel_size": 8,
    "min_panel_size_to_continue": 4,
    "baseline_seeds": [900, 901, 902, 903, 904, 905],
    "draws_per_condition": 256,
}


def _cosine(a, b):
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / norm) if norm > 0 else float("nan")


def _distances(x, y):
    sq = (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T
    return np.sqrt(np.clip(sq, 0, None))


def energy_pair(a, b):
    """Return (V-statistic, U-statistic) energy distance on the full samples."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    ab = _distances(a, b).mean()
    aa = _distances(a, a).sum()
    bb = _distances(b, b).sum()
    v = 2 * ab - aa / n**2 - bb / m**2
    u = 2 * ab - aa / (n * (n - 1)) - bb / (m * (m - 1))
    return float(v), float(u)


def sample_rows(obs, groups, targets, cap, rng):
    """Pick up to `cap` file rows per (group, target).

    Returns original file row numbers (the labels of the annotation index), not
    positions in the filtered table, so dropped reserved rows can never be read.
    """
    picked = {}
    labels = obs.index.to_numpy()
    index = {k: labels[v] for k, v in obs.groupby(["group", "target"], sort=True).indices.items()}
    for group in groups:
        for target in targets:
            rows = index.get((group, target))
            if rows is None or len(rows) == 0:
                continue
            take = min(cap, len(rows))
            picked[(group, target)] = np.sort(rng.choice(rows, size=take, replace=False))
    return picked


def load_rows(screen, picked):
    """Load and normalize all picked rows once; return a dict of dense-able CSR blocks."""
    keys = sorted(picked)
    positions = np.concatenate([picked[k] for k in keys])
    order = np.argsort(positions)
    rows = positions[order]
    matrix = normalize_counts(screen.adata.X[rows, :])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    blocks, start = {}, 0
    for key in keys:
        size = len(picked[key])
        blocks[key] = matrix[inverse[start:start + size]]
        start += size
    return blocks


def fold_projection(train_blocks, n_features, n_components):
    from scipy.sparse import vstack

    stacked = vstack(train_blocks)
    mean = np.asarray(stacked.mean(axis=0)).ravel().astype(np.float64)
    variance = np.asarray(stacked.power(2).mean(axis=0)).ravel() - mean**2
    features = np.argsort(-variance, kind="stable")[:n_features]
    dense = stacked[:, features].toarray().astype(np.float64)
    center = dense.mean(0)
    _, _, vt = np.linalg.svd(dense - center, full_matrices=False)
    axes = vt[:n_components]
    scale = ((dense - center) @ axes.T).std(0)
    keep = scale > 1e-8
    return features, center, axes[keep], scale[keep]


def project(block, features, center, axes, scale):
    return ((block[:, features].toarray().astype(np.float64) - center) @ axes.T) / scale


def shift(blocks, group, target, features):
    t = np.asarray(blocks[(group, target)][:, features].mean(axis=0)).ravel()
    c = np.asarray(blocks[(group, CONTROL)][:, features].mean(axis=0)).ravel()
    return t - c


def target_rna(blocks, group, target, gene_index):
    if target not in gene_index:
        return None
    j = gene_index[target]
    t = float(np.asarray(blocks[(group, target)][:, j].mean()))
    c = float(np.asarray(blocks[(group, CONTROL)][:, j].mean()))
    return t - c


def gaussian_sampler(x, rng, n):
    mean = x.mean(0)
    cov = np.cov(x, rowvar=False) + 1e-6 * np.eye(x.shape[1])
    return rng.multivariate_normal(mean, cov, size=n)


def score_baselines(train, test, panel, seeds, draws, matches):
    """train/test: dict condition -> projected rows. Returns per-arm, per-seed scores."""
    int_mean = train[CONTROL].mean(0)
    arms = ("empirical", "mean_shift", "gaussian", "int_only", "wrong_target")
    out = {arm: [] for arm in arms}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        per_arm = {arm: {} for arm in arms}
        for target in panel:
            observed = test[target]
            tr = train[target]
            generated = {
                "empirical": tr[rng.integers(0, len(tr), draws)],
                "mean_shift": train[CONTROL][rng.integers(0, len(train[CONTROL]), draws)]
                + (tr.mean(0) - int_mean),
                "gaussian": gaussian_sampler(tr, rng, draws),
                "int_only": train[CONTROL][rng.integers(0, len(train[CONTROL]), draws)],
            }
            wrong = train[matches[target]]
            generated["wrong_target"] = wrong[rng.integers(0, len(wrong), draws)]
            obs_shift = observed.mean(0) - test[CONTROL].mean(0)
            for arm, gen in generated.items():
                v, u = energy_pair(gen, observed)
                per_arm[arm][target] = {
                    "energy_v": v,
                    "energy_u": u,
                    "shift_cosine": _cosine(gen.mean(0) - int_mean, obs_shift),
                }
        for arm in arms:
            rows = per_arm[arm]
            out[arm].append({
                "seed": seed,
                "mean_energy_v": float(np.mean([r["energy_v"] for r in rows.values()])),
                "mean_energy_u": float(np.mean([r["energy_u"] for r in rows.values()])),
                "mean_shift_cosine": float(np.mean([r["shift_cosine"] for r in rows.values()])),
                "per_target": rows,
            })
    summary = {}
    for arm, rows in out.items():
        summary[arm] = {
            "mean_energy_v": float(np.mean([r["mean_energy_v"] for r in rows])),
            "mean_energy_u": float(np.mean([r["mean_energy_u"] for r in rows])),
            "mean_shift_cosine": float(np.mean([r["mean_shift_cosine"] for r in rows])),
        }
    for arm in summary:
        summary[arm]["ko_gain_u"] = summary["int_only"]["mean_energy_u"] - summary[arm]["mean_energy_u"]
    return summary, out


def run_fold(screen, blocks, heldout, training, candidates, gene_index, rule, out_dir):
    fold = {"heldout": heldout, "training": list(training)}
    t1, t2 = training
    features, center, axes, scale = fold_projection(
        [blocks[k] for k in blocks if k[0] in training], rule["n_features"], rule["n_components"])
    shifts = {g: {t: shift(blocks, g, t, features) for t in candidates} for g in training}
    rows = []
    for t in candidates:
        r = _cosine(shifts[t1][t], shifts[t2][t])
        wrong = [_cosine(shifts[t1][t], shifts[t2][o]) for o in candidates if o != t]
        wrong += [_cosine(shifts[t1][o], shifts[t2][t]) for o in candidates if o != t]
        wrong = np.asarray([w for w in wrong if np.isfinite(w)])
        rna = [target_rna(blocks, g, t, gene_index) for g in training]
        rows.append({
            "target": t,
            "reproducibility_cosine": r,
            "wrong_target_median": float(np.median(wrong)) if len(wrong) else None,
            "wrong_target_p95": float(np.quantile(wrong, 0.95)) if len(wrong) else None,
            "target_rna_minus_int": rna,
            "cells": {g: int(blocks[(g, t)].shape[0]) for g in training},
        })
    for row in rows:
        rna_ok = all(x is not None and x < 0 for x in row["target_rna_minus_int"])
        row["passes"] = bool(
            np.isfinite(row["reproducibility_cosine"])
            and row["reproducibility_cosine"] >= rule["min_reproducibility_cosine"]
            and row["wrong_target_p95"] is not None
            and row["reproducibility_cosine"] > row["wrong_target_p95"]
            and (rna_ok or not rule["require_target_rna_below_int_in_both_training_groups"])
        )
    passing = sorted((r for r in rows if r["passes"]), key=lambda r: -r["reproducibility_cosine"])
    panel = [r["target"] for r in passing[: rule["panel_size"]]]
    fold["candidates"] = sorted(rows, key=lambda r: -r["reproducibility_cosine"])
    fold["panel"] = panel
    fold["continue"] = len(panel) >= rule["min_panel_size_to_continue"]

    # After selection only: does each selected shift appear in the held-out group?
    if (heldout, CONTROL) in blocks:
        check = {}
        for t in panel:
            if (heldout, t) not in blocks:
                continue
            h = shift(blocks, heldout, t, features)
            train_shift = (shifts[t1][t] + shifts[t2][t]) / 2
            others = [_cosine(train_shift, shift(blocks, heldout, o, features))
                      for o in candidates if o != t and (heldout, o) in blocks]
            check[t] = {"heldout_cosine": _cosine(train_shift, h),
                        "wrong_target_median": float(np.median(others)) if others else None}
        fold["after_selection_heldout_check"] = check

    if not fold["continue"]:
        return fold

    # Matched wrong target: nearest training INT expression of the target gene, within candidates.
    int_expr = {}
    for t in candidates:
        if t in gene_index:
            j = gene_index[t]
            int_expr[t] = float(np.mean([np.asarray(blocks[(g, CONTROL)][:, j].mean()) for g in training]))
    matches = {}
    for t in panel:
        pool = [o for o in candidates if o != t and o in int_expr and o not in panel]
        if t in int_expr and pool:
            matches[t] = min(pool, key=lambda o: abs(int_expr[o] - int_expr[t]))
        else:
            matches[t] = next(o for o in candidates if o != t and o not in panel)
    fold["wrong_target_matches"] = matches

    conditions = [CONTROL, *panel, *sorted(set(matches.values()))]
    train = {c: np.vstack([project(blocks[(g, c)], features, center, axes, scale) for g in training])
             for c in conditions}
    test = {c: project(blocks[(heldout, c)], features, center, axes, scale)
            for c in [CONTROL, *panel] if (heldout, c) in blocks}
    summary, seeds = score_baselines(train, test, panel, rule["baseline_seeds"],
                                     rule["draws_per_condition"], matches)
    fold["baselines"] = summary
    fold["baseline_seeds"] = seeds

    arrays = {}
    for c, x in train.items():
        arrays[f"train__{c}"] = x.astype(np.float32)
    for c, x in test.items():
        arrays[f"test__{c}"] = x.astype(np.float32)
    arrays["projection__features"] = features
    arrays["projection__center"] = center
    arrays["projection__axes"] = axes
    arrays["projection__scale"] = scale
    path = out_dir / f"fold_{heldout}.npz"
    np.savez_compressed(path, **arrays)
    fold["arrays"] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return fold


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-cells", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rule = dict(RULE)
    with ExitStack() as stack:
        screen = stack.enter_context(open_screen(args.data, "iNeurons", args.window_cells))
        obs = screen.obs[~screen.obs["group"].isin(RESERVED)]
        groups = [g for g in DEV_GROUPS if g in set(obs["group"])]
        if len(groups) != 3:
            raise ValueError(f"expected development groups {DEV_GROUPS}, found {sorted(set(obs['group']))}")
        counts = obs.groupby(["target", "group"]).size().unstack(fill_value=0)
        if CONTROL not in counts.index:
            raise ValueError("intergenic controls missing")
        # Candidates must have enough cells in at least two development groups; each fold
        # then filters on its own two training groups.
        loose = counts.drop(CONTROL)
        loose = loose[(loose[groups] >= rule["min_cells_per_training_group"]).sum(axis=1) >= 2].index.tolist()
        rng = np.random.default_rng(args.seed)
        picked = sample_rows(obs, groups, [CONTROL, *loose], rule["cap_cells_per_group"], rng)
        blocks = load_rows(screen, picked)
        gene_index = {g: i for i, g in enumerate(screen.adata.var_names.astype(str))}
        result = {"rule": rule, "window_cells": args.window_cells, "seed": args.seed,
                  "reserved_groups_dropped": list(RESERVED), "development_groups": groups,
                  "folds": []}
        for heldout in groups:
            training = tuple(g for g in groups if g != heldout)
            candidates = [t for t in loose
                          if all(counts.at[t, g] >= rule["min_cells_per_training_group"] for g in training)]
            fold = run_fold(screen, blocks, heldout, training, candidates, gene_index, rule, args.out)
            result["folds"].append(fold)
            print(f"{heldout}: {len(candidates)} candidates, panel {fold['panel']}", flush=True)
    result["stop"] = not all(f["continue"] for f in result["folds"])
    (args.out / "stageA_panels.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({f["heldout"]: f.get("baselines") for f in result["folds"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
