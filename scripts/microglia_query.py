#!/usr/bin/env python3
"""Query the native EMPF microglia model for genes and a pair of pathology stages.

Reads precomputed query bundles (one per trained fit) built by
analysis/query_tool_20261120/build_bundle.py; needs only NumPy, no torch,
weights or raw cells. For each gene it returns generated stage means, the
stage contrast (stage B minus stage A) with generation Monte Carlo and, when
several bundles are given, between-fit uncertainty; calibrated donor-level
intervals; an observed reference with a donor bootstrap; the stage-blind
empirical reference; and flags for genes outside the panel or poorly fit.

Example:
  python3 -B scripts/microglia_query.py --genes CLEC5A PPARG APOE --stages "Not AD" High

The output JSON schema ("microglia-query/1") is documented in
analysis/query_tool_20261120/README.md. Outputs describe an observational,
stage-conditioned generator; they are not treatment-response predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = Path('/ml2/alz/runs/research_20261120/query-tool/bundle_fold0_seed0.npz')
SCHEMA = 'microglia-query/1'
BUNDLE_SCHEMA = 'microglia-query-bundle/1'
STAGES = ['Not AD', 'Low', 'Intermediate', 'High']
ALIASES = {'notad': 0, 'not_ad': 0, 'not ad': 0, 'control': 0, '0': 0, 'low': 1, '1': 1,
           'intermediate': 2, 'int': 2, '2': 2, 'high': 3, '3': 3}
MIN_REAL_CELLS, MIN_REAL_DONORS = 20, 3
SUBTYPE_COLUMNS = [2, 4]


def stage_index(text):
    key = str(text).strip().lower()
    if key not in ALIASES:
        raise ValueError(f'unknown stage {text!r}; use one of {STAGES}')
    return ALIASES[key]


def num(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def pair(lo, hi):
    return [num(lo), num(hi)]


class Bundle:
    def __init__(self, path):
        path = Path(path)
        self.path = path
        self.meta = json.loads(path.with_suffix('.json').read_text())
        if self.meta.get('schema') != BUNDLE_SCHEMA:
            raise ValueError(f'{path}: not a {BUNDLE_SCHEMA} bundle')
        with np.load(path, allow_pickle=False) as z:
            self.z = {k: z[k] for k in z.files}
        self.genes = [str(g) for g in self.z['genes']]
        self.index = {g: i for i, g in enumerate(self.genes)}

    def describe(self):
        m = self.meta
        return dict(bundle=str(self.path), fit=m['fit'], readout=m['readout'], fold=m.get('fold'),
                    final_weights_sha256=m['weights_sha256'].get('final.pt'), replicates=m['replicates'],
                    generation_devices=m.get('generation_devices'), panel_genes=len(self.genes),
                    bundle_sha256=m.get('bundle_npz_sha256'))


def observed_contrast(sums, n, stage, a, b, columns, rng, n_boot):
    """Equal donor weight within column/stage (>=20 cells for subtypes), equal column weight; donor bootstrap."""
    def estimate(idx):
        eff = []
        for c in columns:
            minimum = 1 if c == 0 else MIN_REAL_CELLS
            keep = n[idx, c] >= minimum
            st = stage[idx]
            if (keep & (st == a)).sum() < (1 if c == 0 else MIN_REAL_DONORS) or \
               (keep & (st == b)).sum() < (1 if c == 0 else MIN_REAL_DONORS):
                return None
            means = sums[idx, c] / np.maximum(n[idx, c], 1)
            eff.append((means[keep & (st == a)].mean(), means[keep & (st == b)].mean()))
        return np.mean(eff, axis=0)
    all_idx = np.arange(len(stage))
    point = estimate(all_idx)
    if point is None:
        return None
    ia, ib = np.flatnonzero(stage == a), np.flatnonzero(stage == b)
    draws = []
    for _ in range(n_boot):
        e = estimate(np.concatenate([rng.choice(ia, len(ia)), rng.choice(ib, len(ib))]))
        if e is not None:
            draws.append(e[1] - e[0])
    draws = np.asarray(draws)
    return dict(stage_a_mean=num(point[0]), stage_b_mean=num(point[1]), contrast=num(point[1] - point[0]),
                ci95_donor_bootstrap=(pair(*np.quantile(draws, [.025, .975])) if len(draws) else None),
                bootstrap_valid_draws=int(len(draws)))


def summarize_contrast(per_fit_reps):
    """per_fit_reps: list over fits of replicate contrasts (1-D arrays)."""
    fit_means, mc_sd, mc_se, reps_all = [], [], [], []
    for v in per_fit_reps:
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        if not len(v):
            continue
        fit_means.append(v.mean()); reps_all.append(v)
        mc_sd.append(v.std(ddof=1) if len(v) > 1 else np.nan)
        mc_se.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)
    if not fit_means:
        return None
    fit_means = np.asarray(fit_means)
    est = fit_means.mean()
    between = fit_means.std(ddof=1) if len(fit_means) > 1 else None
    mc_part = np.sqrt(np.nanmean(np.square(mc_se)) / len(fit_means)) if np.isfinite(mc_se).any() else None
    total = np.sqrt((between ** 2 / len(fit_means) if between is not None else 0) + (mc_part or 0) ** 2)
    allv = np.concatenate(reps_all)
    return dict(estimate=num(est), per_fit=[num(x) for x in fit_means],
                mc_replicate_sd=[num(x) for x in mc_sd], mc_standard_error=num(mc_part),
                between_fit_sd=num(between), standard_error_total=num(total),
                approx_ci95=pair(est - 1.96 * total, est + 1.96 * total),
                replicate_range=pair(allv.min(), allv.max()),
                fraction_replicates_positive=num((allv > 0).mean()), replicates=int(len(allv)),
                uncertainty_scope=('generation Monte Carlo and between-fit' if between is not None else
                                   'generation Monte Carlo only; fit-to-fit variability unmeasured (one fit)'))


def query(bundles, genes, a, b, n_boot=2000, seed=20261120):
    if a == b:
        raise ValueError('choose two different stages')
    rows = []
    for gene in genes:
        present = [B for B in bundles if gene in B.index]
        row = dict(gene=gene, in_panel=bool(present), in_panel_by_fit=[gene in B.index for B in bundles])
        if not present:
            row['flags'] = dict(not_in_panel=True)
            row['note'] = 'gene is not among the modelled panel genes; absence is not evidence of no change'
            rows.append(row)
            continue
        stage_reps = {a: [], b: []}; contrast_reps = []; within_reps = []; within_stage = {a: [], b: []}
        diag = []
        for B in present:
            j, z = B.index[gene], B.z
            sm = z['stage_mean_rep'][:, :, j]
            stage_reps[a].append(sm[:, a]); stage_reps[b].append(sm[:, b])
            contrast_reps.append(sm[:, b] - sm[:, a])
            w = z['within_rep'][:, :, j]
            within_reps.append(w[:, b] - w[:, a]); within_stage[a].append(w[:, a]); within_stage[b].append(w[:, b])
            lo, hi, corr = z['interval_lo'][:, :, j], z['interval_hi'][:, :, j], z['correction'][:, j]
            intervals = {}
            for s in (a, b):
                intervals[STAGES[s]] = {
                    c: dict(raw=pair(lo[ci, s], hi[ci, s]),
                            calibrated=pair(max(lo[ci, s] - corr[ci], 0), hi[ci, s] + corr[ci]))
                    for ci, c in enumerate(('mean', 'sd'))}
            rng = np.random.default_rng(seed)
            obs_pooled = observed_contrast(z['obs_subtype_sums'][:, :, j].astype(float), z['obs_subtype_n'],
                                           z['obs_stage'], a, b, [0], rng, n_boot)
            obs_within = observed_contrast(z['obs_subtype_sums'][:, :, j].astype(float), z['obs_subtype_n'],
                                           z['obs_stage'], a, b, SUBTYPE_COLUMNS, rng, n_boot)
            diag.append(dict(
                fit=B.meta['fit'],
                flags=dict(poorly_fit=bool(z['flag_poorly_fit'][j]), pc_blind_spot=bool(z['flag_pc_blind_spot'][j]),
                           low_expression=bool(z['flag_low_expression'][j])),
                heldout_mean_log_rmse=num(z['heldout_rmse'][j]),
                stage_blind_empirical_rmse=num(z['empirical_rmse'][j]), poorly_fit_rmse_cut=num(z['rmse_cut']),
                pc_outside_share_by_stage={STAGES[s]: num(z['pc_outside_share'][s, j]) for s in range(4)},
                pc_standardized_gap_by_stage={STAGES[s]: num(z['pc_gap'][s, j]) for s in range(4)},
                pc_captured_fraction=num(z['pc_captured'][j]), observed_detection=num(z['detection'][j]),
                crossfit_heldout_coverage={c: dict(raw=num(z['crossfit_coverage_raw'][ci, j]),
                                                   calibrated=num(z['crossfit_coverage_cal'][ci, j]))
                                           for ci, c in enumerate(('mean', 'sd'))},
                donor_intervals_80=intervals,
                observed_reference=dict(
                    donors=dict(zip((STAGES[a], STAGES[b]), (int((z['obs_stage'] == a).sum()), int((z['obs_stage'] == b).sum())))),
                    pooled=obs_pooled, within_subtype=obs_within,
                    note='SEA-AD development donors of this fold cache (GAN TRAIN and held-out); data, not model output'),
                stage_blind_empirical_reference=dict(
                    stage_mean=num(z['empirical_stage_blind_mean'][j]), contrast=0.0,
                    donor_interval_80=dict(mean=pair(z['empirical_interval_lo'][0, j], z['empirical_interval_hi'][0, j]),
                                           sd=pair(z['empirical_interval_lo'][1, j], z['empirical_interval_hi'][1, j]))),
            ))
        flags = dict(not_in_panel=False,
                     not_in_panel_of_some_fits=len(present) < len(bundles),
                     poorly_fit=any(d['flags']['poorly_fit'] for d in diag),
                     pc_blind_spot=any(d['flags']['pc_blind_spot'] for d in diag),
                     low_expression=any(d['flags']['low_expression'] for d in diag),
                     single_fit=len(present) == 1)
        row.update(
            flags=flags,
            generated=dict(
                stage_means={STAGES[s]: summarize_contrast(stage_reps[s]) for s in (a, b)},
                contrast=summarize_contrast(contrast_reps),
                within_subtype=dict(stage_means={STAGES[s]: summarize_contrast(within_stage[s]) for s in (a, b)},
                                    contrast=summarize_contrast(within_reps))),
            per_fit=diag)
        rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--genes', nargs='+', required=True)
    p.add_argument('--stages', nargs=2, required=True, metavar=('STAGE_A', 'STAGE_B'),
                   help='contrast is STAGE_B minus STAGE_A; names or 0-3')
    p.add_argument('--bundle', type=Path, nargs='+', default=[DEFAULT_BUNDLE],
                   help='one bundle per trained fit; several form a fits ensemble')
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--out', type=Path, help='write JSON here instead of stdout')
    args = p.parse_args(argv)
    try:
        a, b = (stage_index(s) for s in args.stages)
        bundles = [Bundle(path) for path in args.bundle]
        rows = query(bundles, list(dict.fromkeys(args.genes)), a, b, n_boot=args.n_boot)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(f'microglia_query: {exc}')
    result = dict(
        schema=SCHEMA, query=dict(genes=list(dict.fromkeys(args.genes)), stage_a=STAGES[a], stage_b=STAGES[b],
                                  contrast='stage_b minus stage_a'),
        model=dict(ensemble_size=len(bundles), fits=[B.describe() for B in bundles],
                   units=bundles[0].meta['units'], stage_mean_definition=bundles[0].meta['stage_mean_definition'],
                   within_subtype_definition=bundles[0].meta['within_subtype_definition'],
                   interval_definition=bundles[0].meta['interval_definition'], calibration=bundles[0].meta['calibration'],
                   flag_definitions=bundles[0].meta['flags'],
                   status='owner-selected experimental EMPF default; not population qualified'),
        scope=['Stage-conditioned observational generator; no gene, drug or dose inputs.',
               'Calibrated lower bounds are clipped at 0 (both coordinates are nonnegative); see '
               'model.calibration.heldout_crossfit_evaluation before preferring calibrated over raw intervals.',
               'Intervals are for one donor bag (K=256 cells) at the queried stage, nominal 80%.',
               'Uncertainty excludes biological donor variability of the effect unless stated (observed reference).'],
        genes=rows)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text + '\n')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
