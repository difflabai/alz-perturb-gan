"""Evaluate gene-level donor intervals (PROTOCOL part B), compute gene flags (part D), write a query bundle.

Inputs: the Monte Carlo replicate archives from generate_cells.py and the frozen fold0 cache.
Outputs: ``--eval-out`` (JSON scores) and ``--bundle`` (.npz plus .json sidecar) for
scripts/microglia_query.py. CPU and NumPy only; no weights are loaded.
"""
from __future__ import annotations

import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3')
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

BUNDLE_SCHEMA = 'microglia-query-bundle/1'
RMSE_QUANTILE, RMSE_EMPIRICAL_RATIO = .9, 1.5
BLIND_SHARE, BLIND_MIN_GAP = .5, .05
LOW_DETECTION = .01
N_BOOT = 2000


def load_reps(paths):
    reps = []
    for p in paths:
        receipt = json.loads(Path(p).with_suffix('.receipt.json').read_text())
        if receipt['output_sha256'] != C.sha(p) or receipt['targets'] != 37:
            raise ValueError(f'replicate receipt mismatch or validation-only archive: {p}')
        with np.load(p, allow_pickle=False) as z:
            reps.append({k: z[k] for k in z.files if k != 'pcs'} | {'receipt': receipt})
    for r in reps[1:]:
        if not (np.array_equal(r['targets'], reps[0]['targets']) and np.array_equal(r['genes'], reps[0]['genes'])):
            raise ValueError('replicates differ in targets or genes')
    return reps


def score_block(lo, hi, y):
    """lo/hi [D,G]; y [R,D,G]."""
    return dict(coverage=float(((y >= lo) & (y <= hi)).mean()), below=float((y < lo).mean()),
                above=float((y > hi).mean()), width=float((hi - lo).mean()),
                interval_score=float(C.interval_score(lo, hi, y).mean()))


def per_donor_is(lo, hi, y):
    return C.interval_score(lo, hi, y).mean(axis=(0, 2))  # [D]


def evaluate(name, raw, obs_mean, obs_sd, donors, stage_of_donor, train_donors, halves):
    """raw: {'mean': (lo[4,G], hi[4,G]), 'sd': ...}; observed [R,D,G] for the 12 held-out donors."""
    idx = {d: i for i, d in enumerate(donors)}
    y = {'mean': obs_mean, 'sd': obs_sd}
    lo_raw = {c: raw[c][0][stage_of_donor] for c in y}
    hi_raw = {c: raw[c][1][stage_of_donor] for c in y}
    lo_cal = {c: lo_raw[c].copy() for c in y}; hi_cal = {c: hi_raw[c].copy() for c in y}
    swaps = []
    for cal, test in halves:
        ci, ti = [idx[d] for d in cal], [idx[d] for d in test]
        rec = {}
        for c in y:
            fit = C.fit_correction(lo_raw[c][ci], hi_raw[c][ci], y[c][0, ci], cal, train_donors)
            changed = y[c].copy(); changed[:, ti] = 1e6
            again = C.fit_correction(lo_raw[c][ci], hi_raw[c][ci], changed[0, ci], cal, train_donors)
            if not np.array_equal(fit['correction'], again['correction']):
                raise AssertionError('test donor changed its own correction')
            lo_cal[c][ti], hi_cal[c][ti] = C.apply_correction(lo_raw[c][ti], hi_raw[c][ti], fit['correction'], c == 'sd')
            rec[c] = dict(rank=fit['rank'], n=fit['n_calibration_donors'],
                          median_correction=float(np.median(fit['correction'])))
        swaps.append(dict(calibration_donors=list(cal), evaluation_donors=list(test), fits=rec))
    out = {'swaps': swaps}
    variants = {'raw': (lo_raw, hi_raw),
                'sd_calibrated_pilot_rule': ({'mean': lo_raw['mean'], 'sd': lo_cal['sd']}, {'mean': hi_raw['mean'], 'sd': hi_cal['sd']}),
                'mean_and_sd_calibrated': (lo_cal, hi_cal)}
    donor_is = {}
    for v, (lo, hi) in variants.items():
        out[v] = {c: score_block(lo[c], hi[c], y[c]) for c in y}
        both_lo = np.concatenate([lo['mean'], lo['sd']], 1); both_hi = np.concatenate([hi['mean'], hi['sd']], 1)
        both_y = np.concatenate([y['mean'], y['sd']], 2)
        out[v]['combined'] = score_block(both_lo, both_hi, both_y)
        donor_is[v] = {c: per_donor_is(lo[c], hi[c], y[c]) for c in y} | {'combined': per_donor_is(both_lo, both_hi, both_y)}
    rng = np.random.default_rng(20261120)
    boots = rng.integers(0, len(donors), size=(N_BOOT, len(donors)))
    out['bootstrap_vs_raw'] = {}
    for v in ('sd_calibrated_pilot_rule', 'mean_and_sd_calibrated'):
        out['bootstrap_vs_raw'][v] = {}
        for c in ('mean', 'sd', 'combined'):
            a, b = donor_is[v][c], donor_is['raw'][c]
            rel = (a[boots].mean(1) - b[boots].mean(1)) / b[boots].mean(1)
            out['bootstrap_vs_raw'][v][c] = dict(relative_change=float((a.mean() - b.mean()) / b.mean()),
                                                 ci95=[float(np.quantile(rel, .025)), float(np.quantile(rel, .975))],
                                                 donors_improved=int((a < b).sum()), donors=len(a))
    for v in ('sd_calibrated_pilot_rule', 'mean_and_sd_calibrated'):
        groups = ('sd', 'combined') if v.startswith('sd') else ('mean', 'sd', 'combined')
        out[v]['useful'] = bool(all(abs(out[v][g]['coverage'] - .8) < abs(out['raw'][g]['coverage'] - .8) and
                                    out[v][g]['interval_score'] < out['raw'][g]['interval_score'] for g in groups))
    per_gene = {v: {c: ((y[c] >= lo[c]) & (y[c] <= hi[c])).mean(axis=(0, 1)) for c in y}
                for v, (lo, hi) in variants.items()}
    return out, per_gene, donor_is


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replicates', type=Path, nargs='+', required=True)
    p.add_argument('--eval-out', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    a = p.parse_args(argv)
    reps = load_reps(a.replicates)
    genes = reps[0]['genes'].astype(str)
    stage_t = reps[0]['stage']; held_t = reps[0]['is_heldout']; targets = reps[0]['targets'].astype(str)
    obs = C.load_observed()
    if not np.array_equal(obs['genes'], genes):
        raise ValueError('gene axes differ')
    train_donors = [str(d) for d in obs['train_donors']]
    G = len(genes)

    # generated donor-level draws per stage, pooled over replicates
    gm = np.stack([r['mean_log'] for r in reps]).astype(np.float64)               # [R,T,M,G]
    gs = np.stack([C.generated_sd(r['mean_log'], r['sq_log']) for r in reps])     # [R,T,M,G]
    raw = {}
    for c, arr in (('mean', gm), ('sd', gs)):
        lo = np.zeros((4, G)); hi = np.zeros((4, G))
        for s in range(4):
            draws = arr[:, stage_t == s].reshape(-1, G)
            lo[s], hi[s] = np.quantile(draws, [C.ALPHA / 2, 1 - C.ALPHA / 2], axis=0)
        raw[c] = (lo, hi)
    emp_raw = {c: tuple(np.repeat(q[None], 4, 0) for q in np.quantile(obs[f'empirical_{c}'], [C.ALPHA / 2, 1 - C.ALPHA / 2], axis=0))
               for c in ('mean', 'sd')}

    held = np.flatnonzero(obs['is_heldout'])
    hdonors = [str(d) for d in obs['donors'][held]]
    y_mean = obs['bag_mean'][held].transpose(1, 0, 2)    # [R,D,G]
    y_sd = obs['bag_sd'][held].transpose(1, 0, 2)
    hstage = obs['stage'][held]
    halves = C.pilot_halves()
    if sorted(sum((list(c) for c, _ in halves), [])) != sorted(hdonors):
        raise ValueError('pilot halves differ from the held-out donors')
    model_eval, model_gene_cov, model_donor_is = evaluate('model', raw, y_mean, y_sd, hdonors, hstage, train_donors, halves)
    emp_eval, _, emp_donor_is = evaluate('empirical', emp_raw, y_mean, y_sd, hdonors, hstage, train_donors, halves)
    rng = np.random.default_rng(20261121)
    boots = rng.integers(0, len(hdonors), size=(N_BOOT, len(hdonors)))
    model_vs_emp = {}
    for v in ('raw', 'sd_calibrated_pilot_rule', 'mean_and_sd_calibrated'):
        model_vs_emp[v] = {}
        for c in ('mean', 'sd', 'combined'):
            m_, e_ = model_donor_is[v][c], emp_donor_is[v][c]
            rel = (m_[boots].mean(1) - e_[boots].mean(1)) / e_[boots].mean(1)
            model_vs_emp[v][c] = dict(relative_difference=float((m_.mean() - e_.mean()) / e_.mean()),
                                      ci95=[float(np.quantile(rel, .025)), float(np.quantile(rel, .975))],
                                      donors_model_better=int((m_ < e_).sum()))

    # deployed correction: all 12 held-out donors (n=12, rank 11); cannot be evaluated on those donors
    deployed = {}
    for c, y in (('mean', y_mean), ('sd', y_sd)):
        f = C.fit_correction(raw[c][0][hstage], raw[c][1][hstage], y[0], hdonors, train_donors)
        deployed[c] = f

    # gene flags (part D)
    gen_target_mean = gm.mean(axis=(0, 2))                                         # [T,G]
    tindex = {d: i for i, d in enumerate(targets)}
    pred = gen_target_mean[[tindex[d] for d in hdonors]]
    obs_mean_held = obs['mean_log'][held]
    rmse = np.sqrt(((pred - obs_mean_held) ** 2).mean(0))
    emp_pred = obs['mean_log'][~obs['is_heldout']].mean(0)
    emp_rmse = np.sqrt(((emp_pred[None] - obs_mean_held) ** 2).mean(0))
    rmse_cut = float(np.quantile(rmse, RMSE_QUANTILE))
    W = obs['W']
    trn = ~obs['is_heldout']
    gap = np.zeros((4, G)); share = np.full((4, G), np.nan)
    for s in range(4):
        o = obs['mean_log'][trn & (obs['stage'] == s)].mean(0)
        g_ = gen_target_mean[(~held_t) & (stage_t == s)].mean(0)
        z = (o - g_) / obs['space_sd']
        outside = z - W @ (W.T @ z)
        gap[s] = z
        with np.errstate(divide='ignore', invalid='ignore'):
            share[s] = np.where(np.abs(z) > 0, outside / z, np.nan)
    pc_captured = (W ** 2).sum(1)
    detection = obs['detection'].mean(0)
    flags_poor = (rmse > rmse_cut) | (rmse > RMSE_EMPIRICAL_RATIO * emp_rmse)
    flags_blind = ((share > BLIND_SHARE) & (np.abs(gap) > BLIND_MIN_GAP)).any(0)

    # replicate stage means for the tool
    stage_mean_rep = np.zeros((len(reps), 4, G)); within_rep = np.zeros((len(reps), 4, G)); within_cells = []
    for i, r in enumerate(reps):
        per_target = r['subtype_sums'][:, 0] / r['subtype_n'][:, 0, None]
        stage_mean_rep[i] = np.stack([per_target[stage_t == s].mean(0) for s in range(4)])
        within_rep[i], cells = C.synthetic_stage_means(r['subtype_sums'], r['subtype_n'], stage_t)
        within_cells.append(cells)

    receipt0 = reps[0]['receipt']
    meta = dict(schema=BUNDLE_SCHEMA, fit=receipt0['fit'], readout=receipt0['mode'], fold=0,
                weights_sha256=receipt0['weights_sha256'], cache_manifest_sha256=receipt0['cache_manifest_sha256'],
                replicates=[int(r['replicate']) for r in reps], replicate_sha256=[r['receipt']['output_sha256'] for r in reps],
                generation_devices=sorted({r['receipt']['device'] for r in reps}), M=32, K=256,
                stages=C.STAGES, subtype_groups=[C.TYPES[g - 1] for g in C.GROUPS],
                units='mean per-cell log1p(panel CP10k); donor SD of the same over a K=256-cell bag',
                stage_mean_definition='equal weight over generated target donors of the stage (37 development-donor conditions)',
                within_subtype_definition='equal weight over the two fixed subtypes; generated cells pooled within stage; >=100 cells',
                interval_definition='10%/90% percentiles of generated donor bags pooled over targets of the stage and replicates',
                calibration=dict(rule='coverage pilot nonnegative split-CQR, rank ceil((n+1)*0.8)', alpha=C.ALPHA,
                                 deployed_calibration_donors=hdonors, deployed_rank=deployed['mean']['rank'],
                                 deployed_note='fit on all 12 held-out donors; its own coverage is not evaluable on them; '
                                               'cross-fitted 6/6 evaluation is in the evaluation JSON',
                                 crossfit_halves=[dict(calibration=list(c), evaluation=list(t)) for c, t in halves]),
                flags=dict(poorly_fit=f'held-out donor mean-log RMSE > panel {RMSE_QUANTILE:.0%} quantile ({rmse_cut:.4f}) '
                                      f'or > {RMSE_EMPIRICAL_RATIO} x stage-blind empirical RMSE',
                           pc_blind_spot=f'share of standardized TRAIN stage gap outside the 50-PC span > {BLIND_SHARE} '
                                         f'in any stage with |gap| > {BLIND_MIN_GAP}',
                           low_expression=f'observed mean detection < {LOW_DETECTION:.0%}'),
                observed_donors=dict(train=train_donors, heldout=hdonors),
                source_sha256=C.sha(__file__), common_sha256=C.sha(C.__file__))
    a.bundle.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.bundle, genes=genes, stage_mean_rep=stage_mean_rep, within_rep=within_rep,
        within_cells=np.stack(within_cells),
        interval_lo=np.stack([raw['mean'][0], raw['sd'][0]]), interval_hi=np.stack([raw['mean'][1], raw['sd'][1]]),
        correction=np.stack([deployed['mean']['correction'], deployed['sd']['correction']]),
        crossfit_coverage_raw=np.stack([model_gene_cov['raw']['mean'], model_gene_cov['raw']['sd']]),
        crossfit_coverage_cal=np.stack([model_gene_cov['mean_and_sd_calibrated']['mean'], model_gene_cov['mean_and_sd_calibrated']['sd']]),
        obs_donors=obs['donors'], obs_stage=obs['stage'], obs_heldout=obs['is_heldout'],
        obs_mean_log=obs['mean_log'].astype(np.float32), obs_subtype_sums=obs['subtype_sums'].astype(np.float32),
        obs_subtype_n=obs['subtype_n'], empirical_stage_blind_mean=emp_pred,
        empirical_interval_lo=np.stack([emp_raw['mean'][0][0], emp_raw['sd'][0][0]]),
        empirical_interval_hi=np.stack([emp_raw['mean'][1][0], emp_raw['sd'][1][0]]),
        heldout_rmse=rmse, empirical_rmse=emp_rmse, rmse_cut=np.array(rmse_cut), pc_gap=gap, pc_outside_share=share,
        pc_captured=pc_captured, detection=detection, flag_poorly_fit=flags_poor, flag_pc_blind_spot=flags_blind,
        flag_low_expression=detection < LOW_DETECTION)
    meta['calibration']['heldout_crossfit_evaluation'] = dict(
        raw={c: {k: model_eval['raw'][c][k] for k in ('coverage', 'interval_score')} for c in ('mean', 'sd', 'combined')},
        calibrated={c: {k: model_eval['mean_and_sd_calibrated'][c][k] for k in ('coverage', 'interval_score')}
                    for c in ('mean', 'sd', 'combined')},
        calibration_useful=dict(sd_only_pilot_rule=model_eval['sd_calibrated_pilot_rule']['useful'],
                                mean_and_sd=model_eval['mean_and_sd_calibrated']['useful']),
        note=('raw stage-level gene intervals over-cover on held-out donors; the nonnegative pilot rule can only widen them, '
              'so prefer raw unless the evaluation says calibration is useful'
              if not model_eval['mean_and_sd_calibrated']['useful'] else 'calibration improved held-out coverage and score'))
    C.write_json(a.bundle.with_suffix('.json'), meta | dict(bundle_npz_sha256=C.sha(a.bundle)))

    j = list(genes).index('PPARG') if 'PPARG' in genes else None
    result = dict(
        status='completed_gene_level_interval_evaluation', protocol='analysis/query_tool_20261120/PROTOCOL.md',
        scope='one native EMPF fit (fold0/seed0); 12 held-out development donors x 24 bags x 2000 genes; '
              'cross-fitted with the coverage pilot halves; reused cohort, no independent coverage guarantee',
        replicates=meta['replicates'], model=model_eval, stage_blind_empirical=emp_eval,
        model_minus_empirical_interval_score=model_vs_emp,
        deployed_correction=dict(rank=deployed['mean']['rank'], n=deployed['mean']['n_calibration_donors'],
                                 median_mean_correction=float(np.median(deployed['mean']['correction'])),
                                 median_sd_correction=float(np.median(deployed['sd']['correction']))),
        gene_flags=dict(poorly_fit=int(flags_poor.sum()), pc_blind_spot=int(flags_blind.sum()),
                        low_expression=int((detection < LOW_DETECTION).sum()), rmse_cut=rmse_cut,
                        model_rmse_median=float(np.median(rmse)), empirical_rmse_median=float(np.median(emp_rmse)),
                        genes_model_rmse_below_empirical=int((rmse < emp_rmse).sum()),
                        PPARG=(dict(outside_share_by_stage=[None if not np.isfinite(v) else float(v) for v in share[:, j]],
                                    gap_by_stage=gap[:, j].tolist(), pc_captured=float(pc_captured[j]),
                                    heldout_rmse=float(rmse[j]), empirical_rmse=float(emp_rmse[j]),
                                    flag_pc_blind_spot=bool(flags_blind[j]), flag_poorly_fit=bool(flags_poor[j])) if j is not None else None)),
        bundle=str(a.bundle), bundle_sha256=C.sha(a.bundle),
        inputs={str(p): C.sha(p) for p in a.replicates}, source_sha256=C.sha(__file__))
    C.write_json(a.eval_out, result)
    print(json.dumps({k: result[k] for k in ('gene_flags', 'deployed_correction')}, indent=2))
    for v in ('raw', 'sd_calibrated_pilot_rule', 'mean_and_sd_calibrated'):
        print(v, {c: (round(model_eval[v][c]['coverage'], 4), round(model_eval[v][c]['interval_score'], 4)) for c in ('mean', 'sd', 'combined')},
              'emp', {c: (round(emp_eval[v][c]['coverage'], 4), round(emp_eval[v][c]['interval_score'], 4)) for c in ('mean', 'sd', 'combined')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
