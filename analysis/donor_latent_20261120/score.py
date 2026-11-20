#!/usr/bin/env python3
"""Score the donor-latent comparison with the frozen r5 metrics (see PROTOCOL.md).

    score.py empirical --fold K   stage-blind (and same-stage) empirical resampling reference, M=32, K=256
    score.py score --fold K       control / variant archives at the declared readout
    score.py compare              paired donor-bootstrap comparison -> completion/comparison.json
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'analysis/improve_20261109/r5'))
sys.path.insert(0, str(HERE))
import architecture_score as AS  # noqa: E402  frozen r5 scorer (historical + interval metrics)
import interval_metrics as I      # noqa: E402

RUNS = Path('/ml2/alz/runs/research_20261120/donor-latent')
CACHES = Path('/ml2/alz/runs/improve_20261109/caches/hvg2000')
SCORES = RUNS / 'scores'
COMPLETION = HERE / 'completion'
UPDATES = 10000
MODE = 'cellshape@0.5:biascal:stagehalf'
FOLDS = (0, 1)
ARMS = {'control': 'pbulk', 'variant': 'g.dre,pbulk'}
EMPIRICAL = {'empirical_blind': 'resample_blind', 'empirical_stage': 'resample'}
BLOCKS = {'pc1_10': slice(0, 10), 'pc11_30': slice(10, 30), 'pc31_50': slice(30, 50), 'all50': slice(0, 50)}
GUARDS = {'pb_rmse': .005, 'meanlog_rmse_gene': .02, 'nonzero_rmse_gene': .02, 'logvar_rmse_gene': .02,
          'pc_energy_rep': .02, 'cell_energy_rep': .02}
N_BOOT = 10000
BOOT_SEED = 20261120


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def archive(label, fold):
    if label in ARMS:
        return RUNS / 'generated' / f'u{UPDATES}' / ARMS[label] / f'fold{fold}' / 'seed0' / f'{MODE}.npz'
    return RUNS / 'empirical' / label / f'fold{fold}.npz'


def per_donor(samples, observed):
    """Per-donor percentile-interval diagnostics on the 50 SD coordinates (and blocks) and 50 mean coordinates.

    samples [T, M, 100] generated bag summaries; observed [R, T, 100] real 64-cell bag summaries.
    Returns {group: {coverage, interval_score, width, below, above: [T]}} averaged over repeats and coordinates.
    """
    low, high = I.bounds(samples, 'percentile')
    out = {}
    groups = {f'sd_{b}': slice(50 + s.start, 50 + s.stop) for b, s in BLOCKS.items()}
    groups['mean_all50'] = slice(0, 50)
    groups['combined_all100'] = slice(0, 100)
    for name, sl in groups.items():
        lo, hi, y = low[:, sl], high[:, sl], observed[:, :, sl]
        out[name] = dict(coverage=((y >= lo) & (y <= hi)).mean((0, 2)), interval_score=I.interval_score(lo, hi, y).mean((0, 2)),
                         width=np.broadcast_to(hi - lo, y.shape).mean((0, 2)), below=(y < lo).mean((0, 2)), above=(y > hi).mean((0, 2)))
    return out


def spread_diagnostic(samples, observed, real):
    """Between-donor spread of the SD summaries: generated (within target, across M draws) vs real (across donors within stage)."""
    gen_sd = samples[:, :, 50:].std(1, ddof=1).mean()          # mean over targets/coords of across-draw SD
    obs = observed[0]
    stage = np.asarray(real.stage)
    ws = []
    for s in np.unique(stage):
        m = stage == s
        if m.sum() > 1:
            ws.append(obs[m, 50:].std(0, ddof=1).mean())
    return {'generated_between_draw_sd_of_SD_summary': float(gen_sd),
            'real_within_stage_between_donor_sd_of_SD_summary': float(np.mean(ws))}


def score_archive(label, fold, real, observed):
    path = archive(label, fold)
    out = SCORES / f'fold{fold}' / f'{label}.json'
    frozen = AS.score(path, CACHES / f'fold{fold}', SCORES / f'fold{fold}' / f'{label}.frozen.json')
    generated = AS.E.align_generated(AS.E.load_generated(path), real)
    samples = AS.E.MX.pc_bag_summary(generated['pcs'].astype(np.float32))
    if samples.shape[1] != 32 or generated['pcs'].shape[2] != 64:
        raise ValueError('unexpected archive shape')
    donors = per_donor(samples, observed)
    value = dict(label=label, fold=fold, archive=str(path), archive_sha256=sha(path), score_code_sha256=sha(__file__),
                 donors=real.donors.tolist(), is_heldout=real.is_heldout.tolist(), stage=np.asarray(real.stage).tolist(),
                 historical=frozen['historical'], intervals=frozen['intervals'],
                 per_donor={g: {k: v.tolist() for k, v in d.items()} for g, d in donors.items()},
                 spread=spread_diagnostic(samples, observed, real))
    write(out, value)
    return value


def load_fold(fold):
    real = AS.E.load_real(CACHES / f'fold{fold}')
    observed = AS.observed_summaries(real, 64)
    return real, observed


def run_empirical(fold):
    import baselines as BL
    real, observed = load_fold(fold)
    for label, name in EMPIRICAL.items():
        path = archive(label, fold)
        if not path.exists():
            # Same stream namespace as analysis/empirical_challenge_20261114 empirical32 (replicate 0).
            BL._target_rng = lambda r, t: np.random.default_rng(AS.E.MX.stable_seed(
                'empirical-challenge', 20261114, r.panel, r.fold, r.donors[t], 0))
            arrays, info = BL.generate(name, real, M=32, K=256, kpc=64)
            path.parent.mkdir(parents=True, exist_ok=True)
            AS.E.save_npz(path, **arrays)
            write(path.with_suffix('.receipt.json'), dict(info, baseline=name, generated_SHA256=sha(path), fold=fold,
                                                         stream="stable_seed('empirical-challenge',20261114,panel,fold,donor,0)"))
        v = score_archive(label, fold, real, observed)
        print(json.dumps({label: v['intervals']['heldout']['percentile']['sd']}), flush=True)


def run_score(fold):
    real, observed = load_fold(fold)
    for label in ARMS:
        v = score_archive(label, fold, real, observed)
        print(json.dumps({label: v['intervals']['heldout']['percentile']['sd']}), flush=True)


def boot_ci(diffs_by_fold, rng):
    """Paired donor bootstrap: resample held-out donors within each fold; statistic = mean over folds of mean donor diff."""
    point = float(np.mean([d.mean() for d in diffs_by_fold]))
    stats = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        stats[b] = np.mean([d[rng.integers(len(d), size=len(d))].mean() for d in diffs_by_fold])
    lo, hi = np.quantile(stats, [.025, .975])
    return dict(point=point, ci95=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0),
                n_donors=[int(len(d)) for d in diffs_by_fold])


def compare():
    rows = {f: {label: json.loads((SCORES / f'fold{f}' / f'{label}.json').read_text()) for label in (*ARMS, *EMPIRICAL)}
            for f in FOLDS if (SCORES / f'fold{f}' / 'variant.json').exists()}
    folds = sorted(rows)
    if 0 not in folds:
        raise ValueError('fold0 scores missing')
    rng = np.random.default_rng(BOOT_SEED)

    def donor_metric(f, label, group, metric, split='heldout'):
        r = rows[f][label]
        mask = np.array(r['is_heldout']) if split == 'heldout' else ~np.array(r['is_heldout'])
        return np.array(r['per_donor'][group][metric])[mask]

    summary = {}
    for f in folds:
        summary[f'fold{f}'] = {label: dict(
            heldout=r['intervals']['heldout']['percentile'], train=r['intervals']['train']['percentile'],
            historical={k: r['historical'][k] for k in GUARDS}, spread=r['spread'],
            sd_blocks_heldout={b: {m: float(donor_metric(f, label, f'sd_{b}', m).mean()) for m in ('coverage', 'interval_score', 'width', 'below', 'above')} for b in BLOCKS},
            sd_blocks_train={b: {m: float(donor_metric(f, label, f'sd_{b}', m, 'train').mean()) for m in ('coverage', 'interval_score', 'width')} for b in BLOCKS})
            for label, r in rows[f].items()}
    contrasts = {}
    pairs = [('variant', 'control'), ('variant', 'empirical_blind'), ('control', 'empirical_blind'),
             ('variant', 'empirical_stage')]
    for a, b in pairs:
        key = f'{a}_minus_{b}'
        contrasts[key] = {}
        for group in ('sd_all50', 'sd_pc1_10', 'sd_pc11_30', 'sd_pc31_50', 'combined_all100', 'mean_all50'):
            for metric in ('interval_score', 'coverage'):
                diffs = [donor_metric(f, a, group, metric) - donor_metric(f, b, group, metric) for f in folds]
                contrasts[key][f'{group}/{metric}'] = boot_ci(diffs, rng)
                contrasts[key][f'{group}/{metric}']['per_fold'] = [float(d.mean()) for d in diffs]
        for metric in ('interval_score', 'coverage'):  # motivating train-donor diagnostic
            diffs = [donor_metric(f, a, 'sd_all50', metric, 'train') - donor_metric(f, b, 'sd_all50', metric, 'train') for f in folds]
            contrasts[key][f'train_sd_all50/{metric}'] = boot_ci(diffs, rng)
    v, c, e = 'variant', 'control', 'empirical_blind'
    sd = lambda f, lab: rows[f][lab]['intervals']['heldout']['percentile']['sd']
    decision = dict(
        sd_interval_score_lower_ci=contrasts['variant_minus_control']['sd_all50/interval_score']['ci95'][1] < 0,
        sd_coverage_closer_to_80_every_fold=all(abs(sd(f, v)['coverage'] - .8) < abs(sd(f, c)['coverage'] - .8) for f in folds),
        guards={f'fold{f}/{k}': rows[f][v]['historical'][k] <= rows[f][c]['historical'][k] * (1 + lim)
                for f in folds for k, lim in GUARDS.items()})
    decision['guards_all_pass'] = all(decision['guards'].values())
    decision['variant_improves_on_control'] = bool(decision['sd_interval_score_lower_ci'] and
                                                   decision['sd_coverage_closer_to_80_every_fold'] and decision['guards_all_pass'])
    decision['variant_beats_stage_blind_empirical_sd_score'] = contrasts['variant_minus_empirical_blind']['sd_all50/interval_score']['ci95'][1] < 0
    value = dict(status='completed', protocol_sha256=sha(HERE / 'PROTOCOL.md'), score_code_sha256=sha(__file__),
                 folds=folds, updates=UPDATES, mode=MODE, n_boot=N_BOOT, boot_seed=BOOT_SEED,
                 summary=summary, contrasts=contrasts, decision=decision,
                 score_files={f'fold{f}/{lab}': sha(SCORES / f'fold{f}' / f'{lab}.json') for f in folds for lab in rows[f]})
    write(COMPLETION / 'comparison.json', value)
    print(json.dumps(dict(decision=decision, sd_heldout={f'fold{f}': {lab: sd(f, lab) for lab in rows[f]} for f in folds},
                          key=contrasts['variant_minus_control']['sd_all50/interval_score']), indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('empirical', 'score', 'compare'))
    p.add_argument('--fold', type=int)
    a = p.parse_args()
    if a.phase == 'empirical':
        run_empirical(a.fold)
    elif a.phase == 'score':
        run_score(a.fold)
    else:
        compare()
