#!/usr/bin/env python3
"""Seed-1 training replicate (PROTOCOL_ADDENDUM_SEED1.md), scored with the unchanged score.py functions.

    seed_replicate.py score --fold K    score seed-1 control/variant archives -> RUNS/scores_seed1/foldK
    seed_replicate.py compare           seed contrasts -> completion/seed_replicate.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score as S  # noqa: E402  sets thread/CUDA env and imports the frozen r5 scorer

SEED = 1
SCORES = {0: S.SCORES, 1: S.RUNS / 'scores_seed1'}
GROUPS = ('sd_all50', 'sd_pc1_10', 'sd_pc11_30', 'sd_pc31_50', 'combined_all100')


def seed_archive(label, fold):
    return S.RUNS / 'generated' / f'u{S.UPDATES}' / S.ARMS[label] / f'fold{fold}' / f'seed{SEED}' / f'{S.MODE}.npz'


def run_score(fold):
    S.archive = seed_archive
    S.SCORES = SCORES[1]
    real, observed = S.load_fold(fold)
    for label in S.ARMS:
        v = S.score_archive(label, fold, real, observed)
        print(json.dumps({label: v['intervals']['heldout']['percentile']['sd']}), flush=True)


def compare():
    rows = {(s, f, lab): json.loads((SCORES[s] / f'fold{f}' / f'{lab}.json').read_text())
            for s in (0, 1) for f in S.FOLDS for lab in S.ARMS}
    for f in S.FOLDS:
        for s in (0, 1):
            if rows[(s, f, 'control')]['donors'] != rows[(0, f, 'control')]['donors']:
                raise ValueError('donor order differs between seeds')

    def dm(s, f, lab, group='sd_all50', metric='interval_score'):
        r = rows[(s, f, lab)]
        return np.array(r['per_donor'][group][metric])[np.array(r['is_heldout'])]

    rng = np.random.default_rng(S.BOOT_SEED + 1)
    out = {}
    for group in GROUPS:
        for metric in ('interval_score', 'coverage'):
            key = f'{group}/{metric}'
            c = {}
            for name, fn in {
                'seed1_variant_minus_control': lambda f: dm(1, f, 'variant', group, metric) - dm(1, f, 'control', group, metric),
                'seed0_variant_minus_control': lambda f: dm(0, f, 'variant', group, metric) - dm(0, f, 'control', group, metric),
                'control_seed1_minus_seed0': lambda f: dm(1, f, 'control', group, metric) - dm(0, f, 'control', group, metric),
                'variant_seed1_minus_seed0': lambda f: dm(1, f, 'variant', group, metric) - dm(0, f, 'variant', group, metric),
                'two_seed_variant_minus_control': lambda f: .5 * ((dm(0, f, 'variant', group, metric) - dm(0, f, 'control', group, metric))
                                                                  + (dm(1, f, 'variant', group, metric) - dm(1, f, 'control', group, metric))),
            }.items():
                diffs = [fn(f) for f in S.FOLDS]
                c[name] = S.boot_ci(diffs, rng)
                c[name]['per_fold'] = [float(d.mean()) for d in diffs]
            out[key] = c
    sd = {f'seed{s}/fold{f}/{lab}': rows[(s, f, lab)]['intervals']['heldout']['percentile']['sd'] for s in (0, 1) for f in S.FOLDS for lab in S.ARMS}
    comb = {f'seed{s}/fold{f}/{lab}': rows[(s, f, lab)]['intervals']['heldout']['percentile']['combined'] for s in (0, 1) for f in S.FOLDS for lab in S.ARMS}
    hist = {f'seed{s}/fold{f}/{lab}': {k: rows[(s, f, lab)]['historical'][k] for k in S.GUARDS} for s in (0, 1) for f in S.FOLDS for lab in S.ARMS}
    spread = {f'seed{s}/fold{f}/{lab}': rows[(s, f, lab)]['spread'] for s in (0, 1) for f in S.FOLDS for lab in S.ARMS}
    k = out['sd_all50/interval_score']
    seed0_effect = k['seed0_variant_minus_control']['point']
    noise = max(abs(k['control_seed1_minus_seed0']['point']), abs(k['variant_seed1_minus_seed0']['point']))
    replicated = bool(k['seed1_variant_minus_control']['point'] > 0 and k['two_seed_variant_minus_control']['ci95'][0] > 0
                      and noise < abs(seed0_effect))
    guards1 = {f'fold{f}/{g}': rows[(1, f, 'variant')]['historical'][g] <= rows[(1, f, 'control')]['historical'][g] * (1 + lim)
               for f in S.FOLDS for g, lim in S.GUARDS.items()}
    seed1_improves = bool(k['seed1_variant_minus_control']['ci95'][1] < 0
                          and all(abs(sd[f'seed1/fold{f}/variant']['coverage'] - .8) < abs(sd[f'seed1/fold{f}/control']['coverage'] - .8) for f in S.FOLDS)
                          and all(guards1.values()))
    value = dict(status='completed', addendum_sha256=S.sha(HERE / 'PROTOCOL_ADDENDUM_SEED1.md'), code_sha256=S.sha(__file__),
                 score_code_sha256=S.sha(S.__file__), seeds=[0, 1], folds=list(S.FOLDS), n_boot=S.N_BOOT, boot_seed=S.BOOT_SEED + 1,
                 heldout_sd=sd, heldout_combined=comb, historical=hist, spread=spread, contrasts=out,
                 decision=dict(seed0_effect=seed0_effect, max_abs_native_seed_to_seed=noise, worse_replicated=replicated,
                               seed1_guards=guards1, seed1_variant_improves_on_control=seed1_improves),
                 score_files={f'seed1/fold{f}/{lab}': S.sha(SCORES[1] / f'fold{f}' / f'{lab}.json') for f in S.FOLDS for lab in S.ARMS},
                 archives={f'seed1/fold{f}/{lab}': rows[(1, f, lab)]['archive_sha256'] for f in S.FOLDS for lab in S.ARMS})
    S.write(S.COMPLETION / 'seed_replicate.json', value)
    print(json.dumps(dict(decision=value['decision'], sd=sd, key=k), indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('score', 'compare'))
    p.add_argument('--fold', type=int)
    a = p.parse_args()
    run_score(a.fold) if a.phase == 'score' else compare()
