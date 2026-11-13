"""Finish the frozen r4 confirmation's distribution analysis on existing outputs.

Reads synthetic SEA-AD summaries and the37-donor caches only. Saved external
baseline SCORES may be read; external count files are never opened. No model
updates, generation, calibration, new primary rule, or default selection.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
import time
import numpy as np
import evaluate as E
from resume_confirmation_profiles import C

RUNS = E.RUNS
OUT = RUNS/'results_r5/confirmation_distributions'
MODES = ('snapens', 'stagehalf', 'biascal@stagehalf')
ARMS = ('pbulk', 'g.dispz,pbulk')


def scored(label, fold, seed, generated, real, sources):
    path = OUT/'scores'/label/f'fold{fold}/seed{seed}.json'
    fingerprint = {'sources': sources, 'cache_sha256': E.sha256(real.cache_dir/'manifest.json'),
                   'evaluator_sha256': E.sha256(Path(E.__file__)),
                   'metrics_sha256': E.sha256(Path(E.__file__).with_name('metrics.py'))}
    if path.exists():
        existing = json.loads(path.read_text())
        if existing['fingerprint'] != fingerprint:
            raise ValueError(f'stale confirmation score {path}')
        return existing['headline']
    result, _ = E.score_generated(generated, real, {'all': (np.arange(len(real.genes)), False)})
    value = {'fingerprint': fingerprint, 'headline': E.headline(result), 'M': int(generated['pb_counts'].shape[1])}
    E.write_json(path, value)
    return value['headline']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table, sources, reproduced, baseline_sources, missing_sources = {}, [], [], [], []
    started = time.monotonic()
    for fold in range(9):
        real = E.load_real(RUNS/f'caches/hvg2000/fold{fold}')
        for root, panel, tag, oldtree in (('fits_r3', 'hvg2000', 'r3', 'results_r3'),
                                          ('fits_r4', 'hvg2000x', 'x', 'results_r4')):
            for arm in ARMS:
                for mode in MODES:
                    label = f'{arm}~{tag}#{mode}'
                    members, member_sources = [], []
                    for seed in (0, 1):
                        path = RUNS/root/arm/panel/f'fold{fold}/seed{seed}'/f'generated_{mode}.npz'
                        if not path.exists():
                            missing_sources.append(str(path))
                            continue
                        generated = E.align_generated(E.load_generated(path), real)
                        if generated['pb_counts'].shape[1] != 32:
                            raise ValueError('single-fit confirmation requires M32')
                        fingerprint = {'path': str(path), 'sha256': E.sha256(path)}
                        flat = scored(label, fold, seed, generated, real, [fingerprint])
                        table[(label, fold, seed)] = flat
                        members.append(generated)
                        member_sources.append(fingerprint)
                        sources.append(fingerprint)
                        old = RUNS/oldtree/'fits'/label/'hvg2000'/f'fold{fold}/seed{seed}.json'
                        if old.exists():
                            oldflat = E.headline(json.loads(old.read_text()))
                            differences = {key: abs(value-oldflat[key]) for key, value in flat.items()
                                           if value is not None and oldflat.get(key) is not None
                                           and np.isfinite(value) and np.isfinite(oldflat[key])}
                            if not differences or max(differences.values()) > 1e-12:
                                raise ValueError(f'historical evaluator mismatch {old}: {differences}')
                            reproduced.append({'path': str(old), 'n_metrics': len(differences),
                                               'max_absolute_difference': max(differences.values())})
                    if len(members) == 2:
                        ensemble = E.concat_generated(members)
                        ensemble_label = label+C.ENS_SUFFIX
                        table[(ensemble_label, fold, None)] = scored(ensemble_label, fold, 'ensemble64', ensemble,
                                                                     real, member_sources)
                        matched_label = label+' [seed-ensemble n=2 M=32]'
                        table[(matched_label, fold, None)] = scored(matched_label, fold, 'ensemble32',
                            E.mmatched_generated(members), real, member_sources)
        for name in C.REF_BASELINES+C.X_BASELINES:
            # These files contain aggregate evaluation scores, not counts.
            path = RUNS/'results_r4/baselines/hvg2000'/f'fold{fold}/{name}.json'
            if not path.exists():
                path = RUNS/'results_r3/baselines/hvg2000'/f'fold{fold}/{name}.json'
            if not path.exists():
                raise FileNotFoundError(f'missing saved baseline score {name}, fold{fold}')
            table[(f'baseline:{name}', fold, None)] = E.headline(json.loads(path.read_text()))
            baseline_sources.append({'path': str(path), 'sha256': E.sha256(path)})
        E.write_json(OUT/'status.json', {'status': 'running', 'completed_folds': fold+1, 'pid': os.getpid(),
                                        'wall_seconds': time.monotonic()-started})
        print(f'fold{fold} complete', flush=True)
    result = C.analyse(table)
    if not result['primary']['complete']:
        raise ValueError('missing primary confirmation cells')
    result['scope'] = ('Existing synthetic SEA-AD outputs and37 development-donor caches only; '
                       'saved external baseline scores read without opening external counts. '
                       'Secondary distribution comparisons do not override the original primary null.')
    result['reproduction'] = reproduced
    result['sources'] = sources
    result['baseline_sources'] = baseline_sources
    result['missing_source_files'] = missing_sources
    result['driver_sha256'] = E.sha256(Path(__file__))
    result['factor_analysis_sha256'] = E.sha256(Path(C.__file__))
    result['wall_seconds'] = time.monotonic()-started
    result['M32_matched_ensembles_secondary'] = {mode: {
        cell: {metric: C.baseline_rows(
            {(f'baseline:{cell}', f, None): table[(f'{label}#{mode} [seed-ensemble n=2 M=32]', f, None)]
             for f in range(9) if (f'{label}#{mode} [seed-ensemble n=2 M=32]', f, None) in table},
             C.CONFIRM_REPEATS, (cell,), metric)[f'baseline:{cell}']
             for metric in ('pb_rmse', *C.DIST_METRICS)}
        for cell, label in C.cell_labels().items()} for mode in MODES}
    E.write_json(OUT/'analysis.json', C._jsonable(result))
    (OUT/'analysis.txt').write_text(C.report(result)+'\n'+result['scope']+'\n')
    E.write_json(OUT/'status.json', {'status': 'completed', 'completed_folds': 9,
                                    'wall_seconds': result['wall_seconds'], 'reproduced_scores': len(reproduced)})
    print(C.report(result), flush=True)


if __name__ == '__main__':
    E.set_torch_threads()
    main()
