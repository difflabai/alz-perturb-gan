"""Reconstruct fixed SEA-AD resampling streams as interval references.

No fitting, parameter selection or default change. Preserve original profile
archives/scores; one empirical stream per fold is not two model seed draws.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '3')

import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import architecture_score as AS
import baselines as BL
from compare_architecture_recipes import fixed_references, EVIDENCE_SHA256

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RUNS = HERE.parents[2]/'runs/improve_20261109'
OUT = RUNS/'results_r5/architecture_pass/empirical_interval_references'
MODES = RUNS/'smoke_r5/architecture_pass/empirical_interval_references'
LABELS = ('resample', 'resample_blind')
HISTORICAL = ('pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene',
              'pc_energy_rep', 'cell_energy_rep', 'pc_spread_rep', 'pc_coverage80_rep')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def identical(a, b):
    if a is None or b is None:
        return a is b
    return a == b or (isinstance(a, (int, float)) and isinstance(b, (int, float))
                      and math.isnan(a) and math.isnan(b))


def decision_values(row):
    values = {name: row['historical'][name] for name in HISTORICAL}
    values['abs_log_spread'] = abs(math.log(values['pc_spread_rep']))
    for split in ('train', 'heldout'):
        for group in ('mean', 'sd', 'combined'):
            block = row['intervals'][split]['percentile'][group]
            for name in ('coverage', 'width', 'interval_score'):
                values[f'{split}/{group}/{name}'] = block[name]
            values[f'{split}/{group}/coverage_distance'] = abs(block['coverage'] - .8)
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('nonfinite descriptive comparison metric')
    return values


def aggregate(rows):
    return {'n_archives': len(rows),
            'historical': {name: float(np.mean([r['historical'][name] for r in rows])) for name in HISTORICAL},
            'intervals': {split: {method: {group: {name: float(np.mean([
                r['intervals'][split][method][group][name] for r in rows]))
                for name in ('coverage', 'width', 'interval_score')}
                for group in ('mean', 'sd', 'combined')} for method in ('percentile', 'rank')}
                for split in ('train', 'heldout')}}


def compare(models, baseline):
    model_values = {key: decision_values(row) for key, row in models.items()}
    baseline_values = {fold: decision_values(row) for fold, row in baseline.items()}
    result = {}
    for name in next(iter(model_values.values())):
        per_fold = {fold: float(np.mean([model_values[(fold, seed)][name] for seed in (0, 1)])
                                - baseline_values[fold][name]) for fold in range(3, 9)}
        result[name] = {'mean_difference': float(np.mean(list(per_fold.values()))),
                        'per_fold': per_fold,
                        'repeat_means': {str(repeat): float(np.mean([v for f, v in per_fold.items()
                                                                   if f // 3 == repeat])) for repeat in (1, 2)},
                        'n_folds_lower': sum(v < 0 for v in per_fold.values())}
    return result


def main():
    default_sha = sha(STUDY/'default_config.json')
    count_shape, original = fixed_references()
    source_names = ('score_empirical_intervals.py', 'architecture_score.py', 'evaluate.py',
                    'metrics.py', 'interval_metrics.py', 'baselines.py', 'compare_architecture_recipes.py')
    sources = {name: sha(HERE/name) for name in source_names}
    for name in ('baselines.py', 'metrics.py'):
        if sources[name] != sha(STUDY/'r3'/name):
            raise ValueError(f'baseline reconstruction must use identical frozen r3 math: {name}')
    references, artifacts, historical_checks = {label: {} for label in LABELS}, [], 0
    for fold in range(3, 9):
        cache = RUNS/f'caches/hvg2000/fold{fold}'
        manifest = read(cache/'manifest.json')
        cache_sha = sha(cache/'manifest.json')
        real = AS.E.load_real(cache)
        for label in LABELS:
            old_path = RUNS/f'results_r3/baselines/hvg2000/fold{fold}/{label}.json'
            profile_archive = old_path.with_suffix('.npz')
            old, old_sha, profile_sha = read(old_path), sha(old_path), sha(profile_archive)
            expected = {'kind': 'baseline', 'baseline': label, 'fold': fold, 'T': 37, 'M': 32,
                        'seed': None, 'cache_manifest_sha256': cache_sha,
                        'train_donors': manifest['train_donors'], 'heldout_donors': manifest['heldout_donors']}
            if any(old.get(k) != v for k, v in expected.items()):
                raise ValueError(f'wrong saved baseline metadata: {old_path}')
            if any(old['baseline_info'].get(k) != v for k, v in {'M': 32, 'K': 256, 'kpc': 64}.items()):
                raise ValueError(f'wrong saved generation sizes: {old_path}')
            if old['baseline_version'] != BL.BASELINE_VERSION:
                raise ValueError('baseline version changed')
            archive = MODES/label/f'fold{fold}.npz'
            path = OUT/label/f'fold{fold}.json'
            metadata = {'baseline': label, 'old_score_sha256': old_sha,
                'original_profile_archive_sha256': profile_sha, 'baseline_module_sha256': sources['baselines.py'],
                'math_sources': {name: sources[name] for name in ('baselines.py', 'metrics.py', 'evaluate.py')},
                'generation': old['baseline_info'], 'random_streams_per_fold': 1}
            if path.exists() and not archive.exists():
                row = read(path)
                fp = row['fingerprint']
                expected_fp = {'cache_sha256': cache_sha, 'score_code_sha256': sources['architecture_score.py'],
                               'interval_code_sha256': sources['interval_metrics.py']}
                if (row.get('empirical_reference') != metadata or row.get('pruned_source_sha256') != fp['generated_sha256']
                        or any(fp.get(k) != v for k, v in expected_fp.items())):
                    raise ValueError(f'stale pruned empirical score: {path}')
            else:
                if not archive.exists():
                    generated, info = BL.generate(label, real, M=32, K=256, kpc=64)
                    if info != old['baseline_info'] or generated['pcs'].shape != (37, 32, 64, 50):
                        raise ValueError('reconstructed generation metadata differs from the original')
                    AS.E.save_npz(archive, **generated)
                    del generated
                AS.score(archive, cache, path)
                row = read(path)  # Use the scorer's normalized JSON representation, including null diagnostics.
            if archive.exists():
                subsets = AS.E.gene_subsets(real)
                if set(subsets) != set(old['subset_names']):
                    raise ValueError('saved historical gene subsets differ')
                generated = AS.E.align_generated(AS.E.load_generated(archive), real)
                full, _ = AS.E.score_generated(generated, real, subsets)
                reproduced = AS.E._clean(AS.E.headline(full))
                del generated, full
            else:
                reproduced = row['saved_historical_headline_reproduction']
            previous = AS.E.headline(old)
            if set(previous) != set(reproduced):
                raise ValueError('historical headline fields changed')
            mismatches = {key: [value, reproduced[key]] for key, value in previous.items()
                          if not identical(value, reproduced[key])}
            if mismatches:
                raise ValueError(f'historical baseline scores do not reproduce: {old_path}: {mismatches}')
            historical_checks += sum(value is not None and math.isfinite(value) for value in previous.values())
            row['saved_historical_headline_reproduction'] = reproduced
            row['empirical_reference'] = metadata
            row['pruned_source_sha256'] = row['fingerprint']['generated_sha256']
            write(path, row)
            archive.unlink(missing_ok=True)
            if sha(profile_archive) != profile_sha or sha(old_path) != old_sha:
                raise ValueError('an original baseline input changed during scoring')
            references[label][fold] = row
            artifacts.append({'baseline': label, 'fold': fold, 'original_profile_archive': str(profile_archive),
                'original_profile_archive_sha256': profile_sha, 'old_score_sha256': old_sha, 'score': str(path),
                'reconstructed_summary_sha256': row['pruned_source_sha256'], 'temporary_summary_pruned': True,
                'score_sha256': sha(path), 'cache_manifest_sha256': cache_sha,
                'depth_unmatched_fraction_all_targets': old['baseline_info']['frac_cells_depth_unmatched']})
            print(f'completed {label} fold{fold}; historical scores exact', flush=True)
    if sha(STUDY/'default_config.json') != default_sha:
        raise ValueError('default changed during descriptive scoring; inspect before interpreting')
    if sources != {name: sha(HERE/name) for name in source_names}:
        raise ValueError('scoring/reconstruction sources changed during this run')
    models = {'original_biascal': original, 'selected_count_shape': count_shape}
    value = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'complete_descriptive_reference_comparison',
        'protocol': 'PROTOCOL_EMPIRICAL_INTERVAL_REFERENCE_20261113.txt',
        'protocol_sha256': sha(STUDY/'PROTOCOL_EMPIRICAL_INTERVAL_REFERENCE_20261113.txt'),
        'sources': sources, 'baseline_and_metrics_identical_to_frozen_r3': True,
        'original_promotion_evidence_sha256': EVIDENCE_SHA256,
        'n_folds': 6, 'n_model_seeds_per_fold': 2, 'n_empirical_streams_per_fold_recipe': 1,
        'n_empirical_archives': len(artifacts), 'historical_numeric_values_exact': historical_checks,
        'means': {**{name: aggregate(list(rows.values())) for name, rows in models.items()},
                  **{name: aggregate(list(rows.values())) for name, rows in references.items()}},
        'comparisons_model_minus_empirical': {f'{model}/{label}': compare(rows, references[label])
            for model, rows in models.items() for label in LABELS},
        'artifacts': artifacts, 'default_config_sha256': default_sha, 'default_changed': False,
        'limits': ['Descriptive post-selection comparison on the same37 donors; no significance or promotion rule',
            'Six equally weighted folds; two model seeds averaged within fold, one fixed empirical archive per recipe/fold',
            'Original/count-shape model stagehalf conditioning differs from same-stage and stage-blind resampling',
            'GAN train-only bias/dispersion calibration versus uncalibrated empirical recipes prevents attributing marginal gains specifically to adversarial learning',
            'Empirical source donors cycle over a permutation: dependent stratified draws, not32 independent patients',
            'Resampling excludes each training target from its source pool and depth-qualified selection can bias cell states',
            'Recorded depth-unmatched fractions pool all37 targets, not held-out targets alone',
            'Only the original fixed empirical streams reconstructed; no new fits, tuning, sealed or external counts',
            'Original profile archives/scores retained; original cell-bag summaries unavailable for byte-identity verification',
            'Summary coverage does not establish patient probabilities, cell-state biology or learned patient features']}
    write(OUT/'comparison.json', value)
    print(json.dumps({key: value[key] for key in ('status', 'n_empirical_archives', 'historical_numeric_values_exact')}))


if __name__ == '__main__':
    main()
