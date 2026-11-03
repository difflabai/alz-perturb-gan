"""Verify a settled candidate, retain its evidence, and keep or revert its commit."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path('/ml2/alz')
sys.path.insert(0, str(ROOT))
from selection import decide


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    deployment = read(ROOT / 'artifacts/allocations' / args.attempt / 'deployment.json')
    assert not deployment.get('qualification_only'), 'Qualification jobs cannot select or revert a candidate'
    out, ledger = Path(deployment['output_dir']), Path(deployment['ledger_path'])
    settlement = read(ledger.with_name(args.attempt + '-settlement.json'))
    assert settlement['uncertain'] is None
    assert settlement['settlement']['status'] == 'settled'
    job = read(ledger)['jobs'][0]
    assert job['reservation_released'], 'Do not edit or revert source before proven cleanup'
    assert not git('status', '--porcelain'), 'Commit or reconcile other work before recording a candidate decision'
    if job['status'] != 'completed' or job['exit_code'] != 0:
        failure = {'attempt': args.attempt, 'candidate_commit': deployment['candidate_commit'],
                   'decision': {'accept': False, 'reasons': ['native attempt failed or fixed endpoint incomplete']},
                   'guard_status': job['status'], 'exit_code': job['exit_code'],
                   'charged_GPU_seconds': settlement['settlement']['charged_GPU_seconds'],
                   'previous_incumbent_preserved': True,
                   'worker_failure': read(out / 'failure.json') if (out / 'failure.json').exists() else None,
                   'fit_failure': read(out / 'fit/failure.json') if (out / 'fit/failure.json').exists() else None,
                   'checkpoint': read(out / 'fit/checkpoint_receipt.json') if (out / 'fit/checkpoint_receipt.json').exists() else None}
        result_path = ROOT / 'analysis' / (args.attempt + '-result.json')
        assert not result_path.exists()
        result_path.write_text(json.dumps(failure, indent=2) + '\n')
        subprocess.run(['git', 'add', str(result_path)], cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '-m', 'Retain failed ' + args.attempt + ' before reverting candidate'], cwd=ROOT, check=True)
        if not args.validation:
            subprocess.run(['git', 'revert', '--no-edit', deployment['candidate_commit']], cwd=ROOT, check=True)
        print(json.dumps(failure, indent=2))
        return
    candidate = read(out / 'completed.json')
    checkpoint = Path(candidate['checkpoint_path'])
    receipt = read(checkpoint.parent / 'checkpoint_receipt.json')
    assert receipt['status'] == 'completed' and receipt['completed_updates'] == 20000
    assert checkpoint.name == receipt['path']
    assert sha(checkpoint) == receipt['SHA256'] == candidate['checkpoint_SHA256']
    assert candidate['variant_config'] == deployment['variant_config']
    if deployment.get('hybrid_config') is not None:
        assert candidate['hybrid_config'] == deployment['hybrid_config']
    incumbent = json.loads(git('show', deployment['candidate_commit'] + ':incumbent.json'))
    assert read(ROOT / 'incumbent.json') == incumbent, 'Incumbent changed concurrently; reconcile rather than select against a moving target'
    targets = np.load(ROOT / 'data/calibration_fold0/real_summaries.npz', allow_pickle=False)
    real_cells = np.load(ROOT / 'data/calibration_fold0/cell_reservoir.npz', allow_pickle=False)
    records = []
    for rep in [0, 1]:
        folder = out / f'r{rep}'
        q = read(folder / 'metrics.json')
        assert sha(folder / 'arrays.npz') == q['arrays_SHA256']
        arrays = np.load(folder / 'arrays.npz', allow_pickle=False)
        y, cells = arrays['stage_synthetic_summaries'], arrays['generated_cells_by_stage']
        assert y.shape == (4, 200, 26) and cells.shape == (4, 375, 50)
        scores = np.asarray([np.linalg.norm(y[s] - real, axis=1).mean()
                             - .5 * cdist(y[s], y[s]).sum() / (200 * 199)
                             for s, real in zip(targets['stage'], targets['summary'])])
        np.testing.assert_allclose(scores, q['per_donor_energy_score'], rtol=0, atol=1e-10)
        x, z = cells.reshape(-1, 50), real_cells['equal_stage']
        ed = 2 * cdist(x, z).mean() - cdist(x, x).mean() - cdist(z, z).mean()
        np.testing.assert_allclose(ed, q['cell_energy_distance'], rtol=0, atol=1e-10)
        variance = targets['summary'].var(0, ddof=1).sum()
        spread = np.sqrt(y.reshape(-1, 26).var(0, ddof=1).sum() / variance)
        np.testing.assert_allclose(spread, q['spread_ratio'], rtol=0, atol=1e-10)
        assert q['donors'] == targets['donors'].tolist() == candidate['donors']
        for stage in range(4):
            exemplar = np.load(folder / f'stage{stage}_exemplar.npz', allow_pickle=False)['counts']
            assert exemplar.shape == (1, 400, 2000)
            assert np.isfinite(exemplar).all() and (exemplar >= 0).all()
            assert np.equal(exemplar, np.floor(exemplar)).all()
        records.append(q)
    for key in ['energy_score', 'cell_energy_distance', 'spread_ratio', 'within_condition_spread_ratio', 'coverage80', 'generated_zero_count_rate']:
        np.testing.assert_allclose(candidate[key], np.mean([q[key] for q in records]), rtol=0, atol=1e-10)
    if args.validation:
        def architecture(value):
            return {'z_dim': value['z_dim'], 'stage_balance': value.get('stage_balance', False),
                    'cell_z_dim': value.get('cell_z_dim', 8)}
        assert architecture(candidate['variant_config']) == architecture(incumbent['variant_config'])
        assert candidate.get('hybrid_config') == incumbent.get('hybrid_config')
        assert candidate.get('method', 'pg') == incumbent.get('method', 'pg')
        seed = candidate['training_seed']
        assert seed in [1, 2], 'Confirmation seeds are fixed prospectively'
        baseline = read(ROOT / 'analysis' / f'baseline_calibration_seed{seed}.json')
        for key in ['donors', 'training_seed', 'generation_replicates', 'calibration_manifest_SHA256', 'trace_manifest_SHA256', 'sampling_contract']:
            assert candidate[key] == baseline[key], 'Unmatched confirmation baseline: ' + key
        result = {'status': 'SEED_CONFIRMATION_RECORDED_NO_MODEL_SELECTION',
                  'attempt': args.attempt, 'candidate': candidate, 'baseline': baseline,
                  'ES_relative_improvement': 1 - candidate['energy_score'] / baseline['energy_score'],
                  'cell_ED_difference': candidate['cell_energy_distance'] - baseline['cell_energy_distance'],
                  'joint_improvement': candidate['energy_score'] < baseline['energy_score']
                                       and candidate['cell_energy_distance'] <= baseline['cell_energy_distance']
                                       and .7 <= candidate['spread_ratio'] <= 1.3,
                  'incumbent_unchanged': True, 'training_seed_was_not_ranked': True,
                  'independently_verified_saved_scores_counts_and_checkpoint': True,
                  'charged_GPU_seconds': settlement['settlement']['charged_GPU_seconds']}
        path = ROOT / 'analysis' / (args.attempt + '-confirmation.json')
        assert not path.exists()
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        subprocess.run(['git', 'add', str(path)], cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '-m', 'Record matched seed confirmation ' + args.attempt + ' without selecting seeds'], cwd=ROOT, check=True)
        print(json.dumps({key: result[key] for key in ['status', 'attempt', 'ES_relative_improvement', 'cell_ED_difference', 'joint_improvement']}, indent=2))
        return
    decision = decide(candidate, incumbent)
    baseline = read(ROOT / 'baseline_calibration.json')
    result = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'attempt': args.attempt, 'candidate_commit': deployment['candidate_commit'],
        'candidate': candidate, 'previous_incumbent': incumbent, 'decision': decision,
        'ES_change': candidate['energy_score'] - incumbent['energy_score'],
        'cell_ED_change': candidate['cell_energy_distance'] - incumbent['cell_energy_distance'],
        'ES_relative_improvement_vs_resampling': 1 - candidate['energy_score'] / baseline['energy_score'],
        'cell_ED_difference_vs_resampling': candidate['cell_energy_distance'] - baseline['cell_energy_distance'],
        'charged_GPU_seconds': settlement['settlement']['charged_GPU_seconds'],
        'independent_saved_scores_checkpoint_and_count_exemplars_verified': True,
        'scope': 'Four adaptive calibration donors, one training seed; no evaluation metric was used for selection.',
    }
    result_path = ROOT / 'analysis' / (args.attempt + '-result.json')
    assert not result_path.exists(), 'Decision already retained; inspect instead of duplicating'
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if decision['accept']:
        candidate['candidate_commit'] = deployment['candidate_commit']
        (ROOT / 'incumbent.json').write_text(json.dumps(candidate, indent=2) + '\n')
        subprocess.run(['git', 'add', str(result_path), 'incumbent.json'], cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '-m', 'Keep ' + args.attempt + ': improved calibration ES and distribution match'], cwd=ROOT, check=True)
    else:
        subprocess.run(['git', 'add', str(result_path)], cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '-m', 'Record rejected ' + args.attempt + ' with verified calibration evidence'], cwd=ROOT, check=True)
        subprocess.run(['git', 'revert', '--no-edit', deployment['candidate_commit']], cwd=ROOT, check=True)
    print(json.dumps({k: result[k] for k in ['attempt', 'decision', 'ES_change', 'cell_ED_change', 'ES_relative_improvement_vs_resampling', 'charged_GPU_seconds']}, indent=2))


if __name__ == '__main__':
    main()
