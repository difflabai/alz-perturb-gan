"""Verify artifacts behind the fixed count-shape promotion comparison.

Read-only checks: recipe/device/sample sizes, train-only calibration metadata,
frozen fit/checkpoint hashes and unchanged scoring/generation source. A partial
audit cannot report passed. No counts or model weights are deserialized.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[2]/'runs/improve_20261109'
BASE = RUNS/'results_r5/architecture_pass/shape_confirmation'
MODES = RUNS/'smoke_r5/architecture_pass/shape_confirmation'
RECIPES = {'default': 'biascal@stagehalf', 'count_shape': 'cellshape@0.5:biascal:stagehalf'}
STATES = [f'snap_{step}' for step in range(15000, 20000, 1000)]+['final']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit():
    source_manifest = read(BASE/'source_manifest.json')
    for name, expected in source_manifest.items():
        require(sha(HERE/name) == expected, f'source changed: {name}')
    fits, artifacts, missing = {}, [], []
    for fold in range(3, 9):
        cache = RUNS/f'caches/hvg2000/fold{fold}'
        cache_sha = sha(cache/'manifest.json')
        manifest = read(cache/'manifest.json')
        for seed in (0, 1):
            paths = {label: BASE/label/f'fold{fold}/seed{seed}.json' for label in RECIPES}
            if not all(path.exists() for path in paths.values()):
                missing.append({'fold': fold, 'seed': seed})
                continue
            fit = RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{seed}'
            config, fit_receipt = read(fit/'config.json'), read(fit/'fit_receipt.json')
            require(all(config.get(k) == v for k, v in {'arm': 'pbulk', 'flags': ['pbulk'],
                'panel': 'hvg2000', 'fold': fold, 'seed': seed, 'updates': 20000,
                'particlegan_version': '0.9.7', 'protocol_split': True,
                'cache_manifest_SHA256': cache_sha}.items()), f'fit recipe mismatch: {fit}')
            require(fit_receipt['status'] == 'completed' and fit_receipt['completed_updates'] == 20000,
                    f'incomplete fit: {fit}')
            require(config['train_donors'] == manifest['train_donors'] and
                    config['heldout_donors'] == manifest['heldout_donors'], f'fit donor mismatch: {fit}')
            hashes = {}
            for state in STATES:
                name = 'final.pt' if state == 'final' else state+'.pt'
                actual = sha(fit/name)
                expected = (fit_receipt['final_SHA256'] if state == 'final' else
                            fit_receipt['snapshots_SHA256'][state.removeprefix('snap_')])
                require(actual == expected, f'checkpoint changed: {fit/name}')
                hashes[name] = actual
            fits[f'fold{fold}/seed{seed}'] = {'config_sha256': sha(fit/'config.json'),
                'fit_receipt_sha256': sha(fit/'fit_receipt.json'), 'checkpoints': hashes}
            n_train, n_held = len(manifest['train_donors']), len(manifest['heldout_donors'])
            require(n_train+n_held == 37, f'wrong development target count: {cache}')
            for label, recipe in RECIPES.items():
                score_path = paths[label]
                receipt_path = MODES/label/f'fold{fold}/seed{seed}.receipt.json'
                score, receipt = read(score_path), read(receipt_path)
                require(score['fold'] == fold and score['M'] == 32 and
                        score['generation_device'] == 'cpu' and score['generation_recipe'] == recipe,
                        f'score recipe mismatch: {score_path}')
                fingerprint = score['fingerprint']
                require(fingerprint['cache_sha256'] == cache_sha and
                        fingerprint['score_code_sha256'] == source_manifest['architecture_score.py'] and
                        fingerprint['interval_code_sha256'] == source_manifest['interval_metrics.py'],
                        f'score source mismatch: {score_path}')
                require(all(receipt.get(k) == v for k, v in {'mode': recipe, 'M': 32, 'K': 256,
                    'generation_device': 'cpu', 'targets': 37, 'heldout_targets': n_held,
                    'states_used': STATES, 'fit': str(fit)}.items()), f'generation mismatch: {receipt_path}')
                require(fingerprint['generated_sha256'] == score['pruned_source_sha256'] == receipt['generated_SHA256'],
                        f'generated hash mismatch: {receipt_path}')
                bias = receipt['mode_extra']['biascal']
                require(bias['n_train_targets'] == n_train and sorted(bias['states']) == sorted(STATES),
                        f'bias calibration metadata mismatch: {receipt_path}')
                if label == 'count_shape':
                    extra = receipt['mode_extra']
                    require(extra['cellshape']['alpha'] == .5 and extra['zcal']['n_train_targets'] == n_train
                            and sorted(extra['zcal']['states']) == sorted(STATES) and extra['zcal']['joint_biascal'],
                            f'count-shape calibration metadata mismatch: {receipt_path}')
                artifacts.append({'label': label, 'fold': fold, 'seed': seed,
                    'score_sha256': sha(score_path), 'receipt_sha256': sha(receipt_path),
                    'generated_sha256': fingerprint['generated_sha256']})
    return {'status': 'passed' if not missing else 'incomplete', 'complete': not missing,
            'verified_pairs': len(fits), 'missing_pairs': missing, 'sources': source_manifest,
            'fits': fits, 'artifacts': artifacts, 'auditor_sha256': sha(__file__),
            'scope': 'artifact/source verification; numerical promotion gates are evaluated separately'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-incomplete', action='store_true')
    args = p.parse_args()
    value = audit()
    destination = BASE/('audit_partial.json' if not value['complete'] else 'audit.json')
    destination.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({key: value[key] for key in ('status', 'verified_pairs', 'missing_pairs')}, indent=2))
    if not value['complete'] and not args.allow_incomplete:
        raise SystemExit('full12-pair artifact audit required for promotion')
