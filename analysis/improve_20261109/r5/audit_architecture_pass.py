"""Read-only provenance audit of the fixed18-fit architecture screen.

Hashes files without deserializing counts or model weights. Completeness is
separate from numerical eligibility; this checker never changes a default.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RUNS = HERE.parents[2]/'runs/improve_20261109'
BASE = RUNS/'results_r5/architecture_pass'
FITS = RUNS/'fits_r5/architecture_pass'
MODES = RUNS/'smoke_r5/architecture_pass'
CACHES = RUNS/'caches/hvg2000'
ORIGINAL = RUNS/'fits_r3/pbulk/hvg2000'
ARMS = ('d.randproj,pbulk', 'g.split,pbulk', 'g.hier,pbulk')
EXPECTED = {(arm, fold, seed) for arm in ARMS for fold in range(3) for seed in (0, 1)}
TRAIN_SOURCES = ('trainer.py', 'arms.py', 'arm_d_randproj.py', 'arm_g_hier.py', 'arm_pbulk.py')
HYPERPARAMETERS = {'generator_lr': .012, 'critic_lr': .018, 'particle_lr': .06,
                   'reg_coeff': 1., 'reg_kappa': 1., 'prior_reg': .01}
SPECIFICATION = ORIGINAL/'fold0/seed0/config.json'
SPECIFICATION_SHA256 = 'bc943a36bb8e1f63c5eac67eeec466ffe9f18f3a1a3222376b559ecf00409e47'
COMMON_TRAINING = ('recipe', 'learning_rates', 'training_options', 'hyperparameters',
                   'recipe_total_steps', 'checkpoint_every', 'snapshot_updates',
                   'shadow_ema_decays', 'native_ema_decay', 'default_generation_state',
                   'n_genes', 'critic_features')
FOLDS = STUDY/'folds.json'
FOLDS_SHA256 = '53ae204b0730416383112702a3e09d0ac6a336249ad963484787b757015baa52'
EXTENDED_FOLDS_SHA256 = '6e5c587cc9f4e08dfa2c17e420da333c971edc8a3d961dfbdc1ad783433f765a'
CACHE_FILES = {'train/counts.npz', 'train/metadata.npz', 'heldout/counts.npz',
               'heldout/metadata.npz', 'space.npz', 'type_classifier.pt'}
CACHE_MANIFEST_SHA256 = {
    0: '65d5e2f405ceeeb14ac35f8660bc9c42272ac35c64aeda4f7c3bd9ee21dbd09c',
    1: 'fb856d7f69eb37655bc08e4dd511da5ccf243cf17a700edd392af288a579140b',
    2: 'b46b7d37e47d9ca796e129bbccf91fc3acd48ff9e900ca30c2f6d9c59ff9fb92'}
STATES = [f'snap_{step}' for step in range(15000, 20000, 1000)]+['final']
STREAMS = 'SeedSequence([20261109,2,inner_fold,development_donor_index]) -> global,latent,cell,count,library'
PROMOTION = STUDY/'default_config_distribution_evidence_20261113.json'
PROMOTION_SHA256 = '3fb5bcb23efd58c47b25946a00cf63cd051b40f9b86d1b9ad4f7ec62498dcbdc'


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


def sources():
    recorded = read(BASE/'source_manifest.json')['source_sha256']
    require(set(recorded) == set(TRAIN_SOURCES) | {'donor_encoder.py', 'interval_metrics.py'},
            'wrong architecture training source manifest')
    require(recorded == read(STUDY/'architecture_pass_evidence_20261113.json')[
        'training_source_manifest']['source_sha256'], 'training source anchors disagree')
    require(sha(PROMOTION) == PROMOTION_SHA256, 'frozen inference source anchor changed')
    inference = read(PROMOTION)['artifact_audit']['sources']
    require({'generate.py', 'mode_biascal.py', 'mode_stagedefault.py', 'architecture_score.py',
             'evaluate.py', 'metrics.py'} <= set(inference), 'inference source anchor incomplete')
    for name in set(recorded) & set(inference):
        require(recorded[name] == inference[name], f'source anchors disagree: {name}')
    merged = {**recorded, **inference}
    for name, expected in merged.items():
        require(sha(HERE/name) == expected, f'source changed: {name}')
    return merged


def specification():
    require(sha(SPECIFICATION) == SPECIFICATION_SHA256, 'frozen training specification changed')
    require(sha(FOLDS) == FOLDS_SHA256, 'frozen donor folds changed')
    for path in (HERE/'folds_r2.json', STUDY/'r3/folds_r2.json'):
        require(sha(path) == EXTENDED_FOLDS_SHA256, f'extended donor folds changed: {path}')
    return read(SPECIFICATION), read(FOLDS)['inner_folds']


def audit_cache(fold, partition):
    cache = CACHES/f'fold{fold}'
    manifest = read(cache/'manifest.json')
    train, held = manifest['train_donors'], manifest['heldout_donors']
    require(len(train)+len(held) == 37 and len(set(train)) == len(train)
            and len(set(held)) == len(held) and not set(train) & set(held),
            f'wrong or duplicate development partition: {cache}')
    require(manifest['inner_fold'] == partition['inner_fold'] == fold
            and train == partition['train_donors'] and held == partition['heldout_donors'],
            f'frozen donor partition mismatch: {cache}')
    require(manifest['panel'] == 'hvg2000' and manifest['n_genes'] == 2000
            and manifest['n_pcs'] == 50, f'cache space mismatch: {cache}')
    # The fixed allowlist also prevents following arbitrary manifest paths.
    require(set(manifest['files_SHA256']) == CACHE_FILES, f'cache payload set mismatch: {cache}')
    payloads = {name: sha(cache/name) for name in sorted(CACHE_FILES)}
    require(payloads == manifest['files_SHA256'], f'cache payload changed: {cache}')
    cache_sha = sha(cache/'manifest.json')
    require(cache_sha == CACHE_MANIFEST_SHA256[fold], f'frozen cache manifest changed: {cache}')
    return {'manifest_sha256': cache_sha, 'payload_sha256': payloads,
            'train_donors': train, 'heldout_donors': held}


def training(config, arm, fold, seed, cache_info, spec, label):
    folds_file = STUDY/'r3/folds_r2.json' if arm == 'pbulk' else HERE/'folds_r2.json'
    expected = {'arm': arm, 'flags': arm.split(','), 'fold': fold, 'seed': seed, 'panel': 'hvg2000',
                'updates': 20000, 'device': 'cuda', 'particlegan_version': '0.9.7',
                'protocol_split': True, 'cache_manifest_SHA256': cache_info['manifest_sha256'],
                'cache': str(CACHES/f'fold{fold}'), 'batch': [32, 128],
                'stream_root': 20261109, 'native_recipe': 'bispf_optimized',
                'folds_file': str(folds_file),
                'generator_kwargs': {} if arm in ('pbulk', 'd.randproj,pbulk') else {'zd': 8, 'zc': 8, 'h': 128}}
    require(all(config.get(k) == v for k, v in expected.items()), f'fit recipe mismatch: {label}')
    require(config['hyperparameters'] == HYPERPARAMETERS and
            all(config.get(k) == spec[k] for k in COMMON_TRAINING),
            f'resolved training specification mismatch: {label}')
    require(all(config[k] == cache_info[k] for k in ('train_donors', 'heldout_donors')),
            f'fit donor partition mismatch: {label}')


def generation(receipt, n_train, n_held, label):
    expected = {'mode': 'biascal@stagehalf', 'M': 32, 'K': 256, 'generation_device': 'cuda',
                'targets': 37, 'heldout_targets': n_held, 'states_used': STATES,
                'generation_streams': STREAMS}
    require(all(receipt.get(k) == v for k, v in expected.items()), f'generation recipe mismatch: {label}')
    bias = receipt['mode_extra']['biascal']
    require(bias['n_train_targets'] == n_train and sorted(bias['states']) == sorted(STATES)
            and bias['inner_mode'] == 'stagehalf', f'train-only calibration metadata mismatch: {label}')


def audit_fit(arm, fold, seed, source_hashes, cache_info, spec):
    fit = FITS/arm/f'hvg2000/fold{fold}/seed{seed}'
    score_path = BASE/arm/f'fold{fold}/seed{seed}.json'
    receipt_path = MODES/arm/f'hvg2000/fold{fold}/seed{seed}.receipt.json'
    temporary = receipt_path.with_suffix('').with_suffix('.npz')
    score, receipt = read(score_path), read(receipt_path)
    config, fit_receipt = read(fit/'config.json'), read(fit/'fit_receipt.json')
    cache_sha = cache_info['manifest_sha256']
    training(config, arm, fold, seed, cache_info, spec, str(fit))
    n_train, n_held = len(cache_info['train_donors']), len(cache_info['heldout_donors'])
    require(read(fit/'architecture_source.json') == {name: source_hashes[name] for name in TRAIN_SOURCES},
            f'per-fit training source mismatch: {fit}')
    require(all(fit_receipt.get(k) == v for k, v in {'status': 'completed', 'completed_updates': 20000,
                'device': 'cuda', 'threads': 3}.items()), f'incomplete or mismatched fit: {fit}')
    checkpoints = {'final.pt': sha(fit/'final.pt')}
    require(checkpoints['final.pt'] == fit_receipt['final_SHA256'], f'final checkpoint changed: {fit}')
    require(set(fit_receipt['snapshots_SHA256']) == {'5000', '10000', '15000', '16000', '17000', '18000', '19000'},
            f'snapshot set mismatch: {fit}')
    for step, expected_sha in fit_receipt['snapshots_SHA256'].items():
        name = f'snap_{step}.pt'
        checkpoints[name] = sha(fit/name)
        require(checkpoints[name] == expected_sha, f'snapshot changed: {fit/name}')
    generation(receipt, n_train, n_held, str(receipt_path))
    require(receipt['fit'] == str(fit), f'generation fit mismatch: {receipt_path}')
    require(Path(score['source']) == temporary, f'score source path mismatch: {score_path}')
    fp = score['fingerprint']
    require(score['fold'] == fold and score['M'] == 32 and fp['cache_sha256'] == cache_sha and
            fp['score_code_sha256'] == source_hashes['architecture_score.py'] and
            fp['interval_code_sha256'] == source_hashes['interval_metrics.py'], f'score provenance mismatch: {score_path}')
    require(fp['generated_sha256'] == score['pruned_source_sha256'] == receipt['generated_SHA256']
            and not temporary.exists(), f'generated/score/receipt hash or prune mismatch: {score_path}')
    native_sha = sha(fit/'generated.npz')
    require(native_sha == fit_receipt['generation']['generated_SHA256'], f'native summary changed: {fit}')
    return {'arm': arm, 'fold': fold, 'seed': seed, 'cache_manifest_sha256': cache_sha,
            'config_sha256': sha(fit/'config.json'), 'fit_receipt_sha256': sha(fit/'fit_receipt.json'),
            'mode_receipt_sha256': sha(receipt_path), 'score_sha256': sha(score_path),
            'checkpoints': checkpoints, 'native_summary_sha256': native_sha,
            'pruned_mode_summary_sha256': fp['generated_sha256']}


def audit():
    source_hashes = sources()
    spec, partitions = specification()
    caches = {fold: audit_cache(fold, partitions[fold]) for fold in range(3)}
    artifacts, missing, baselines = [], [], {}
    for arm, fold, seed in sorted(EXPECTED):
        score_path = BASE/arm/f'fold{fold}/seed{seed}.json'
        temporary = MODES/arm/f'hvg2000/fold{fold}/seed{seed}.npz'
        if not score_path.exists() or not read(score_path).get('pruned_source_sha256') or temporary.exists():
            missing.append({'arm': arm, 'fold': fold, 'seed': seed, 'reason': 'score not finalized'})
            continue
        artifacts.append(audit_fit(arm, fold, seed, source_hashes, caches[fold], spec))
        key = f'fold{fold}/seed{seed}'
        if key not in baselines:
            baseline_path = BASE/'baseline32'/f'fold{fold}/seed{seed}.json'
            archive = ORIGINAL/f'fold{fold}/seed{seed}/generated_biascal@stagehalf.npz'
            receipt_path = archive.with_suffix('.receipt.json')
            row, receipt = read(baseline_path), read(receipt_path)
            config_path = archive.parent/'config.json'
            config = read(config_path)
            training(config, 'pbulk', fold, seed, caches[fold], spec, str(config_path))
            generation(receipt, len(caches[fold]['train_donors']), len(caches[fold]['heldout_donors']), str(receipt_path))
            require(receipt['fit'] == str(archive.parent), f'original generation fit mismatch: {receipt_path}')
            require(Path(row['source']) == archive, f'original score source path mismatch: {baseline_path}')
            fp = row['fingerprint']
            require(row['fold'] == fold and row['M'] == 32 and fp['generated_sha256'] == sha(archive)
                    == receipt['generated_SHA256'] and fp['cache_sha256'] == artifacts[-1]['cache_manifest_sha256']
                    and fp['score_code_sha256'] == source_hashes['architecture_score.py'] and
                    fp['interval_code_sha256'] == source_hashes['interval_metrics.py'], f'original reference mismatch: {baseline_path}')
            baselines[key] = {'score_sha256': sha(baseline_path), 'receipt_sha256': sha(receipt_path),
                              'config_sha256': sha(config_path), 'generated_sha256': fp['generated_sha256']}
    return {'status': 'passed' if not missing else 'incomplete', 'complete': not missing,
            'verified_fits': len(artifacts), 'planned_fits': 18, 'missing': missing,
            'sources': source_hashes, 'source_anchor_promotion_sha256': PROMOTION_SHA256,
            'training_specification_sha256': SPECIFICATION_SHA256, 'fold_definitions_sha256': FOLDS_SHA256,
            'extended_fold_definitions_sha256': EXTENDED_FOLDS_SHA256, 'verified_caches': caches,
            'artifacts': artifacts, 'matched_original_references': baselines, 'auditor_sha256': sha(__file__),
            'scope': 'Artifact/metadata/source verification only; numerical gates remain separate; pruned arrays not rescored'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args(argv)
    value = audit()
    path = BASE/('screen_artifact_audit.json' if value['complete'] else 'screen_artifact_audit_partial.json')
    path.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k: value[k] for k in ('status', 'verified_fits', 'planned_fits')}, indent=2))
    if not value['complete'] and not args.allow_incomplete:
        raise SystemExit('all18 finalized architecture fits required for a complete screen audit')


if __name__ == '__main__':
    main()
