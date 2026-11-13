"""Check completeness and provenance failures without counts or weight loads."""
import copy
import json

import pytest

import audit_architecture_pass as A


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    recorded = A.read(A.STUDY/'architecture_pass_evidence_20261113.json')['training_source_manifest']
    for name in ('BASE', 'FITS', 'MODES', 'CACHES', 'ORIGINAL'):
        monkeypatch.setattr(A, name, tmp_path/name.lower())
    write(A.BASE/'source_manifest.json', recorded)
    hashes = A.sources()
    spec, partitions = A.specification()
    monkeypatch.setattr(A, 'CACHE_MANIFEST_SHA256', {})
    for fold in range(3):
        cache = A.CACHES/f'fold{fold}'
        payloads = {}
        for name in A.CACHE_FILES:
            path = cache/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'not a count or weight archive')
            payloads[name] = A.sha(path)
        manifest = {**partitions[fold], 'panel': 'hvg2000', 'n_genes': 2000, 'n_pcs': 50,
                    'files_SHA256': payloads}
        write(cache/'manifest.json', manifest)
        A.CACHE_MANIFEST_SHA256[fold] = A.sha(cache/'manifest.json')

    def add(arm='g.split,pbulk', fold=0, seed=0):
        cache = A.CACHES/f'fold{fold}'
        manifest = A.read(cache/'manifest.json')
        cache_sha = A.sha(cache/'manifest.json')
        fit = A.FITS/arm/f'hvg2000/fold{fold}/seed{seed}'
        fit.mkdir(parents=True, exist_ok=True)
        config = {**copy.deepcopy({k: spec[k] for k in A.COMMON_TRAINING}),
                  'arm': arm, 'flags': arm.split(','), 'fold': fold, 'seed': seed, 'panel': 'hvg2000',
                  'updates': 20000, 'device': 'cuda', 'particlegan_version': '0.9.7', 'protocol_split': True,
                  'cache_manifest_SHA256': cache_sha, 'cache': str(cache),
                  'batch': [32, 128], 'stream_root': 20261109,
                  'folds_file': str(A.HERE/'folds_r2.json'),
                  'generator_kwargs': {} if arm == 'd.randproj,pbulk' else {'zd': 8, 'zc': 8, 'h': 128},
                  'native_recipe': 'bispf_optimized',
                  'train_donors': manifest['train_donors'], 'heldout_donors': manifest['heldout_donors'],
                  }
        write(fit/'config.json', config)
        write(fit/'architecture_source.json', {name: hashes[name] for name in A.TRAIN_SOURCES})
        # These deliberately are not pickle or NPZ payloads. A successful
        # audit demonstrates file hashing without model/count deserialization.
        (fit/'final.pt').write_bytes(b'not a serialized model')
        snapshots = {}
        for step in (5000, 10000, 15000, 16000, 17000, 18000, 19000):
            path = fit/f'snap_{step}.pt'
            path.write_text(f'not a model: step{step}')
            snapshots[str(step)] = A.sha(path)
        (fit/'generated.npz').write_bytes(b'not a count archive')
        write(fit/'fit_receipt.json', {'status': 'completed', 'completed_updates': 20000,
            'device': 'cuda', 'threads': 3, 'final_SHA256': A.sha(fit/'final.pt'),
            'snapshots_SHA256': snapshots, 'generation': {'generated_SHA256': A.sha(fit/'generated.npz')}})
        temporary = A.MODES/arm/f'hvg2000/fold{fold}/seed{seed}.npz'
        mode_path = temporary.with_suffix('.receipt.json')
        receipt = {'mode': 'biascal@stagehalf', 'M': 32, 'K': 256, 'generation_device': 'cuda',
                   'targets': 37, 'heldout_targets': len(manifest['heldout_donors']), 'states_used': A.STATES,
                   'generation_streams': A.STREAMS, 'fit': str(fit), 'generated_SHA256': 'pruned-summary',
                   'mode_extra': {'biascal': {'n_train_targets': len(manifest['train_donors']),
                                            'states': A.STATES, 'inner_mode': 'stagehalf'}}}
        write(mode_path, receipt)
        score_path = A.BASE/arm/f'fold{fold}/seed{seed}.json'
        fingerprint = {'generated_sha256': receipt['generated_SHA256'], 'cache_sha256': cache_sha,
                       'score_code_sha256': hashes['architecture_score.py'],
                       'interval_code_sha256': hashes['interval_metrics.py']}
        write(score_path, {'fold': fold, 'M': 32, 'source': str(temporary), 'fingerprint': fingerprint,
                           'pruned_source_sha256': receipt['generated_SHA256']})
        original = A.ORIGINAL/f'fold{fold}/seed{seed}'
        original.mkdir(parents=True, exist_ok=True)
        archive = original/'generated_biascal@stagehalf.npz'
        archive.write_bytes(b'not an empirical count archive')
        baseline_receipt = {**receipt, 'fit': str(original), 'generated_SHA256': A.sha(archive)}
        baseline_path = archive.with_suffix('.receipt.json')
        write(baseline_path, baseline_receipt)
        write(original/'config.json', {**config, 'arm': 'pbulk', 'flags': ['pbulk'], 'generator_kwargs': {},
                                      'folds_file': str(A.STUDY/'r3/folds_r2.json')})
        baseline_score = A.BASE/'baseline32'/f'fold{fold}/seed{seed}.json'
        write(baseline_score, {'fold': fold, 'M': 32, 'source': str(archive),
            'fingerprint': {**fingerprint, 'generated_sha256': A.sha(archive)}})
        return {'fit': fit, 'score': score_path, 'mode': mode_path, 'temporary': temporary,
                'baseline_mode': baseline_path, 'baseline_score': baseline_score,
                'baseline_config': original/'config.json', 'cache': cache}

    return add


def test_partial_screen_cannot_report_a_complete_audit(sandbox):
    sandbox()
    result = A.audit()
    assert result['status'] == 'incomplete' and not result['complete']
    assert result['verified_fits'] == 1 and len(result['missing']) == 17
    with pytest.raises(SystemExit, match='all18'):
        A.main([])
    assert A.read(A.BASE/'screen_artifact_audit_partial.json')['status'] == 'incomplete'
    assert not (A.BASE/'screen_artifact_audit.json').exists()


def test_all_predeclared_fits_are_required_and_hashing_never_deserializes(sandbox):
    for key in A.EXPECTED:
        sandbox(*key)
    result = A.audit()
    assert result['status'] == 'passed' and result['complete']
    assert result['verified_fits'] == 18 and not result['missing']
    assert len(result['matched_original_references']) == 6


def test_snapshot_corruption_cannot_pass(sandbox):
    paths = sandbox()
    (paths['fit']/'snap_15000.pt').write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='snapshot changed'):
        A.audit()


@pytest.mark.parametrize('where', ('candidate', 'original_reference'))
def test_incompatible_generation_size_cannot_pass(sandbox, where):
    paths = sandbox()
    path = paths['mode'] if where == 'candidate' else paths['baseline_mode']
    receipt = A.read(path)
    receipt['K'] = 64
    write(path, receipt)
    with pytest.raises(ValueError, match='generation recipe mismatch'):
        A.audit()


def test_calibration_claiming_heldout_targets_cannot_pass(sandbox):
    paths = sandbox()
    receipt = A.read(paths['mode'])
    receipt['mode_extra']['biascal']['n_train_targets'] = 37
    write(paths['mode'], receipt)
    with pytest.raises(ValueError, match='train-only calibration metadata'):
        A.audit()


@pytest.mark.parametrize('where', ('candidate', 'original_reference'))
@pytest.mark.parametrize('field', ('lr', 'regularization', 'training_options', 'learning_rates'))
def test_changed_resolved_training_settings_cannot_pass(sandbox, where, field):
    paths = sandbox()
    path = paths['fit']/'config.json' if where == 'candidate' else paths['baseline_config']
    config = A.read(path)
    if field == 'lr':
        config['recipe']['lr'] *= 2
        config['learning_rates']['generator'] *= 2
    elif field == 'regularization':
        config['recipe']['reg_coeff'] *= 2
    elif field == 'training_options':
        config['training_options']['dlr'] *= 2
    else:
        config['learning_rates']['schedule']['lr_floor'] /= 2
    write(path, config)
    with pytest.raises(ValueError, match='resolved training specification mismatch'):
        A.audit()


def test_different_training_source_cannot_pass(sandbox):
    paths = sandbox()
    path = paths['fit']/'architecture_source.json'
    recorded = A.read(path)
    recorded['arm_g_hier.py'] = 'different code'
    write(path, recorded)
    with pytest.raises(ValueError, match='per-fit training source mismatch'):
        A.audit()


def test_score_receipt_hash_disagreement_cannot_pass(sandbox):
    paths = sandbox()
    score = A.read(paths['score'])
    score['pruned_source_sha256'] = 'different generated summary'
    write(paths['score'], score)
    with pytest.raises(ValueError, match='generated/score/receipt hash'):
        A.audit()


def test_parent_finalization_window_stays_incomplete(sandbox):
    paths = sandbox()
    paths['temporary'].write_bytes(b'parent has not pruned this mode yet')
    result = A.audit()
    assert result['verified_fits'] == 0 and not result['complete']
    assert len(result['missing']) == 18


@pytest.mark.parametrize('name', sorted(A.CACHE_FILES))
def test_cache_payload_corruption_cannot_pass_with_unchanged_manifest(sandbox, name):
    paths = sandbox()
    (paths['cache']/name).write_bytes(b'changed cache bytes')
    with pytest.raises(ValueError, match='cache payload changed'):
        A.audit()


@pytest.mark.parametrize('change', ('duplicate', 'substitute', 'wrong_fold', 'extra_payload'))
def test_cache_partition_and_payload_scope_are_fixed(sandbox, change):
    paths = sandbox()
    path = paths['cache']/'manifest.json'
    manifest = A.read(path)
    if change == 'duplicate':
        manifest['train_donors'][0] = manifest['train_donors'][1]
        expected = 'duplicate development partition'
    elif change == 'substitute':
        manifest['train_donors'][0] = 'a donor outside the frozen development list'
        expected = 'frozen donor partition mismatch'
    elif change == 'wrong_fold':
        manifest['inner_fold'] = 2
        expected = 'frozen donor partition mismatch'
    else:
        manifest['files_SHA256']['../../unexpected.npz'] = 'irrelevant'
        expected = 'cache payload set mismatch'
    write(path, manifest)
    with pytest.raises(ValueError, match=expected):
        A.audit()


@pytest.mark.parametrize('field', ('fit', 'source'))
def test_original_reference_artifact_links_are_fixed(sandbox, field):
    paths = sandbox()
    path = paths['baseline_mode'] if field == 'fit' else paths['baseline_score']
    value = A.read(path)
    value[field] = '/incorrect/original/reference'
    write(path, value)
    with pytest.raises(ValueError, match='original (generation fit|score source path) mismatch'):
        A.audit()


@pytest.mark.parametrize('anchor', ('SPECIFICATION', 'FOLDS'))
def test_frozen_specification_anchors_cannot_change(sandbox, tmp_path, monkeypatch, anchor):
    sandbox()
    replacement = tmp_path/'changed-anchor.json'
    replacement.write_text('{}')
    monkeypatch.setattr(A, anchor, replacement)
    with pytest.raises(ValueError, match='frozen (training specification|donor folds) changed'):
        A.audit()
