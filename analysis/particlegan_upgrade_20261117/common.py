"""Pins and lightweight resource limits for the isolated two-lane pilot."""
import hashlib
import json
from pathlib import Path
import host

DEFAULT = host.ROOT / 'analysis/improve_20261109/default_config.json'
DEFAULT_SHA = 'fecbfce1e92d31a5d8c3d646f0a6dfb036c975b599dbdc880c332c1023bb3c23'
PIN = host.HERE / 'source_manifest.json'
NAMESPACE = [20261114, 6]
MODE = 'cellshape@0.5:biascal:stagehalf'


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sources():
    predecessor = host.ROOT / 'analysis/covariance_challenge_20261114/gpu_verification_source_manifest_prelaunch.json'
    assert host.sha(predecessor) == '750321a1a53ecc7875d4c0e94f5b9abb54d15a462b473803f6153a8810d9059f'
    old = json.loads(predecessor.read_text())
    for path, digest in old.items():
        assert host.sha(path) == digest, ('frozen dependency changed', path)
    paths = [*host.HERE.glob('*.py'), host.HERE/'PROTOCOL.txt', DEFAULT,
             host.ROOT/'scripts/microglia.py', predecessor,
             host.RELEASE/'dist/particlegan-0.10.0-py3-none-any.whl',
             host.HERE/'release_receipt.json',
             host.HERE/'resolved_recipes.json',
             host.ROOT/'analysis/covariance_challenge_20261114/screen_comparison_complete_20261116.json',
             *[host.ROOT/f'runs/improve_20261109/results_r5/covariance_challenge_20261114/{n}/fold0/seed0.json'
               for n in ('current32','original32','empirical32')],
             *sorted((host.RELEASE/'site-packages/particlegan').rglob('*.py'))]
    result = dict(old)
    result.update({str(p.resolve()):host.sha(p) for p in paths})
    assert host.sha(DEFAULT) == DEFAULT_SHA
    return result


def check_sources():
    current = sources()
    assert current == json.loads(PIN.read_text()), 'isolated pilot source manifest differs'
    return current


def preflight():
    sources=check_sources()
    record=json.loads((host.HERE/'prelaunch.json').read_text())
    assert record['status']=='passed_fixed_release_pilot_preflight'
    assert record['source_manifest_sha256']==host.sha(PIN)
    root=host.RUNS/'verification'
    names=('verification_0.9.7','verification_0.10.0','verification_0.10.0_rhbi',
           'fresh_process_compat','fresh_process_rhbi','compatibility')
    assert set(record['proof_sha256'])=={str(root/(n+'.json')) for n in names}
    for path,digest in record['proof_sha256'].items():
        assert host.sha(path)==digest, ('changed preflight proof',path)
    proofs={n:json.loads((root/(n+'.json')).read_text()) for n in names}
    for name in names[:-1]:
        proof=proofs[name]
        assert proof['verification_code_sha256']==host.sha(host.HERE/'verify.py')
        if name.startswith('fresh_process'):
            assert proof['status']=='passed_fresh_process_exact_200_to_300'
            assert proof['full_checkpoint_and_metrics_exact'] is True
        else:
            assert proof['status']=='passed'
            assert proof['rejected_checkpoint_mismatches']==['particlegan_version','native_recipe']
            assert proof['fresh_five_step_sha256']==host.sha(root/('fresh_'+proof['version']+('_rhbi' if name.endswith('_rhbi') else '')+'.pt'))
        if name!='verification_0.9.7':
            assert proof['provenance']['host_derivation_sha256']==host.sha(host.HERE/'host.py')
            for item in proof['provenance']['files'].values():
                assert host.sha(item['original'])==item['original_sha256']
                assert host.sha(item['derived'])==item['derived_sha256']
            assert proof['full_checkpoint_and_metrics_exact'] is True
        if name.endswith('_rhbi'):
            semantics=proof['rhbi_semantics']
            assert semantics['conditional_transport_calls']==(100 if name.startswith('fresh_process') else 300)
            assert semantics['nonzero_values']>0 and semantics['shared_record'] is True
    compatibility=proofs['compatibility']
    assert compatibility['status']=='passed_exact_initial_and_five_updates_on_shared_GPU1'
    assert compatibility['comparison_code_sha256']==host.sha(host.HERE/'compare_compat.py')
    assert compatibility['additive_metadata_fields']==[]
    for name,digest in compatibility['fresh_checkpoints_sha256'].items():
        assert host.sha(root/name)==digest
    return sources


def cache_anchor(held=False):
    path=host.ROOT/'analysis/empirical_challenge_20261114/cache_and_fit_anchors.json'
    anchor=json.loads(path.read_text())['caches']['0']
    cache=host.ROOT/'runs/improve_20261109/caches/hvg2000/fold0'
    assert host.sha(cache/'manifest.json')==anchor['manifest_sha256']
    payloads=anchor['payload_sha256'] if held else {
        n:anchor['payload_sha256'][n] for n in ('space.npz','train/counts.npz','train/metadata.npz')}
    for name,digest in payloads.items():
        assert host.sha(cache/name)==digest, ('cache payload changed',name)
    return {'manifest_sha256':anchor['manifest_sha256'],'payload_sha256':payloads}


def fit_path(lane):
    return host.RUNS / f'fits/{lane}/pbulk/hvg2000/fold0/seed0'


def valid_fit(lane):
    fit=fit_path(lane); path=fit/'release_test_receipt.json'
    if not path.exists():return False
    record=json.loads(path.read_text())
    assert record['status']=='completed_native_TRAIN_only' and record['lane']==lane
    assert record['source_manifest_sha256']==host.sha(PIN) and record['default_changed'] is False
    assert record['provenance']['host_derivation_sha256']==host.sha(host.HERE/'host.py')
    names=('config.json','fit_receipt.json','checkpoint_receipt.json','learning_curve.jsonl')
    assert set(record['metadata_sha256'])==set(names)
    for name,digest in record['metadata_sha256'].items():assert host.sha(fit/name)==digest
    expected={'final.pt',*(f'snap_{u}.pt' for u in (5000,10000,15000,16000,17000,18000,19000))}
    assert set(record['weights_sha256'])==expected
    for name,digest in record['weights_sha256'].items():assert host.sha(fit/name)==digest
    cfg=json.loads((fit/'config.json').read_text())
    specs=json.loads((host.HERE/'resolved_recipes.json').read_text())
    assert cfg['recipe']==specs[lane]['recipe'] and cfg['particlegan_version']=='0.10.0'
    assert cfg['native_recipe']==specs[lane]['native_recipe']
    assert cfg['updates']==20000 and cfg['flags']==['pbulk'] and cfg['fold']==cfg['seed']==0
    assert cfg['cache_manifest_SHA256']==record['cache_manifest_sha256']==record['cache_anchor']['manifest_sha256']
    receipt=json.loads((fit/'fit_receipt.json').read_text())
    assert receipt['status']=='training_completed' and receipt['completed_updates']==receipt['endpoint']==20000
    assert receipt['final_SHA256']==record['weights_sha256']['final.pt'] and not receipt['snapshots_missing']
    return True


def valid_score(lane):
    path=host.RUNS/f'results/{lane}.json'
    if not path.exists():return False
    assert valid_fit(lane)
    row=json.loads(path.read_text()); linkage=row['release_test']
    output=host.RUNS/f'generated/{lane}.npz'; receipt_path=output.with_suffix('.receipt.json')
    receipt=json.loads(receipt_path.read_text())
    assert not output.exists(), 'score is not finalized/pruned'
    assert linkage['lane']==receipt['lane']==lane and linkage['release']==receipt['release']=='0.10.0'
    assert linkage['source_manifest_sha256']==receipt['source_manifest_sha256']==host.sha(PIN)
    assert linkage['generation_receipt_sha256']==host.sha(receipt_path)
    assert linkage['pruned_generated_sha256']==receipt['generated_sha256']==row['fingerprint']['generated_sha256']
    assert receipt['fit_release_receipt_sha256']==host.sha(fit_path(lane)/'release_test_receipt.json')
    assert receipt['fit']==str(fit_path(lane)) and receipt['generation_device']=='cpu'
    assert receipt['mode']==MODE and receipt['M']==32 and receipt['K']==256 and receipt['targets']==37
    assert receipt['states_used']==[f'snap_{u}' for u in range(15000,20000,1000)]+['final']
    assert row['source']==str(output) and row['fold']==0 and row['M']==32
    assert row['fingerprint']['score_code_sha256']==host.sha(host.OLD/'architecture_score.py')
    assert row['fingerprint']['interval_code_sha256']==host.sha(host.OLD/'interval_metrics.py')
    return True


def disk_bytes():
    return sum(p.stat().st_size for p in host.RUNS.rglob('*') if p.is_file())


def check_disk():
    assert disk_bytes() < 1500000000, 'isolated pilot exceeded its1.5GB output cap'
