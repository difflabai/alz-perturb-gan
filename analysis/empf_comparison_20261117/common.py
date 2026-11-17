"""Immutable ancestry, actual EMPF proof admission and retained-fit links."""
import json
from pathlib import Path
import host

_old = host.load('frozen_release_common', host.BASE/'common.py')
write = _old.write
cache_anchor = _old.cache_anchor
fit_path = _old.fit_path
check_disk = _old.check_disk
MODE = _old.MODE
PIN = host.HERE/'source_manifest.json'


def sources():
    ancestor = host.BASE/'postfit/source_manifest.json'
    assert host.sha(ancestor) == 'f24736824a3e012ff8572392d4be5235be770d5f2213d0fabc6cc272f3091b7c'
    value = json.loads(ancestor.read_text())
    for p, digest in value.items():
        assert host.sha(p) == digest, ('changed release ancestry', p)
    paths = [ancestor, *host.HERE.glob('*.py'), host.HERE/'PROTOCOL.txt',
             host.HERE/'resolved_recipes.json', host.BASE/'completion/final_audit.json',
             host.BASE/'completion/pilot_comparison.json',
             host.BASE/'completion/independent_final_review.json',
             host.BASE/'completion/lazy_regularizer_audit.json',
             host.BASE/'prelaunch.json',host.BASE/'verification/verification_0.10.0.json',
             host.ROOT/'runs/particlegan_upgrade_20261117/verification/fresh_0.10.0.pt']
    value.update({str(p): host.sha(p) for p in paths})
    return value


def check_sources():
    current = sources()
    assert current == json.loads(PIN.read_text()), 'EMPF execution source differs'
    return current


def control_identity():
    root=host.ROOT/'runs/particlegan_upgrade_20261117/verification'
    checkpoint=root/'fresh_0.10.0.pt'
    proof=host.BASE/'verification/verification_0.10.0.json'
    value=json.loads(proof.read_text())
    prelaunch=json.loads((host.BASE/'prelaunch.json').read_text())
    assert value['status']=='passed' and value['version']=='0.10.0'
    assert prelaunch['proof_sha256'][str(root/proof.name)]==host.sha(proof)
    assert value['fresh_five_step_sha256']==host.sha(checkpoint)
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=host.sha(checkpoint),
                proof=str(proof),proof_sha256=host.sha(proof))


def valid_fit(lane):
    if not _old.valid_fit(lane):return False
    path=fit_path(lane)/'no_skip_audit.json'
    if not path.exists():return False
    value=json.loads(path.read_text())
    assert value['status']=='passed_actual_full_EMPF_no_skips'
    assert value['source_manifest_sha256']==host.sha(PIN) and value['lane']==lane
    assert value['final_checkpoint_sha256']==host.sha(fit_path(lane)/'final.pt')
    assert value['penalty_calls']==value['observed_steps']==value['completed_updates']==20000
    assert value['lazy_k']==1 and value['native_highdim_prototype_bank_inactive']
    return True


def valid_score(lane):
    path=host.RUNS/f'results/{lane}.json'
    if not path.exists():return False
    assert valid_fit(lane)
    return _old.valid_score(lane)


def preflight():
    value = check_sources()
    record = json.loads((host.HERE/'prelaunch.json').read_text())
    assert record['status'] == 'passed_fixed_two_lane_EMPF_preflight'
    assert record['source_manifest_sha256'] == host.sha(PIN)
    paths = {str(host.RUNS/f'verification/{prefix}_{lane}.json')
             for lane in ('native', 'matched') for prefix in ('verification', 'fresh_process')}
    assert set(record['proof_sha256']) == paths
    for p, digest in record['proof_sha256'].items():
        assert host.sha(p) == digest
        proof = json.loads(Path(p).read_text())
        lane=Path(p).stem.rsplit('_',1)[1]
        fresh_process=Path(p).name.startswith('fresh_process_')
        assert proof['status']==('passed_fresh_process_exact_200_to_300' if fresh_process else 'passed_actual_EMPF_semantics_and_full_replay')
        assert proof['lane']==proof['provenance']['lane']==lane
        assert set(proof['checkpoint_sha256'])=={f'replay200_{lane}.pt',f'expected300_{lane}.pt'}
        assert proof['initial_parameter_control']==control_identity()
        assert proof['full_checkpoint_and_metrics_exact'] is True
        assert proof['verification_code_sha256'] == host.sha(host.HERE/'verify.py')
        assert proof['provenance']['host_derivation_sha256'] == host.sha(host.HERE/'host.py')
        for item in proof['provenance']['files'].values():
            assert host.sha(item['original']) == item['original_sha256']
            assert host.sha(item['derived']) == item['derived_sha256']
        for name, digest in proof['checkpoint_sha256'].items():
            assert host.sha(host.RUNS/'verification'/name) == digest
        assert proof['semantics']['lazy_k'] == 1
        assert proof['semantics']['regularizer_calls'] == (100 if fresh_process else 300)
        assert proof['semantics']['prototype_bank_inactive_native_highdim_route']
        if not fresh_process:
            assert proof['rejected_checkpoint_mismatches'] == ['particlegan_version', 'native_recipe']
            assert proof['initial_G_D_prior_parameters_equal_release_control']
    return value
