"""Audit prepared grid source/parameter identities without launching candidates."""
import argparse
import ast
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/alz')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    assert not output.exists(), 'Use a new timestamped inventory; preserve historical observations'
    protocol = ROOT / 'analysis/released_tuning/protocol.json'
    readout_path = ROOT / 'analysis/released_tuning_search_readout.json'
    readout = {row['id']: row for row in read(readout_path)['rows']}
    records = []
    source_pins = {str(p): sha(p) for p in (Path(__file__).resolve(), protocol, readout_path)}
    for config in read(protocol)['stage1']['configs']:
        cid = config['id']
        name = 'released_rate_' + cid.removeprefix('lr_').replace('_', '')
        folder = ROOT / (name + '_runtime')
        proof_path = ROOT / 'analysis' / (name + '_runtime_CPU_qualification.json')
        proof = read(proof_path)
        assert proof['status'] == 'PASSED_ISOLATED_RELEASED_PARAMETER_RUNTIME_CPU_QUALIFICATION'
        assert proof['tests_passed'] == 24 and proof['GPU_allocation_created'] is False
        for path, pin in proof['source_SHA256'].items():
            assert sha(path) == pin, path
        constants = {node.targets[0].id: ast.literal_eval(node.value)
                     for node in ast.parse((folder / 'preset_scope.py').read_text()).body
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in ('CANDIDATE_ID', 'HYPER')}
        plan = read(folder / 'plan.json')
        assert constants == {'CANDIDATE_ID': cid, 'HYPER': config['parameters']}
        assert plan['candidate_id'] == cid
        current = readout[cid]
        if current['state'] == 'audited_complete':
            result_path = Path(current['result_path'])
            assert sha(result_path) == current['result_SHA256']
            assert read(result_path)['predeclared_all_three_checkpoint_saved_scores_and_counts_verified'] is True
            state = 'AUDITED_COMPLETE'
            assert plan['enabled'] is False
            source_pins[str(result_path)] = sha(result_path)
        elif current['state'] == 'allocated_quality_not_audited':
            assert cid == 'lr_g050_d100_p050'
            assert plan['enabled'] is True
            owner = Path('/proc/125248')
            worker = Path('/proc/125280')
            assert owner.exists() and worker.exists()
            assert str(folder / 'launch_core_gpu0.py').encode() in (owner / 'cmdline').read_bytes()
            progress_path = ROOT / 'runs' / current['attempt'] / 'fit/progress.json'
            progress = read(progress_path)
            assert 0 < progress['step'] <= 20000
            state = 'LIVE_NATIVE_PHASE_OBSERVED_AT_INVENTORY_TIME'
        else:
            assert current['state'] == 'not_started'
            assert plan['enabled'] is False
            assert not (ROOT / 'artifacts/allocations' / ('CAL_GPU0_V1_' + name + '_f0_pg_s0')).exists()
            state = 'QUALIFIED_DISABLED_UNUSED_NO_GPU_ADMISSION'
        policy = folder / 'policy.prepared.json'
        source_pins.update({str(p): sha(p) for p in (proof_path, policy, folder / 'plan.json')})
        record = {'candidate_id': cid, 'parameters': config['parameters'], 'state': state,
                  'runtime_directory': str(folder), 'prepared_GPU_uuid': read(policy)['gpu'],
                  'runtime_CPU_proof_path': str(proof_path), 'runtime_CPU_tests_passed': 24,
                  'source_hashes_match': True, 'plan_enabled': plan['enabled']}
        if state == 'LIVE_NATIVE_PHASE_OBSERVED_AT_INVENTORY_TIME':
            record.update(owner_PID=125248, worker_PID=125280, observed_logged_updates=progress['step'])
        records.append(record)
    assert len(records) == 27 and len({r['candidate_id'] for r in records}) == 27
    assert sum(row['plan_enabled'] for row in records) == 1
    result = {'status': 'ALL27_DECLARED_RATE_SETTINGS_HAVE_SOURCE_MATCHING_ACTUAL_CPU_QUALIFICATION',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'counts_by_state': dict(Counter(row['state'] for row in records)),
              'prepared_card_counts': dict(Counter(row['prepared_GPU_uuid'] for row in records)),
              'candidates': records, 'source_SHA256': source_pins,
              'new_training_or_allocation_started_by_inventory': False,
              'scope': 'Read-only source, parameter, policy and disabled-plan inventory. Runtime CPU fixture checks establish guarded source wiring and refusals, not convergence. Five independently audited calibration endpoints and one live native phase are distinguished from21 unused source-qualified candidates. Every future launch still needs fresh capacity/source review, a separate activation and one-use admission. Liveness is observed only at this timestamp. No promotion, seed dropping, evaluation access or stable joint ATH claim.'}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'counts_by_state': result['counts_by_state'],
                      'prepared_card_counts': result['prepared_card_counts']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
