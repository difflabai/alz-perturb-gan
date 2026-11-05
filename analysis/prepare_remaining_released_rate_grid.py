"""Prepare and CPU-qualify missing declared rate settings, without GPU admission."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess

ROOT = Path('/ml2/alz')
PREDECESSOR = 'CAL_GPU0_V1_released_rate_g050d100p050_f0_pg_s0'
OUTPUT = ROOT / 'analysis/released_remaining_rate_grid_CPU_preparation_20261105.json'
LOGS = ROOT / 'artifacts/diagnostics/released_remaining_rate_grid_CPU_20261105'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualify(item):
    folder = Path(item['runtime_directory'])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (LOGS / (item['candidate_id'] + '.stdout')).open('xb') as out, \
         (LOGS / (item['candidate_id'] + '.stderr')).open('xb') as err:
        result = subprocess.run(['/usr/bin/python3', '-I', '-S', '-B', str(folder / 'qualify.py')],
                                cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, timeout=150)
    proof = ROOT / 'analysis' / (folder.name + '_CPU_qualification.json')
    record = {'candidate_id': item['candidate_id'], 'returncode': result.returncode,
              'runtime_directory': str(folder), 'physical_GPU_uuid': item['physical_GPU_uuid']}
    if result.returncode == 0:
        q = read(proof)
        assert q['status'] == 'PASSED_ISOLATED_RELEASED_PARAMETER_RUNTIME_CPU_QUALIFICATION'
        assert q['tests_passed'] == 24 and q['GPU_allocation_created'] is False
        for path, pin in q['source_SHA256'].items():
            assert sha(path) == pin, path
        assert read(folder / 'plan.json')['enabled'] is False
        record.update(status='ACTUAL_24_CPU_CHECKS_PASSED_DISABLED_NO_GPU_ADMISSION',
                      proof_path=str(proof), proof_SHA256=sha(proof))
    else:
        record['status'] = 'CPU_QUALIFICATION_FAILED_INSPECT_CAPTURED_LOGS'
    return record


def main():
    assert not OUTPUT.exists(), 'Inspect existing preparation proof; do not overwrite or rerun passed checks'
    assert not LOGS.exists()
    LOGS.mkdir(parents=True)
    generator_path = ROOT / 'analysis/prepare_released_rate_candidate.py'
    generator = runpy.run_path(str(generator_path))['make_candidate']
    protocol_path = ROOT / 'analysis/released_tuning/protocol.json'
    configs = read(protocol_path)['stage1']['configs']
    assert len(configs) == 27
    generated = []
    for config in configs:
        candidate_id = config['id']
        name = 'released_rate_' + candidate_id.removeprefix('lr_').replace('_', '') + '_runtime'
        if (ROOT / name).exists():
            continue
        # These are disabled card proposals; future fresh admission remains mandatory.
        item = generator(candidate_id, len(generated) % 2, PREDECESSOR)
        assert item['status'] == 'GENERATED_DISABLED_DECLARED_RATE_CANDIDATE_NOT_QUALIFIED_OR_DEPLOYED'
        generated.append(item)
        print('prepared', candidate_id, item['physical_GPU_uuid'], flush=True)
    result = {
        'status': 'REMAINING_REGISTERED_RATE_SETTINGS_CPU_PREPARATION_IN_PROGRESS',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'declared_configurations': 27, 'newly_generated': len(generated), 'qualified': [],
        'GPU_allocation_created': False, 'GPU_job_started': False,
        'predecessor_attempt': PREDECESSOR,
        'source_SHA256': {str(p): sha(p) for p in (Path(__file__).resolve(), generator_path, protocol_path)},
        'scope': 'Only missing declared rate-grid members. Inactive per-candidate GPU0/GPU1 proposals alternate; each actual private CPU source/controller/lease/accounting/refusal qualification is required. Existing qualified or consumed runtimes are untouched. No training, native admission, automatic successor, changed science or incumbent promotion.',
    }
    progress = LOGS / 'progress.json'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for start in range(0, len(generated), 2):
            futures = [executor.submit(qualify, item) for item in generated[start:start + 2]]
            batch = [future.result() for future in futures]
            result['qualified'].extend(batch)
            progress.write_text(json.dumps(result, indent=2) + '\n')
            for record in batch:
                print(record['status'], record['candidate_id'], flush=True)
            assert all(item['returncode'] == 0 for item in batch), 'Inspect failed CPU checks before preparing further execution'
    assert len(result['qualified']) == len(generated)
    result.update(status='ALL_NEWLY_PREPARED_REMAINING_RATE_SETTINGS_ACTUAL_CPU_CHECKS_PASSED_NO_GPU_ADMISSION',
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  actual_CPU_tests_passed=24 * len(generated))
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'newly_qualified_settings': len(generated),
                      'actual_CPU_tests': result['actual_CPU_tests_passed'], 'GPU_jobs_started': 0}), flush=True)


if __name__ == '__main__':
    main()
