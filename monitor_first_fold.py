"""Retain read-only progress observations for the existing guarded ALZ job."""
import datetime
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/ml2/alz')
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
OUT = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103/runs_gpu0_v1/GPU0_V1_f0_pg_s0')
LEDGERS = PROJECT / 'core_gpu0_v1_ledgers'
ATTEMPT = 'GPU0_V1_f0_pg_s0'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def observe():
    ledger = read(LEDGERS / (ATTEMPT + '.json'))
    job = ledger['jobs'][0] if ledger and ledger.get('jobs') else {}
    health = read(LEDGERS / 'cooperative_live.json')
    progress = read(OUT / 'fit/progress.json')
    result = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'step': progress.get('step') if progress else None,
        'guard_status': job.get('status'),
        'elapsed_seconds': time.time() - job['started_unix'] if job else None,
        'checkpoint': read(OUT / 'fit/checkpoint_receipt.json'),
        'health': health,
        'completed': read(OUT / 'completed.json'),
        'failure': read(OUT / 'failure.json'),
        'fit_failure': read(OUT / 'fit/failure.json'),
        'settlement': read(LEDGERS / (ATTEMPT + '-settlement.json')),
        'evaluations': {name: read(OUT / name / 'metrics.json') for name in
                        ['evaluation_r0', 'resampling_r0', 'evaluation_r1',
                         'resampling_r1', 'stage_swap_r0']},
    }
    return result


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    while True:
        result = observe()
        atomic(ROOT / 'latest_progress.json', result)
        with (ROOT / 'progress_history.jsonl').open('a') as stream:
            stream.write(json.dumps(result, allow_nan=False) + '\n')
        if result['settlement']:
            paths = [p for p in OUT.rglob('*') if p.is_file()]
            paths += [LEDGERS / (ATTEMPT + '.json'),
                      LEDGERS / (ATTEMPT + '-settlement.json')]
            hashes = {}
            for path in paths:
                h = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(chunk)
                hashes[str(path)] = h.hexdigest()
            atomic(ROOT / 'terminal_observation.json', {
                'observation': result, 'files_SHA256': hashes,
                'scope': 'one fold, one training seed; no full-matrix inference',
            })
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
