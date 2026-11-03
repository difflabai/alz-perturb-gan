"""Read-only observations of one existing finite candidate; no process control."""
import argparse
import datetime
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    root = Path('/ml2/alz')
    deployment = read(root / 'artifacts/allocations' / args.attempt / 'deployment.json')
    out, ledger = Path(deployment['output_dir']), Path(deployment['ledger_path'])
    monitor = root / 'artifacts/monitor' / args.attempt
    monitor.mkdir(parents=True, exist_ok=False)
    gpu_peak, ram_peak = 0, 0
    while True:
        now = time.time()
        record = read(ledger)
        job = record['jobs'][0] if record and record.get('jobs') else {}
        health = read(ledger.parent / 'cooperative_live.json')
        if health:
            gpu_peak = max(gpu_peak, health['own_gpu_bytes'])
            ram_peak = max(ram_peak, health['own_ram_bytes'])
        settlement = read(ledger.with_name(args.attempt + '-settlement.json'))
        final = read(out / 'completed.json')
        result = {
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'attempt': args.attempt, 'candidate_commit': deployment['candidate_commit'],
            'guard_status': job.get('status'),
            'elapsed_seconds': now - job['started_unix'] if job else None,
            'progress': read(out / 'fit/progress.json'),
            'checkpoint': read(out / 'fit/checkpoint_receipt.json'),
            'health': health, 'sampled_GPU_peak_bytes': gpu_peak,
            'sampled_host_RAM_peak_bytes': ram_peak,
            'failure': read(out / 'failure.json'),
            'fit_failure': read(out / 'fit/failure.json'),
            'completed': final,
            'settled': bool(settlement and settlement.get('settlement')),
            'uncertain': settlement.get('uncertain') if settlement else None,
            'charged_GPU_seconds': settlement['settlement'].get('charged_GPU_seconds') if settlement and settlement.get('settlement') else None,
        }
        atomic(monitor / 'latest.json', result)
        with (monitor / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(result, allow_nan=False) + '\n')
        if settlement:
            atomic(monitor / 'terminal.json', result)
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
