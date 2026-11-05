"""Compare saved training telemetry; this does not establish checkpoint/quality parity."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/alz')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def main():
    protocol_path = ROOT/'analysis/released_best_rate_extension40k_protocol.json'
    protocol = read(protocol_path)
    historical = Path(protocol['training_log']['immutable_copy_path'])
    assert sha(historical) == protocol['training_log']['SHA256']
    old = {r['step']:r for r in map(json.loads,historical.read_text().splitlines())}
    attempt = 'CAL_GPU0_V1_released_v095_bridge_f0_pg_s0'
    logfile = ROOT/'runs'/attempt/'fit/learning_curve.jsonl'
    raw = logfile.read_bytes()
    complete = b''.join(line for line in raw.splitlines(keepends=True) if line.endswith(b'\n'))
    rows = [json.loads(line) for line in complete.splitlines()]
    assert rows[0]['step'] == 1
    assert all(a['step'] < b['step'] for a,b in zip(rows,rows[1:]))
    assert all(r['step'] in old for r in rows)
    differences = []
    for row in rows:
        reference = old[row['step']]
        for key in row.keys() | reference.keys():
            if row.get(key) != reference.get(key):
                differences.append({'step':row['step'],'field':key,'historical':reference.get(key),'latest':row.get(key)})
    folder = ROOT/'artifacts/diagnostics'/f'v095_training_prefix_u{rows[-1]["step"]:05d}'
    folder.mkdir(exist_ok=False,parents=True)
    snapshot = folder/'latest_learning_curve.jsonl';snapshot.write_bytes(complete)
    result = {'status':'ACTUAL_NATIVE_VERSION_TRAINING_TELEMETRY_PREFIX_COMPARISON',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'historical_version':'0.9.4','current_version':'0.9.5',
        'historical_attempt':protocol['parent_attempt'],'current_attempt':attempt,
        'same_rates':protocol['released_hyperparameters'],
        'latest_saved_update':rows[-1]['step'],'saved_rows_compared':len(rows),
        'all_compared_telemetry_exactly_equal':not differences,'different_field_records':differences,
        'latest_log_snapshot_path':str(snapshot),'latest_log_snapshot_SHA256':sha(snapshot),
        'historical_training_log_path':str(historical),'historical_training_log_SHA256':sha(historical),
        'comparison_source_SHA256':sha(__file__),
        'entire_checkpoint_state_parity_established':False,'latest_release_quality_established':False,
        'scope':'Read-only comparison of the actually saved loss/rate/gradient/usage diagnostics at matched update numbers. Equal telemetry does not prove complete model/RNG/optimizer state or final calibration parity. The fixed20k native run and independent saved-state/count/score audit must complete.'}
    destination=ROOT/'analysis'/f'v095_vs094_actual_training_prefix_u{rows[-1]["step"]:05d}.json'
    assert not destination.exists();destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('different_field_records',)},indent=2))


if __name__=='__main__':main()
