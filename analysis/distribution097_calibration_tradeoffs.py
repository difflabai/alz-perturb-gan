"""Keep pooled-mixture wins distinct from stage and donor tradeoffs."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/alz')
def read(path):
    return json.loads(Path(path).read_bytes())
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    sources = [('scVI400', ROOT / 'analysis/CAL_GPU0_V1_matched_scvi_eval1_f0_scvi_s0-result.json'),
               ('CFGen300AE1500FM', ROOT / 'analysis/CAL_GPU0_V1_matched_cfgen_f0_cfgen_s0-result.json'),
               ('hierarchical_resampling', ROOT / 'analysis/baseline_calibration_seed0.json'),
               ('c014_native096_20k', ROOT / 'analysis/hpo20k/results/CAL_GPU0_V1_h20_c014_f0_pg_s0.json')]
    sources += [(f'{cid}_native097_20k', ROOT / f'analysis/distribution097/results/CAL_GPU0_V1_dist097_{cid}_f0_pg_s0.json')
                for cid in ('d000', 'd001', 'd002')]
    rows = []
    reference = read(ROOT / 'analysis/baseline_calibration_seed0.json')
    for name, path in sources:
        if not path.exists():
            continue
        q = read(path)
        if q.get('finite') is False:
            rows.append(dict(method=name, finite=False, source=str(path), source_SHA256=sha(path)))
            continue
        m = q.get('metrics', q.get('candidate', q))
        assert m['donors'] == reference['donors'] and m['training_seed'] == 0
        assert m['calibration_manifest_SHA256'] == reference['calibration_manifest_SHA256']
        assert m['trace_manifest_SHA256'] == reference['trace_manifest_SHA256']
        assert m['sampling_contract'] == reference['sampling_contract']
        assert m['generation_replicates'] == 2 and len(m['per_stage_cell_energy_distance']) == 4
        keys = ['cell_energy_distance', 'energy_score', 'spread_ratio', 'within_condition_spread_ratio',
                'coverage80', 'per_donor_energy_score', 'per_stage_cell_energy_distance']
        rows.append(dict(method=name, finite=True, **{k: m[k] for k in keys},
                         diagnostic_macro_stage_distance=sum(m['per_stage_cell_energy_distance']) / 4,
                         source=str(path), source_SHA256=sha(path)))
    finite = [r for r in rows if r['finite']]
    result = dict(status='MATCHED_SEED0_CALIBRATION_POOLED_STAGE_AND_DONOR_TRADEOFFS',
                  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows,
                  primary_pooled_mixture_distance_ranking=[r['method'] for r in sorted(finite, key=lambda r: r['cell_energy_distance'])],
                  diagnostic_per_stage_rankings=[[r['method'] for r in sorted(finite, key=lambda r: r['per_stage_cell_energy_distance'][stage])] for stage in range(4)],
                  diagnostic_macro_stage_ranking=[r['method'] for r in sorted(finite, key=lambda r: r['diagnostic_macro_stage_distance'])],
                  donor_energy_ranking=[r['method'] for r in sorted(finite, key=lambda r: r['energy_score'])],
                  primary_objective_unchanged=True, heldout_used=False, SOTA_claim=False,
                  source_SHA256=sha(__file__),
                  limitations='Four adaptively reused calibration donors, one training seed, two generation replicates. Pooled mixture distance is the declared primary objective. A pooled win is not a win in every stage; stage-specific distances and donor-level energy/spread remain visible. Macro-stage distance is diagnostic only and does not replace the declared primary objective.')
    path = ROOT / 'analysis/distribution097_calibration_tradeoffs_20261106.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(path=str(path), primary=result['primary_pooled_mixture_distance_ranking'], diagnostic_macro_stage=result['diagnostic_macro_stage_ranking'])))

if __name__ == '__main__':
    main()
