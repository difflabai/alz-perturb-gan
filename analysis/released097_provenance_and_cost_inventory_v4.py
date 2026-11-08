"""Append-only settled inventory retaining original failures and fresh replacements."""
import hashlib, importlib.util, json, math, sys
from pathlib import Path
ROOT=Path('/ml2/alz')
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    original=ROOT/'analysis/released097_provenance_and_cost_inventory_v3.py'
    spec=importlib.util.spec_from_file_location('settled_cost_inventory_original',original)
    inventory=importlib.util.module_from_spec(spec);spec.loader.exec_module(inventory)
    inventory.PHASES=('faithful097','localfaith097','followfaith097','confirmfaith097','benchmark097')
    inventory.main()
    output=Path(sys.argv[sys.argv.index('--output')+1]);q=read(output)
    seen={r['attempt'] for r in q['corrected_phase_attempts']}
    seen_ledgers={str(read(ROOT/'artifacts/allocations'/attempt/'authorization.json')['ledger_path']) for attempt in seen}
    records=[]
    for path in sorted((ROOT/'analysis/benchmark097_infra1/results').glob('*.json')):
        r=read(path);attempt=r['attempt'];assert attempt not in seen;seen.add(attempt)
        dep=read(ROOT/'artifacts/allocations'/attempt/'deployment.json');auth=Path(dep['authorization_path']);a=read(auth)
        assert sha(auth)==dep['authorization_SHA256'] and a['attempt_id']==attempt
        assert a['operation']=='external097_infra1_cell' and a['metric_role']=='final_evaluation'
        assert a['cell']['experiment']=='external097infra1' and a['cell']['fold']==2 and a['cell']['seed']==1 and a['cell']['method'] in ('pg','cfgen')
        assert a['initial_completed_updates']==0 and not a['is_continuation'] and not a['reuse_completed_model'] and a['source_model_path'] is a['source_model_SHA256'] is None
        ledger=Path(a['ledger_path']);assert str(ledger) not in seen_ledgers;seen_ledgers.add(str(ledger))
        data=read(ledger);assert len(data['jobs'])==1 and data['gpu_uuid']==a['gpu_uuid'];job=data['jobs'][0]
        settlement=ledger.with_name(attempt+'-settlement.json');closed=read(settlement)
        assert job['reservation_released'] and closed['uncertain'] is None and closed['settlement']['status']=='settled'
        seconds=closed['settlement']['charged_GPU_seconds'];assert math.isfinite(seconds) and seconds>=0 and math.isclose(seconds,r['charged_GPU_seconds'],rel_tol=0,abs_tol=1e-8)
        complete=r['complete'];assert type(complete) is bool
        if complete:assert job['status']=='completed' and job['exit_code']==0
        records.append(dict(phase='benchmark097_infra1',attempt=attempt,method=a['cell']['method'],fold=2,seed=1,target_endpoint=a['fixed_endpoint'],target_new_update_quantity=a['new_model_updates'],target_update_unit=a['new_model_update_unit'],attained_target=complete,from_scratch=True,initial_completed_updates=0,source_checkpoint=None,charged_GPU_seconds=seconds,scientific_or_operational_failure=not complete,source_result_path=str(path),source_result_SHA256=sha(path)))
        for source in (path,auth,ledger,settlement):q['input_SHA256'][str(source)]=sha(source)
    q['corrected_phase_attempts'].extend(records)
    q['corrected_phase_counts_and_costs']['benchmark097_infra1']=dict(settled_attempts=len(records),successful_full_endpoints=sum(r['attained_target'] for r in records),failed_attempts=sum(not r['attained_target'] for r in records),charged_GPU_seconds=sum(r['charged_GPU_seconds'] for r in records))
    total=sum(r['charged_GPU_seconds'] for r in q['corrected_phase_attempts'])
    project=q['canonical_project_budget_snapshot']['charged_GPU_seconds'];assert total<=project+1e-8
    q.update(total_settled_corrected_phase_GPU_seconds=total,historical_or_other_project_work_GPU_seconds=project-total,phase_extension='Adds two explicitly versioned from-zero infrastructure replacements. All original failed charges retained; no extra logical seeds or SOTA conclusion.',full45_external_comparison_completed=(q['corrected_phase_counts_and_costs']['benchmark097']['successful_full_endpoints']+sum(r['attained_target'] for r in records)==45))
    q['source_SHA256'][str(Path(__file__).resolve())]=sha(__file__)
    output.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(infrastructure_replacements=q['corrected_phase_counts_and_costs']['benchmark097_infra1'],corrected_settled_GPU_hours=total/3600)))
if __name__=='__main__':main()
