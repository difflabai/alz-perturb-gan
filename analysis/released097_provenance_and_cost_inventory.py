"""Append-only provenance correction and settled cost inventory; no score selection."""
import argparse,datetime,hashlib,importlib.util,json,math,sys
from pathlib import Path
ROOT=Path('/ml2/alz');PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
PHASES=('faithful097','localfaith097','confirmfaith097','benchmark097')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();assert args.output.resolve().parent==ROOT/'analysis' and not args.output.exists()
 source=ROOT/'analysis/benchmark097_recipe_and_search_disclosure_20261107.json';old=read(source);native=ROOT/'analysis/faithful097/native_trainer_parity_CPU.json';proof=read(native)
 assert proof['status']=='EXACT_RELEASED_GANTRAINER_UPDATE_PARITY_ON_FULLWIDTH_CONDITIONAL_FIXTURE_PASS'
 model=ROOT/'analysis/faithful097/model/released_update.py';assert sha(model)==proof['source_SHA256'][str(model)]
 records=[];seen_attempts=set();seen_ledgers=set();pins={str(source):sha(source),str(native):sha(native)}
 for phase in PHASES:
  for path in sorted((ROOT/'analysis'/phase/'results').glob('*.json')):
   r=read(path);attempt=r['attempt'];assert attempt not in seen_attempts;seen_attempts.add(attempt)
   folder=ROOT/'artifacts/allocations'/attempt;dep=read(folder/'deployment.json');auth_path=Path(dep['authorization_path']);a=read(auth_path);assert sha(auth_path)==dep['authorization_SHA256']
   assert a['attempt_id']==attempt and a['initial_completed_updates']==0 and not a['is_continuation'] and not a['reuse_completed_model'] and a['source_model_path'] is a['source_model_SHA256'] is None
   ledger=Path(a['ledger_path']);assert str(ledger) not in seen_ledgers;seen_ledgers.add(str(ledger));job=read(ledger)['jobs'][0];settlement=ledger.with_name(attempt+'-settlement.json');closed=read(settlement)
   assert job['reservation_released'] and closed['uncertain'] is None and closed['settlement']['status']=='settled'
   seconds=closed['settlement']['charged_GPU_seconds'];assert math.isclose(seconds,r['charged_GPU_seconds'],rel_tol=0,abs_tol=1e-8)
   if a['cell']['method']=='pg':assert a['files_SHA256'][str(model)]==sha(model)
   complete=r['complete'] if phase=='benchmark097' else r['finite']
   records.append(dict(phase=phase,attempt=attempt,method=a['cell']['method'],fold=a['cell']['fold'],seed=a['cell']['seed'],target_endpoint=a['fixed_endpoint'],target_new_update_quantity=a['new_model_updates'],target_update_unit=a.get('new_model_update_unit','optimizer_pairs'),attained_target=complete,from_scratch=True,initial_completed_updates=0,source_checkpoint=None,charged_GPU_seconds=seconds,scientific_or_operational_failure=not complete,source_result_path=str(path),source_result_SHA256=sha(path)))
   for p in (path,auth_path,ledger,settlement):pins[str(p)]=sha(p)
 budget_path=PROJECT/'project_gpu_budget.json';budget=read(budget_path);controller=PROJECT/'budget_executor_v2_20261103_r2/budget_controller.py'
 assert sha(controller)=='c8520f426e6b7a94741dd96804b96d9d22a224bff973b7e85d765e5b83c09848'
 spec=importlib.util.spec_from_file_location('read_only_budget_status',controller);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);status=m.Controller(PROJECT)._status(budget)
 total=sum(q['charged_GPU_seconds'] for q in records);assert total<=status['charged_GPU_seconds']+1e-8
 phases={p:dict(settled_attempts=sum(r['phase']==p for r in records),successful_full_endpoints=sum(r['phase']==p and r['attained_target'] for r in records),failed_attempts=sum(r['phase']==p and not r['attained_target'] for r in records),charged_GPU_seconds=sum(r['charged_GPU_seconds'] for r in records if r['phase']==p)) for p in PHASES}
 q=dict(status='CORRECTED_RELEASED_PG_PROVENANCE_AND_SETTLED_COST_SNAPSHOT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   corrects_prior_disclosure_path=str(source),prior_disclosure_retained_unchanged=True,
   correction='The earlier disclosure used package-level native/released labels too broadly for PG. Its sampled-row SBNP adapter is a distinct objective variant from GANTrainer full512table regularization. Retain all historical numerical scores and costs; do not use those old variant fits as exact released-formulation selection/confirmation evidence. Historical RUNNING statuses there are as-of its00:04timestamp.',
   old_named_results=dict(previous500step_HPO_fits=old['pg']['previous500step_HPO_complete_fits'],previous500step_HPO_GPU_seconds=old['pg']['previous500step_HPO_GPU_seconds'],c01420k_GPU_seconds=old['pg']['previous_c01420k_GPU_seconds'],six_old20k_selection_GPU_seconds=old['pg']['current_full20k_selection_GPU_seconds'],classification='Previously measured sampled-row adapter variants; these named studies are not an exhaustive historical method-cost inventory.'),
   current_qualified_PG_formulation='Released0.9.7 bispf_optimized nonEMA, full512row SBNP regularization, explicit critic thermal binding; positive public scalar parameters, fixed PG8 count application.',
   comparator_recipes={k:old[k] for k in ('scvi','cfgen','hierarchical_resampling')},comparison_scope=old['comparison_scope'],
   corrected_phase_counts_and_costs=phases,corrected_phase_attempts=records,total_settled_corrected_phase_GPU_seconds=total,
   canonical_project_budget_snapshot={k:status[k] for k in ('revision','cap_GPU_hours','charged_GPU_seconds','charged_GPU_hours','reserved_GPU_hours','available_after_reservations_GPU_hours','counted_ledgers','excluded_duplicate_ledgers')},
   historical_or_other_project_work_GPU_seconds=status['charged_GPU_seconds']-total,
   cost_limits='Settled wall/card-hour charges only, not instantaneous utilization. Active elapsed time is not falsely charged twice or treated as final. Previous measured controls incur zero new fit charge; their original cost remains counted. CPU synthetic native-prefix qualification is not a real ALZ fit or GPU-hour charge.',
   unequal_tuning_budgets=True,not_an_equal_search_effort_comparison=True,no_new_training=True,no_new_target_arrays_read=True,full45_external_comparison_completed=(phases['benchmark097']['successful_full_endpoints']==45),
   source_SHA256={str(Path(__file__).resolve()):sha(__file__),str(controller):sha(controller)},input_SHA256=pins,SOTA_claim=False,full_goal_complete=False)
 args.output.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(phases=phases,corrected_settled_GPU_hours=total/3600,project_charged_GPU_hours=status['charged_GPU_hours'],project_reserved_GPU_hours=status['reserved_GPU_hours'])))
if __name__=='__main__':main()
