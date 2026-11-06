"""Freeze distinct confirmation configurations after both complete20k audits."""
import argparse,datetime,hashlib,importlib.util,json,subprocess
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/confirmation097';RUNTIME=ROOT/'confirmation097_runtime'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):Path(p).write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage1-audit',type=Path,required=True);p.add_argument('--stage2-audit',type=Path,required=True);args=p.parse_args()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
 assert read(STUDY/'protocol.json')['status']=='PREPARED_PENDING_ALL_SIX_AUDITED_SELECTION_OUTCOMES'
 assert not read(RUNTIME/'plan.json')['enabled']
 rows=[];evidence=[]
 for audit_path,stage,name in ((args.stage1_audit,1,'distribution097'),(args.stage2_audit,2,'distribution097b')):
  a=read(audit_path);assert a['status']==f'INDEPENDENT_NATIVE097_FRESH20K_STAGE{stage}_COUNTS_SCORES_AND_SETTLEMENT_AUDIT_PASS'
  assert a[f'complete_stage{stage}'] and a['all_new_models_from_update_zero'] and not a['heldout_values_read'] and not a['formulation_changed']
  assert not read(ROOT/(name+'_runtime')/'plan.json')['enabled']
  report=ROOT/'analysis'/name/f'stage{stage}_report.json';assert not read(report)['full_goal_complete']
  for row in a['rows']:
   result=ROOT/'analysis'/name/'results'/f'CAL_GPU0_V1_dist097_{row["candidate_id"]}_f0_pg_s0.json'
   assert sha(result)==row['result_SHA256'];evidence.append(result)
  rows+=a['rows'];evidence.extend([audit_path.resolve(),report])
 canonical=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/project_gpu_budget.json')
 assert not read(canonical)['open_reservations']
 assert not read(ROOT/'artifacts/gpu1-distribution097b-stage2-20261106/live_registry/registry.json')['jobs']
 rule=ROOT/'analysis/confirmation097_selection_rule_20261106.json';selector=ROOT/'analysis/confirmation097_selection.py'
 for path,pin in read(rule)['source_SHA256'].items():assert sha(path)==pin
 spec=importlib.util.spec_from_file_location('frozen_confirmation_selector',selector);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 baseline=ROOT/'analysis/baseline_calibration_seed0.json';selected=module.choose_configurations(rows,read(baseline)['energy_score'])
 controls=ROOT/'analysis/confirmation097_resampling_controls_CPU_20261106.json';assert set(read(controls)['metrics_by_seed'])=={'1','2','3','4'}
 evidence.extend([rule,selector,baseline,controls,Path(__file__).resolve()])
 q=read(STUDY/'protocol.json');q.update(status='FROZEN_FRESH_NATIVE097_FULL20K_MATCHED_INITIALIZATION_CONFIRMATION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        selected_configurations=selected,stage1_complete_audit_path=str(args.stage1_audit.resolve()),stage2_complete_audit_path=str(args.stage2_audit.resolve()),
        selection_evidence_SHA256={str(p):sha(p) for p in evidence},selection_seed0_outcomes=rows)
 write(STUDY/'protocol.json',q)
 for choice in selected:
  path=STUDY/'proposals'/(choice['candidate_id']+'.json');assert not path.exists()
  write(path,dict(id=choice['candidate_id'],hyperparameters=choice['hyperparameters'],roles=choice['roles'],native_version='0.9.7',endpoint=20000,
        fold=0,selection_seed=0,training_seeds=[1,2,3,4],initial_completed_updates=0,from_scratch=True,source_checkpoint=None,heldout_results_used=False,protocol_SHA256=sha(STUDY/'protocol.json')))
 write(STUDY/'materialization.json',dict(status='FROZEN_DISTINCT_CONFIGURATIONS_RUNTIME_DISABLED_REQUIRES_NEW_CPU_AND_RESOURCE_REVIEW',
       UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_configurations=selected,required_fresh_fits=4*len(selected),
       fresh_seeds=[1,2,3,4],ranking_seed0_excluded=True,source_SHA256=sha(__file__),GPU_allocation_created=False,activation=False))
 print(json.dumps(dict(selected_configurations=selected,required_fits=4*len(selected),GPU_allocation_created=False)))
if __name__=='__main__':main()
