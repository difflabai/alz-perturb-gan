"""Materialize matched new-configuration followup only after all20 audited fits."""
import argparse,datetime,hashlib,importlib.util,json,subprocess
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/followup097';PAIRED=ROOT/'analysis/refinement097'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--paired-audit',type=Path,required=True);args=parser.parse_args()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
 p=read(STUDY/'protocol.json');assert p['status']=='PREPARED_PENDING_ALL20_PAIRED_REFINEMENT_ENDPOINTS_AND_AUDIT' and not read(ROOT/'followup097_runtime/plan.json')['enabled']
 audit=read(args.paired_audit);report=read(PAIRED/'report.json')
 assert audit['status']=='INDEPENDENT_NATIVE097_PAIRED_REFINEMENT20_FITS_AUDIT_PASS' and audit['required_attempts']==20 and audit['all_planned_attempts_resolved'] and audit['all_new_models_from_update_zero']
 assert audit['activation_actually_reverted'] and audit['owned_processes_and_leases_absent'] and not audit['heldout_values_read']
 assert audit['report_SHA256']==sha(PAIRED/'report.json') and audit['protocol_SHA256']==sha(PAIRED/'protocol.json')
 for key,value in audit['summary'].items():assert report[key]==value
 assert not read(ROOT/'refinement097_runtime/plan.json')['enabled'] and not read(ROOT/'artifacts/gpu1-refinement097-paired-search-20261107/live_registry/registry.json')['jobs']
 project=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');assert not read(project/'project_gpu_budget.json')['open_reservations']
 choices=report['proposed_followup_configurations'];new=[q for q in choices if q['candidate_id']!='d000'];assert 0<=len(new)<=2
 baseline=read(ROOT/'analysis/confirmation097_resampling_controls_CPU_20261106.json');references=[];reference_paths=[]
 for seed in (2,3,4):
  path=ROOT/f'analysis/confirmation097/results/CAL_GPU0_V1_confirm097_d000_f0_pg_s{seed}.json';reference_paths.append(path);r=read(path)
  assert r['candidate_id']=='d000' and r['seed']==seed and r['endpoint']==20000 and r['from_scratch'] and r['initial_completed_updates']==0
  references.append(dict(candidate_id='d000',hyperparameters=r['hyperparameters'],seed=seed,endpoint=20000,initial_completed_updates=0,from_scratch=True,finite=r['finite'],metrics=r['metrics'],baseline_metrics=baseline['metrics_by_seed'][str(seed)],
    source_result_path=str(path),source_result_SHA256=sha(path),read_only_previously_measured_reference=True,new_model_fitted=False,new_GPU_charge=0))
 evidence=[args.paired_audit.resolve(),PAIRED/'report.json',PAIRED/'protocol.json',PAIRED/'runtime_CPU_qualification.json',
   ROOT/'analysis/confirmation097_final_independent_audit_20261107.json',ROOT/'analysis/confirmation097_resampling_controls_CPU_20261106.json',
   ROOT/'analysis/refinement097_followup_selection_rule_20261107.json',ROOT/'analysis/refinement097_followup_selection.py',Path(__file__).resolve(),*reference_paths,
   *[PAIRED/'results'/(key+'.json') for key in report['result_SHA256']]]
 for key,pin in report['result_SHA256'].items():assert sha(PAIRED/'results'/(key+'.json'))==pin==audit['result_SHA256'][key]
 p.update(status='FROZEN_MATCHED_SEED234_FOLLOWUP_AFTER_PAIRED_REFINEMENT' if new else 'NO_NEW_FOLLOWUP_CONFIGURATIONS_SELECTED_REFERENCE_ONLY_DECISION_PENDING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   selected_configurations=new,all_configurations_for_ranking=choices,reference_controls=references,paired_selection_audit_path=str(args.paired_audit.resolve()),paired_selection_report_path=str(PAIRED/'report.json'),selection_evidence_SHA256={str(f):sha(f) for f in evidence})
 (STUDY/'protocol.json').write_text(json.dumps(p,indent=2,allow_nan=False)+'\n')
 for c in new:
  q=dict(id=c['candidate_id'],hyperparameters=c['hyperparameters'],native_version='0.9.7',endpoint=20000,fold=0,selection_seeds=[0,1],training_seeds=[2,3,4],initial_completed_updates=0,from_scratch=True,source_checkpoint=None,heldout_results_used=False,protocol_SHA256=sha(STUDY/'protocol.json'))
  path=STUDY/'proposals'/(c['candidate_id']+'.json');assert not path.exists();path.write_text(json.dumps(q,indent=2)+'\n')
 if not new:
  spec=importlib.util.spec_from_file_location('reference_only_followup',ROOT/'analysis/refinement097_followup_selection.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  summary=m.summarize(report,[],references,{str(s):baseline['metrics_by_seed'][str(s)] for s in (2,3,4)})
  path=STUDY/'reference_only_decision.json';assert not path.exists();path.write_text(json.dumps(dict(status=p['status'],summary=summary,new_model_fitted=False,GPU_allocation_created=False,independent_new_reference_evidence=False,full_goal_complete=False),indent=2)+'\n')
 print(json.dumps(dict(status=p['status'],selected_new_candidate_ids=[q['candidate_id'] for q in new],required_new_fits=3*len(new),reference_refitted=False,activation=False,GPU_allocation_created=False)))
if __name__=='__main__':main()
