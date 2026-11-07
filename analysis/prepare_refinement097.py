"""Freeze the conditional paired calibration study after all four old seeds."""
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/refinement097'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--reference-audit',type=Path,required=True);parser.add_argument('--decision',type=Path,required=True);args=parser.parse_args()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
 p=read(STUDY/'protocol.json');assert p['status']=='PREPARED_PENDING_COMPLETE_REFERENCE_CONFIRMATION_AUDIT'
 assert not read(ROOT/'refinement097_runtime/plan.json')['enabled'] and not list((STUDY/'proposals').glob('*.json'))
 audit=read(args.reference_audit);decision=read(args.decision)
 assert audit['status']=='INDEPENDENT_FRESH_NATIVE097_MATCHED_SEED_CONFIRMATION_AUDIT_PASS' and audit['all_planned_attempts_resolved'] and audit['all_new_models_from_update_zero']
 assert audit['activation_actually_reverted'] and audit['owned_processes_and_leases_absent'] and not audit['heldout_values_read'] and not audit['SOTA_claim']
 assert decision['calibration_refinement_triggered'] and decision['complete_audit_SHA256']==sha(args.reference_audit)
 assert not decision['benchmark_candidate_frozen'] and not decision['GPU_allocation_created']
 assert not read(ROOT/'artifacts/gpu1-confirmation097-matched-seeds-20261106/live_registry/registry.json')['jobs']
 assert not read(ROOT/'confirmation097_runtime/plan.json')['enabled']
 project=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');assert not read(project/'project_gpu_budget.json')['open_reservations']
 design=ROOT/'analysis/distribution097_conditional_next_search_design_20261107.json';scalar=ROOT/'analysis/distribution097_conditional_continuation_CPU_model_qualification_20261107.json'
 assert p['candidates']==read(design)['candidates'] and read(scalar)['actual_full2000gene_optimizer_pairs']==40
 for path,pin in read(design)['source_SHA256'].items():assert sha(path)==pin
 for path,pin in read(scalar)['source_SHA256'].items():assert sha(path)==pin
 controls=read(STUDY/'resampling_controls_CPU.json');references=[];reference_paths=[ROOT/'analysis/distribution097/results/CAL_GPU0_V1_dist097_d000_f0_pg_s0.json',ROOT/'analysis/confirmation097/results/CAL_GPU0_V1_confirm097_d000_f0_pg_s1.json']
 for seed,path in enumerate(reference_paths):
  r=read(path);assert r['finite'] and r['candidate_id']=='d000' and r['seed']==seed and r['endpoint']==20000 and r['from_scratch'] and r['physical_start_update']==0
  assert r['hyperparameters']==read(design)['reference_hyperparameters']
  references.append(dict(candidate_id='d000',seed=seed,hyperparameters=r['hyperparameters'],endpoint=20000,from_scratch=True,initial_completed_updates=0,
    finite=True,metrics=r['metrics'],baseline_metrics=controls['metrics_by_seed'][str(seed)],read_only_previously_measured_reference=True,
    source_result_path=str(path),source_result_SHA256=sha(path),new_model_fitted=False,new_GPU_charge=0))
 evidence=[args.reference_audit.resolve(),args.decision.resolve(),design,scalar,STUDY/'resampling_controls_CPU.json',*reference_paths,
   ROOT/'analysis/distribution097_stage1_final_independent_audit_20261106.json',ROOT/'analysis/refinement097_selection.py',Path(__file__).resolve()]
 p.update(status='FROZEN_CONDITIONAL_PAIRED_SEED_FULL20K_NATIVE097_REFINEMENT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   original_reference_confirmation_audit_path=str(args.reference_audit.resolve()),original_reference_next_decision_path=str(args.decision.resolve()),
   design_evidence_SHA256={str(f):sha(f) for f in evidence},reference_controls=references,
   selection_rule='Resolve every new setting at both seeds0/1, rank finite complete paired means by cell energy distance then donor energy; retain joint-quality tradeoffs and measured d000 control. No failed-seed dropping.',
   followup_rule='Distinct primary-distance and joint-quality choices plus measured d000 reference; new choices require frozen matched-seed2..4 full20k followup. d000 followup scores are already known controls and must not be called new independent evidence.')
 (STUDY/'protocol.json').write_text(json.dumps(p,indent=2,allow_nan=False)+'\n')
 for c in p['candidates']:
  q=dict(id=c['id'],hyperparameters=c['hyperparameters'],changed_field=c['changed_field'],native_version='0.9.7',endpoint=20000,fold=0,training_seeds=[0,1],initial_completed_updates=0,from_scratch=True,source_checkpoint=None,heldout_results_used=False,protocol_SHA256=sha(STUDY/'protocol.json'))
  out=STUDY/'proposals'/(c['id']+'.json');assert not out.exists();out.write_text(json.dumps(q,indent=2)+'\n')
 print(json.dumps(dict(status=p['status'],new_fits=20,reference_controls_refitted=False,activation=False,GPU_allocation_created=False)))
if __name__=='__main__':main()
