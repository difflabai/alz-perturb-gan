"""Materialize fresh seed2..4 choices only after a complete released-objective audit."""
import argparse,datetime,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/confirmfaith097';sys.path.insert(0,str(ROOT/'analysis'))
from released097_selection import FORMULATION,proposed_confirmation
from released097_external_selection import validate_summary

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--selection-audit',type=Path,required=True);args=parser.parse_args()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
 p=read(STUDY/'protocol.json');assert p['status']=='PREPARED_PENDING_CORRECTED_RELEASED_PAIRED_SELECTION_AUDIT'
 assert not read(ROOT/'confirmfaith097_runtime/plan.json')['enabled'] and not list((STUDY/'proposals').glob('*.json'))
 a=read(args.selection_audit);assert a['status'] in ('INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS','INDEPENDENT_RELEASED097_PAIRED_SELECTION_AUDIT_PASS')
 assert a['formulation']==FORMULATION and a['training_seeds']==[0,1] and a['all_planned_attempts_resolved'] and a['all_new_models_from_update_zero']
 assert a['activation_actually_reverted'] and a['owned_processes_and_leases_absent'] and not a['heldout_values_read'] and not a['CUDA_initialized']
 validate_summary(a['summary'],[0,1]);assert a['required_attempts']==2*len(a['summary']['candidates'])
 choices=proposed_confirmation(a['summary'],a['summary']['reference_id']);assert choices==a['summary']['proposed_confirmation_configurations'] and 1<=len(choices)<=3
 paths=[args.selection_audit.resolve(),Path(__file__).resolve(),ROOT/'analysis/released097_selection.py',ROOT/'analysis/released097_external_selection.py',Path(a['model_parity_proof_path'])]
 for path,pin in a['source_SHA256'].items():assert sha(path)==pin;paths.append(Path(path))
 for key in ('protocol','report'):
  path=Path(a[key+'_path']);assert sha(path)==a[key+'_SHA256'];paths.append(path)
 for attempt,pin in a['result_SHA256'].items():
  path=Path(a['report_path']).parent/'results'/(attempt+'.json');assert sha(path)==pin;paths.append(path)
 assert sha(a['model_parity_proof_path'])==a['model_parity_proof_SHA256']
 project=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');assert not read(project/'project_gpu_budget.json')['open_reservations']
 p.update(status='FROZEN_RELEASED097_MATCHED_SEED234_CONFIRMATION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_configurations=choices,
   selection_audit_path=str(args.selection_audit.resolve()),selection_evidence_SHA256={str(x):sha(x) for x in paths},all_selected_models_are_new_fits=True,maximum_new_fits=3*len(choices),
   ranking_rule='All selected settings at all fresh initialization seeds2..4; minimize mean full20k calibration cell energy distance, then donor energy score. Exploratory seeds0/1 excluded. Retain failures, SD, controls, spread and coverage.')
 (STUDY/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
 for c in choices:
  q=dict(id=c['candidate_id'],hyperparameters=c['hyperparameters'],native_version='0.9.7',endpoint=20000,fold=0,training_seeds=[2,3,4],selection_seeds=[0,1],initial_completed_updates=0,from_scratch=True,source_checkpoint=None,heldout_results_used=False,protocol_SHA256=sha(STUDY/'protocol.json'))
  path=STUDY/'proposals'/(c['candidate_id']+'.json');assert not path.exists();path.write_text(json.dumps(q,indent=2)+'\n')
 print(json.dumps(dict(status=p['status'],new_full20k_fits=p['maximum_new_fits'],activation=False,GPU_allocation_created=False)))
if __name__=='__main__':main()
