"""Freeze a corrected released-PG selection after full fresh confirmation audits."""
import argparse,datetime,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/benchmark097'
sys.path.insert(0,str(ROOT/'analysis'))
from released097_external_selection import resolve

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser()
 for name in ('selection-audit','confirmation-audit','confirmation-protocol','confirmation-report'):parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
 p=read(STUDY/'protocol.json');assert p['status']=='PREPARED_NOT_FROZEN_PG_SELECTION_AND_CONFIRMATION_PENDING'
 assert not read(ROOT/'benchmark097_runtime/plan.json')['enabled']
 assert not (STUDY/'matrix_complete.json').exists() and not list((STUDY/'results').glob('*.json'))
 assert not (ROOT/'artifacts/benchmark097-fullmatrix/identity.json').exists()
 s=read(args.selection_audit);a=read(args.confirmation_audit);cp=read(args.confirmation_protocol);report=read(args.confirmation_report)
 choice=resolve(s,a,cp)
 assert sha(args.confirmation_protocol)==a['protocol_SHA256'] and sha(args.confirmation_report)==a['report_SHA256']
 assert a['protocol_path']==str(args.confirmation_protocol.resolve()) and a['report_path']==str(args.confirmation_report.resolve())
 paths=[args.selection_audit,args.confirmation_audit,args.confirmation_protocol,args.confirmation_report,
   ROOT/'analysis/prepare_benchmark097.py',ROOT/'analysis/released097_external_selection.py',ROOT/'analysis/released097_selection.py']
 for audit in (s,a):
  assert sha(audit['protocol_path'])==audit['protocol_SHA256'] and sha(audit['report_path'])==audit['report_SHA256']
  for path,pin in audit['source_SHA256'].items():assert sha(path)==pin;paths.append(Path(path))
  for path in (audit['protocol_path'],audit['report_path'],audit['model_parity_proof_path']):paths.append(Path(path))
  for attempt,pin in audit['result_SHA256'].items():
   result=Path(audit['report_path']).parent/'results'/(attempt+'.json');assert sha(result)==pin;paths.append(result)
  assert sha(audit['model_parity_proof_path'])==audit['model_parity_proof_SHA256']
 model=read(s['model_parity_proof_path']);assert model['status']=='EXACT_RELEASED_GANTRAINER_UPDATE_PARITY_ON_FULLWIDTH_CONDITIONAL_FIXTURE_PASS' and not model['native_package_modified']
 for path,pin in model['source_SHA256'].items():assert sha(path)==pin;paths.append(Path(path))
 project=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');assert not read(project/'project_gpu_budget.json')['open_reservations']
 p.update(status='FROZEN_NATIVE097_ALL45_FRESH_MODELS_BEFORE_NEW_EVALUATION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   PG_candidate_id=choice['candidate_id'],PG_hyperparameters=choice['hyperparameters'],PG_fresh_seed_confirmation_complete=True,
   PG_selected_before_new_heldout_evaluation=True,PG_heldout_results_used_for_selection=False,
   PG_selection_audit_path=str(args.selection_audit.resolve()),PG_confirmation_audit_path=str(args.confirmation_audit.resolve()),
   PG_confirmation_report_path=str(args.confirmation_report.resolve()),PG_confirmation_protocol_path=str(args.confirmation_protocol.resolve()),
   PG_frozen_selection_evidence_SHA256={str(f.resolve()):sha(f) for f in paths},
   frozen_selection_criterion='Lowest full20k mean calibration cell energy distance over all three fresh confirmation seeds2..4; exploratory seeds0/1 excluded. Joint quality and every failure retained.',
   frozen_candidate_joint_quality=choice['joint_quality'],frozen_choice=choice,activation=False,SOTA_claim=False,full_goal_complete=False)
 (STUDY/'protocol.json').write_text(json.dumps(p,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status=p['status'],candidate=choice['candidate_id'],hyperparameters=p['PG_hyperparameters'],required_fresh_models=45,GPU_allocation_created=False,activation=False)))
if __name__=='__main__':main()
