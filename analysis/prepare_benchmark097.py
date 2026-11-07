"""Freeze the prespecified confirmation winner before any new external scores."""
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/benchmark097';CONFIRM=ROOT/'analysis/confirmation097'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--confirmation-audit',type=Path,required=True);args=parser.parse_args()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
 p=read(STUDY/'protocol.json');assert p['status']=='PREPARED_NOT_FROZEN_PG_SELECTION_AND_CONFIRMATION_PENDING'
 assert not read(ROOT/'benchmark097_runtime/plan.json')['enabled']
 assert not (STUDY/'matrix_complete.json').exists() and not list((STUDY/'results').glob('*.json'))
 assert not (ROOT/'artifacts/benchmark097-fullmatrix/identity.json').exists()
 audit=read(args.confirmation_audit);assert audit['status']=='INDEPENDENT_FRESH_NATIVE097_MATCHED_SEED_CONFIRMATION_AUDIT_PASS'
 assert audit['all_planned_attempts_resolved'] and audit['all_new_models_from_update_zero'] and audit['training_seeds']==[1,2,3,4]
 assert audit['activation_actually_reverted'] and audit['owned_processes_and_leases_absent'] and not audit['heldout_values_read'] and not audit['CUDA_initialized']
 assert not audit['broader_goal_complete'] and not audit['SOTA_claim']
 cp=read(CONFIRM/'protocol.json');report=read(CONFIRM/'report.json')
 assert sha(CONFIRM/'protocol.json')==audit['protocol_SHA256'] and sha(CONFIRM/'report.json')==audit['report_SHA256']
 for path,pin in audit['source_SHA256'].items():assert sha(path)==pin
 for key,value in audit['summary'].items():assert report[key]==value
 winner=audit['summary']['proposed_external_candidate'];assert winner is not None,'No fully finite fresh-seed candidate; inspect rather than silently fall back'
 selected=next(q for q in cp['selected_configurations'] if q['candidate_id']==winner)
 entry=next(q for q in audit['summary']['candidates'] if q['candidate_id']==winner)
 assert entry['all_four_fresh_seeds_finite'] and entry['seeds']==[1,2,3,4] and not entry['failed_seeds']
 paths=[args.confirmation_audit,CONFIRM/'protocol.json',CONFIRM/'report.json',CONFIRM/'runtime_CPU_qualification.json',ROOT/'analysis/audit_confirmation097.py',
        ROOT/'analysis/prepare_benchmark097.py',ROOT/'analysis/confirmation097_selection.py',ROOT/'analysis/confirmation097_selection_rule_20261106.json',
        *[Path(x) for x in cp['selection_evidence_SHA256']],*[CONFIRM/'results'/(attempt+'.json') for attempt in report['result_SHA256']]]
 for attempt,pin in report['result_SHA256'].items():assert sha(CONFIRM/'results'/(attempt+'.json'))==pin==audit['result_SHA256'][attempt]
 p.update(status='FROZEN_NATIVE097_ALL45_FRESH_MODELS_BEFORE_NEW_EVALUATION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    PG_candidate_id=winner,PG_hyperparameters=selected['hyperparameters'],PG_fresh_seed_confirmation_complete=True,
    PG_selected_before_new_heldout_evaluation=True,PG_heldout_results_used_for_selection=False,
    PG_confirmation_audit_path=str(args.confirmation_audit.resolve()),PG_confirmation_report_path=str(CONFIRM/'report.json'),
    PG_confirmation_protocol_path=str(CONFIRM/'protocol.json'),PG_frozen_selection_evidence_SHA256={str(f.resolve()):sha(f) for f in paths},
    frozen_selection_criterion='Minimum mean calibration cell energy distance over all four new initialization seeds1..4; seed0 excluded. Joint quality and all failures retained separately.',
    frozen_candidate_joint_quality=entry['joint_quality'],activation=False,SOTA_claim=False,full_goal_complete=False)
 (STUDY/'protocol.json').write_text(json.dumps(p,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status=p['status'],candidate=winner,hyperparameters=p['PG_hyperparameters'],required_fresh_models=45,GPU_allocation_created=False,activation=False)))
if __name__=='__main__':main()
