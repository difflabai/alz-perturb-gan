"""Freeze optional follow-up only from the complete original12-fit audit."""
import argparse,datetime,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/followfaith097'
sys.path.insert(0,str(ROOT/'analysis'))
from released097_followup_design import design
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selection-audit',type=Path,required=True);args=parser.parse_args()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    p=read(STUDY/'protocol.json');assert p['status']=='PREPARED_PENDING_COMPLETE_ORIGINAL12_AND_FOLLOWUP_DESIGN'
    assert not read(ROOT/'followfaith097_runtime/plan.json')['enabled'] and not list((STUDY/'proposals').glob('*.json'))
    audit_path=args.selection_audit.resolve();a=read(audit_path)
    now=time.time();window=ROOT/'analysis/research_window_20261104_five_days.json'
    forecast=ROOT/'analysis/benchmark097_historical_cost_forecast_20261107.json'
    frozen=design(a,now,read(window)['expires_unix'],read(forecast)['estimated_matrix_GPU_hours'])
    # Even the prospective skip retains its exact full-scope time calculation.
    dest=STUDY/'frozen_design.json';assert not dest.exists();dest.write_text(json.dumps(frozen,indent=2)+'\n')
    if not frozen['candidates']:
        print(json.dumps(dict(status='FOLLOWUP_SKIPPED',design_path=str(dest),design=frozen,GPU_allocation_created=False)));return
    paths=[audit_path,Path(__file__).resolve(),ROOT/'analysis/released097_followup_design.py',
        ROOT/'analysis/released097_followup_design_CPU_qualification_20261107.json',
        ROOT/'analysis/released097_selection.py',ROOT/'analysis/released097_external_selection.py',
        forecast,window,STUDY/'selection.py',STUDY/'possible_settings_native_CPU_qualification.json',
        STUDY/'prepared_accounting_CPU_qualification.json']
    for path,pin in a['source_SHA256'].items():assert sha(path)==pin;paths.append(Path(path))
    for key in ('protocol','report'):
        path=Path(a[key+'_path']);assert sha(path)==a[key+'_SHA256'];paths.append(path)
    old_path=Path(read(a['protocol_path'])['baseline_audit_path']);old=read(old_path)
    assert old['status']=='INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS';paths.append(old_path)
    controls=[];combined={}
    for upstream in (old,a):
        for path,pin in upstream['source_SHA256'].items():assert sha(path)==pin;paths.append(Path(path))
        for attempt,pin in upstream['result_SHA256'].items():
            path=Path(upstream['report_path']).parent/'results'/(attempt+'.json')
            assert attempt not in combined and sha(path)==pin;combined[attempt]=pin;paths.append(path)
            r=read(path);controls.append(dict(r,read_only_previously_measured_reference=True,
                new_model_fitted=False,new_GPU_charge=0,source_result_path=str(path),source_result_SHA256=pin))
    assert len(controls)==len(combined)==16
    proof=STUDY/'possible_settings_native_CPU_qualification.json';q=read(proof)
    assert q['status']=='ALL4_POSSIBLE_FOLLOWUP_SETTING_SEED_NATIVE_TRAINER_PREFIX_CASES_PASS'
    for path,pin in q['source_SHA256'].items():assert sha(path)==pin
    native=Path(a['model_parity_proof_path']);assert sha(native)==a['model_parity_proof_SHA256'];paths.append(native)
    project=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
    assert not read(project/'project_gpu_budget.json')['open_reservations']
    p.update(status='FROZEN_RELEASED097_FULL20K_FOLLOWUP_COORDINATE_SEARCH',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_audit_path=str(audit_path),
        upstream_control_audit_path=str(old_path),reference_result_SHA256=combined,
        design_now_unix=now,matrix_cost_forecast_path=str(forecast),candidates=frozen['candidates'],
        reference_controls=sorted(controls,key=lambda r:(r['candidate_id'],r['seed'])),
        maximum_new_fits=frozen['required_new_fits'],design_evidence_SHA256={str(x):sha(x) for x in paths},
        selection_rule='Keep all16 previously measured endpoints and every new paired20k outcome; mean cell distance then donor energy, failed seeds retained; no warm start or new control charge.',
        confirmation_rule='Distinct primary, joint and nominal-reference choices require fresh20k initialization seeds2..4 before any new evaluation.')
    (STUDY/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    for c in frozen['candidates']:
        q=dict(id=c['id'],hyperparameters=c['hyperparameters'],native_version='0.9.7',endpoint=20000,
            fold=0,training_seeds=[0,1],initial_completed_updates=0,from_scratch=True,
            source_checkpoint=None,heldout_results_used=False,protocol_SHA256=sha(STUDY/'protocol.json'))
        path=STUDY/'proposals'/(c['id']+'.json');assert not path.exists();path.write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps(dict(status=p['status'],required_new_fits=p['maximum_new_fits'],controls_refitted=False,
        activation=False,GPU_allocation_created=False,forecast=frozen['forecast'])))
if __name__=='__main__':main()
