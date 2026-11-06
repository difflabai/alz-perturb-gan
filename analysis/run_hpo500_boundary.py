"""Finite, reproducible500-step HPO with full calibration and fresh-seed checks."""
import argparse,datetime,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/hpo500_boundary';STATE=ROOT/'artifacts/hpo500-boundary-batch'
RUNTIME=ROOT/'hpo500_boundary_runtime'
PG=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/bin/python')
SITE=PG.parent.parent/'lib/python3.12/site-packages'
SHARED=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/.venv312/lib/python3.12/site-packages')
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):
    p=Path(p);temp=p.with_suffix('.partial');temp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');temp.replace(p)
def command(args,timeout=120):
    p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError(f'Command failed({p.returncode}): {args[0]}: {p.stderr[-3000:]}')
    return p.stdout
def commit(paths,message):
    command(['git','add','--',*paths]);command(['git','commit','--only','-m',message,'--',*paths])
def key(cid,seed,phase):return f'CAL_GPU0_V1_h5_{cid}_{phase}_f0_pg_s{seed}'
def output(cid,seed,phase):return STUDY/'results'/(key(cid,seed,phase)+'.json')
def import_search():
    sys.path[:0]=[str(STUDY),str(SITE),str(SHARED)]
    import search
    return search
def report(kind,event,summary,details):
    spec=importlib.util.spec_from_file_location('core_reporter',ROOT/'analysis/external_benchmark_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.report(kind,event,summary,json.dumps(details))
def verify_result(q,cid,seed,phase):
    assert q['status'] in ('HPO500_NATIVE_COUNTS_SCORES_CLEAN_AUDIT_PASS','HPO500_SCIENTIFIC_NONFINITE_FAILURE_CLEAN_AUDIT')
    assert q['candidate_id']==cid and q['seed']==seed and q['phase']==phase and q['endpoint']==500
    assert q['protocol_SHA256']==sha(STUDY/'prospective_protocol.json') and q['proposal_SHA256']==sha(STUDY/'proposals'/(cid+'.json'))
    assert q['owned_processes_and_lease_absent'] and q['clean_settlement'] and q['calibration_only']
    for path,pin in q.get('files_SHA256',{}).items():assert sha(path)==pin
def wait_for_capacity(cid,seed,phase):
    # No allocation, lease, model import, source change, or trial clock reset.
    deadline=read(STUDY/'prospective_protocol.json')['same_five_day_window']
    expiry=read(deadline)['expires_unix']
    args=[sys.executable,'-I','-S','-B',RUNTIME/'prepare.py','--candidate',cid,'--seed',seed,'--phase',phase,'--check-only']
    known_capacity=('Exact parallel GPU growth forecasts do not fit','Exact parallel host growth forecasts do not fit','ALZ early stop: actual GPU free below3.5GiB','ALZ early stop: host available below12GiB')
    while True:
        assert time.time()<expiry-150,'Original trial/research expiry leaves no safe continuation interval'
        p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=120)
        if p.returncode==0:return
        if not any(message in p.stderr for message in known_capacity):raise RuntimeError('Read-only preflight failed: '+p.stderr[-3000:])
        write(STATE/'progress.json',dict(status='WAITING_FOR_SHARED_CAPACITY_NO_ALLOCATION',candidate=cid,seed=seed,phase=phase,attempt=key(cid,seed,phase),audited_results=len(list((STUDY/'results').glob('*.json'))),original_expiry_unix=expiry))
        time.sleep(15)
def run_cell(cid,seed,phase):
    result=output(cid,seed,phase)
    if result.exists():
        q=read(result);verify_result(q,cid,seed,phase);return q
    physical=key(cid,seed,phase)
    folder=ROOT/'artifacts/allocations'/physical
    if not folder.exists():
        assert not command(['git','status','--porcelain']),'Uncommitted work; study stops without touching it'
        wait_for_capacity(cid,seed,phase)
        text=command([sys.executable,'-I','-S','-B',RUNTIME/'prepare.py','--candidate',cid,'--seed',seed,'--phase',phase]);(STATE/(key(cid,seed,phase)+'-prepare.stdout')).write_text(text)
    dep=read(folder/'deployment.json');auth=read(dep['authorization_path']);assert sha(dep['authorization_path'])==dep['authorization_SHA256']
    receipt=folder/'launch_receipt.json';owner=None
    if not receipt.exists():
        assert not Path(auth['ledger_path']).exists() and not Path(auth['output_dir']).exists() and time.time()<auth['expires_unix']-150
        with (folder/'owner.stdout').open('xb') as out,(folder/'owner.stderr').open('xb') as err:
            owner=subprocess.Popen([str(PG),'-I','-S','-B',str(RUNTIME/'launch_core_gpu0.py'),'--authorization',dep['authorization_path']],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        birth=int(Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()[19])
        write(receipt,dict(pid=owner.pid,birth=birth,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    write(STATE/'progress.json',dict(status='NATIVE500STEP_FIT_OR_CALIBRATION_ACTIVE',candidate=cid,seed=seed,phase=phase,attempt=physical,audited_results=len(list((STUDY/'results').glob('*.json')))))
    ledger=Path(auth['ledger_path']);settlement=ledger.with_name(physical+'-settlement.json')
    while not settlement.exists():
        assert time.time()<auth['expires_unix']+660,'Finite phase failed to settle; no retry'
        time.sleep(5)
    if owner is not None:owner.wait(timeout=60)
    else:
        identity=read(receipt);assert not Path('/proc',str(identity['pid'])).exists(),'Previous owner still present; inspect'
    text=command([str(PG),'-I','-S','-B',RUNTIME/'finish.py','--attempt',physical],timeout=600)
    (folder/'audit.stdout').write_text(text)
    q=read(result);verify_result(q,cid,seed,phase)
    commit([str(result)],f'Record500step HPO {cid} {phase} seed{seed}, retaining outcome')
    return q
def final_report(records,frozen,confirmation):
    import numpy as np
    original=ROOT/'analysis/hpo500';previous=read(original/'complete_report.json')
    controls=[read(original/'results'/f'CAL_GPU0_V1_h5_c000_confirm_f0_pg_s{s}.json') for s in (1,2,3,4)]
    anchor=read(STUDY/'anchor.json');anchor_rows=[read(original/'results'/f'CAL_GPU0_V1_h5_{anchor["anchor_candidate"]}_confirm_f0_pg_s{s}.json') for s in (1,2,3,4)]
    comparisons=list(previous['comparisons']);new_comparison=None
    if confirmation:
        assert [r['seed'] for r in confirmation]==[1,2,3,4]
        good=all(r['finite'] for r in confirmation+controls)
        new_comparison=dict(candidate=frozen['candidate_id'],eligible=False)
        if good:
            es=np.array([r['metrics']['energy_score']-c['metrics']['energy_score'] for r,c in zip(confirmation,controls)])
            ed=np.array([r['metrics']['cell_energy_distance']-c['metrics']['cell_energy_distance'] for r,c in zip(confirmation,controls)])
            spread=float(np.mean([r['metrics']['spread_ratio'] for r in confirmation]))
            indices=np.random.default_rng(20261106).integers(0,4,size=(10000,4))
            new_comparison.update(eligible=bool(es.mean()<0 and ed.mean()<=0 and .7<=spread<=1.3 and (es<0).sum()>=3),mean_energy_score=float(np.mean([r['metrics']['energy_score'] for r in confirmation])),mean_cell_energy_distance=float(np.mean([r['metrics']['cell_energy_distance'] for r in confirmation])),mean_spread_ratio=spread,paired_seed_ES_deltas=es.tolist(),paired_seed_cellED_deltas=ed.tolist(),mean_ES_delta=float(es.mean()),mean_cellED_delta=float(ed.mean()),seeds_improved=int((es<0).sum()),ES_delta95percent_seed_bootstrap_interval=np.percentile(es[indices].mean(1),[2.5,97.5]).tolist())
            if all(r['finite'] for r in anchor_rows):
                new_comparison['paired_seed_ES_deltas_vs_anchor']=[r['metrics']['energy_score']-a['metrics']['energy_score'] for r,a in zip(confirmation,anchor_rows)]
                new_comparison['paired_seed_cellED_deltas_vs_anchor']=[r['metrics']['cell_energy_distance']-a['metrics']['cell_energy_distance'] for r,a in zip(confirmation,anchor_rows)]
        else:new_comparison['reason']='All failed/nonfinite seeds retained, no seed dropping'
        comparisons.append(new_comparison)
    eligible=sorted([r for r in comparisons if r['eligible']],key=lambda r:(r['mean_energy_score'],r['mean_cell_energy_distance'],r['candidate']))
    selected=eligible[0]['candidate'] if eligible else 'c000'
    source=STUDY if selected.startswith('k') else original
    return dict(status='COMPLETE_PROSPECTIVE500STEP_KAPPA_BOUNDARY_AND_ORIGINAL_CONFIRMATION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),endpoint=500,native_version='0.9.6',EMA_decay=0,formulation_changed=False,original_report_path=str(original/'complete_report.json'),original_report_SHA256=sha(original/'complete_report.json'),original_selection_candidates=25,original_confirmation_fits=12,boundary_selection_candidates=len(records),boundary_confirmation_fits=len(confirmation),new_boundary_comparison=new_comparison,comparisons=comparisons,selected_candidate=selected,selected_hyperparameters=read(source/'proposals'/(selected+'.json'))['hyperparameters'],joint_improvement_confirmed=bool(eligible),control_confirmation_means={k:float(np.mean([c['metrics'][k] for c in controls])) for k in ('energy_score','cell_energy_distance','spread_ratio')} if all(r['finite'] for r in controls) else None,anchor_candidate=anchor['anchor_candidate'],anchor_selection_hyperparameters=anchor['anchor_hyperparameters'],original_actual_GPU_seconds=previous['total_actual_GPU_seconds'],new_actual_GPU_seconds=sum(r['charged_GPU_seconds'] for r in records+confirmation),total_actual_GPU_seconds=previous['total_actual_GPU_seconds']+sum(r['charged_GPU_seconds'] for r in records+confirmation),new_result_SHA256={r['attempt']:sha(STUDY/'results'/(r['attempt']+'.json')) for r in records+confirmation},protocol_SHA256=sha(STUDY/'prospective_protocol.json'),fresh_seed_confirmation_seeds=[1,2,3,4],failed_seeds_dropped=False,heldout_results_used=False,automatic_incumbent_promotion=False,limitation='Best measured within this adaptive study and predeclared boundary investigation. Four adaptive calibration donors; seed bootstrap is conditional on them. No global optimum, independent cohort or SOTA claim.')
def main():
    p=argparse.ArgumentParser();p.add_argument('--activation',required=True);args=p.parse_args()
    STATE.mkdir(exist_ok=True);(STUDY/'results').mkdir(exist_ok=True)
    with (STATE/'driver.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (STATE/'failure.json').exists(),'Prior failure requires inspection; no retry'
        write(STATE/'driver_identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=sha(__file__),activation=args.activation))
        try:
            assert (ROOT/'analysis/hpo500/complete_report.json').exists()
            search=import_search();records=[]
            def declare_and_run(index):
                proposal=search.proposal(index,records);path=STUDY/'proposals'/(proposal['id']+'.json')
                if path.exists():assert read(path)==proposal
                else:write(path,proposal);commit([str(path)],f'Declare prospective500step kappa proposal {proposal["id"]}')
                q=run_cell(proposal['id'],0,'search');records.append(q)
                write(STATE/'search_summary.json',dict(completed=len(records),maximum=6,rows=[{k:r[k] for k in ('candidate_id','finite','metrics','charged_GPU_seconds')} for r in records]))
            for index in range(4):declare_and_run(index)
            extra=search.refinements(records)
            for index in range(4,4+len(extra)):declare_and_run(index)
            winner=search.best(records);a=read(STUDY/'anchor.json');improved=winner['candidate_id']!=a['anchor_candidate']
            frozen=dict(status='SUPPLEMENTAL_SEED0_SELECTION_FROZEN',candidate_id=winner['candidate_id'] if improved else None,anchor_candidate=a['anchor_candidate'],anchor_improved_at_seed0=improved,frozen_before_new_confirmation=True,confirmation_seeds=[1,2,3,4],selection_result_SHA256={r['attempt']:sha(STUDY/'results'/(r['attempt']+'.json')) for r in records},protocol_SHA256=sha(STUDY/'prospective_protocol.json'))
            path=STUDY/'frozen_selection.json'
            if path.exists():assert read(path)==frozen
            else:write(path,frozen);commit([str(path)],'Freeze kappa boundary winner before four fresh training seeds')
            confirmation=[]
            if improved:
                for seed in (1,2,3,4):confirmation.append(run_cell(frozen['candidate_id'],seed,'confirm'))
            result=final_report(records,frozen,confirmation);path=STUDY/'complete_report.json';assert not path.exists();write(path,result)
            gp=ROOT/'analysis/current_research_goal.json';goal=read(gp);goal['study_status']=result['status'];goal['boundary_supplement_pending']=False;goal['boundary_result_path']=str(path);goal['joint_improvement_confirmed']=result['joint_improvement_confirmed'];write(gp,goal)
            commit([str(path),str(gp)],'Record boundary-refined500step results with all original and new fresh-seed comparisons')
            command(['git','revert','--no-edit',args.activation])
            write(STATE/'progress.json',dict(status=result['status'],result_path=str(path),selected_candidate=result['selected_candidate'],activation_reverted=True))
            report('completed','native096-hpo500-boundary-complete','Prospective same-formulation500step kappa boundary investigation completed with all negative outcomes retained and activation reverted.',dict(result_path=str(path),selected_candidate=result['selected_candidate'],joint_improvement_confirmed=result['joint_improvement_confirmed'],new_actual_GPU_seconds=result['new_actual_GPU_seconds']))
        except BaseException as exc:
            write(STATE/'failure.json',dict(status='HPO500_BOUNDARY_STOPPED_RETAINING_ALL_EVIDENCE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_retry=False))
            report('failed','native096-hpo500-boundary-stopped','Kappa boundary queue stopped on an operational/audit issue; evidence retained, no automatic retry.',dict(error=repr(exc),failure_path=str(STATE/'failure.json')))
            raise
if __name__=='__main__':main()
