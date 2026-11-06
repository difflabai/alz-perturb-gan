"""Finite, reproducible500-step HPO with full calibration and fresh-seed checks."""
import argparse,datetime,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/hpo500';STATE=ROOT/'artifacts/hpo500-r3'
RUNTIME=ROOT/'hpo500_r3_runtime'
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
    assert q['protocol_SHA256']==sha(STUDY/'protocol.json') and q['proposal_SHA256']==sha(STUDY/'proposals'/(cid+'.json'))
    assert q['owned_processes_and_lease_absent'] and q['clean_settlement'] and q['calibration_only']
    for path,pin in q.get('files_SHA256',{}).items():assert sha(path)==pin
def wait_for_capacity(cid,seed,phase,resume):
    # No allocation, lease, model import, source change, or trial clock reset.
    deadline=read(STUDY/'protocol.json')['research_window_path']
    expiry=read(deadline)['expires_unix']
    if resume:expiry=min(expiry,read(ROOT/'analysis/hpo500_recovery_r3/parent178_CPU_audit.json')['parent_expiry_unix'])
    args=[sys.executable,'-I','-S','-B',RUNTIME/'prepare.py','--candidate',cid,'--seed',seed,'--phase',phase,'--check-only']+(['--resume'] if resume else [])
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
    resume=(cid=='c008' and seed==0 and phase=='search')
    physical=f'CAL_GPU0_V1_h5_{cid}_{phase}'+('_r1' if resume else '')+f'_f0_pg_s{seed}'
    folder=ROOT/'artifacts/allocations'/physical
    if not folder.exists():
        assert not command(['git','status','--porcelain']),'Uncommitted work; study stops without touching it'
        wait_for_capacity(cid,seed,phase,resume)
        text=command([sys.executable,'-I','-S','-B',RUNTIME/'prepare.py','--candidate',cid,'--seed',seed,'--phase',phase]+(['--resume'] if resume else []));(STATE/(key(cid,seed,phase)+'-prepare.stdout')).write_text(text)
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
def final_report(search_results,finalists,confirmation):
    import numpy as np
    by={(r['candidate_id'],r['seed']):r for r in confirmation}
    comparisons=[];rng=np.random.default_rng(20261106)
    indices=rng.integers(0,4,size=(10000,4))
    control_ok=all(by[('c000',s)]['finite'] for s in (1,2,3,4))
    for cid in finalists['candidates']:
        if cid=='c000':continue
        rows=[by[(cid,s)] for s in (1,2,3,4)]
        if not control_ok or not all(r['finite'] for r in rows):
            comparisons.append(dict(candidate=cid,eligible=False,reason='Nonfinite candidate or matched control seed retained; no seed dropping'));continue
        controls=[by[('c000',s)] for s in (1,2,3,4)]
        es=np.array([r['metrics']['energy_score']-c['metrics']['energy_score'] for r,c in zip(rows,controls)])
        ed=np.array([r['metrics']['cell_energy_distance']-c['metrics']['cell_energy_distance'] for r,c in zip(rows,controls)])
        spread=float(np.mean([r['metrics']['spread_ratio'] for r in rows]))
        eligible=bool(es.mean()<0 and ed.mean()<=0 and .7<=spread<=1.3 and (es<0).sum()>=3 and not finalists['exploratory_without_feasible_seed0_candidate'])
        comparisons.append(dict(candidate=cid,eligible=eligible,mean_energy_score=float(np.mean([r['metrics']['energy_score'] for r in rows])),mean_cell_energy_distance=float(np.mean([r['metrics']['cell_energy_distance'] for r in rows])),mean_spread_ratio=spread,paired_seed_ES_deltas=es.tolist(),paired_seed_cellED_deltas=ed.tolist(),mean_ES_delta=float(es.mean()),mean_cellED_delta=float(ed.mean()),ES_delta95percent_seed_bootstrap_interval=np.percentile(es[indices].mean(1),[2.5,97.5]).tolist(),seeds_improved=int((es<0).sum())))
    eligible=sorted([c for c in comparisons if c['eligible']],key=lambda c:(c['mean_energy_score'],c['mean_cell_energy_distance'],c['candidate']))
    selected=eligible[0]['candidate'] if eligible else 'c000'
    return dict(status='COMPLETE_BOUNDED500STEP_HYPERPARAMETER_SEARCH_AND_CONFIRMATION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_version='0.9.6',EMA_decay=0,formulation_changed=False,endpoint=500,search_candidates=len(search_results),fresh_confirmation_seeds=[1,2,3,4],comparisons=comparisons,selected_candidate=selected,selected_hyperparameters=read(STUDY/'proposals'/(selected+'.json'))['hyperparameters'],joint_improvement_confirmed=bool(eligible),control_failed_on_any_confirmation_seed=not control_ok,automatic_incumbent_promotion=False,heldout_results_used=False,limitation='Best found in this bounded adaptive4donor calibration search, not a global optimum, independent cohort confirmation, or SOTA result. Seed bootstrap is conditional on these same calibration donors.',total_actual_GPU_seconds=sum(r['charged_GPU_seconds'] for r in search_results+confirmation),protocol_SHA256=sha(STUDY/'protocol.json'),result_SHA256={r['attempt']:sha(STUDY/'results'/(r['attempt']+'.json')) for r in search_results+confirmation})
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--activation',required=True);parser.add_argument('--resource-activation',required=True);parser.add_argument('--prior-resource-activation',required=True);args=parser.parse_args()
    STATE.mkdir(exist_ok=True);(STUDY/'results').mkdir(exist_ok=True)
    with (STATE/'driver.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (STATE/'failure.json').exists(),'Prior failure requires inspection; no automatic retry'
        write(STATE/'driver_identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=sha(__file__),activation=args.activation,resource_activation=args.resource_activation,prior_resource_activation=args.prior_resource_activation))
        try:
            search=import_search();records=[]
            for i in range(25):
                proposal=search.make_proposal(i,records);path=STUDY/'proposals'/(proposal['id']+'.json')
                if path.exists():assert read(path)==proposal,'Immutable prior proposal differs'
                else:
                    write(path,proposal);commit([str(path)],f'Declare adaptive500step scalar-only proposal {proposal["id"]}')
                q=run_cell(proposal['id'],0,'search');records.append(q)
                if i==0:assert q['finite'],'Control nonfinite; inspect before optimization'
                write(STATE/'search_summary.json',dict(completed=len(records),maximum=25,endpoint=500,rows=[{k:r[k] for k in ('candidate_id','finite','metrics','charged_GPU_seconds')} for r in records]))
            expected=search.shortlist(records)
            shortlist=STUDY/'finalists.json'
            if shortlist.exists():assert read(shortlist)==expected
            else:write(shortlist,expected);commit([str(shortlist)],'Freeze500step finalists before four fresh confirmation seeds')
            confirmation=[]
            for seed in (1,2,3,4):
                for cid in expected['candidates']:confirmation.append(run_cell(cid,seed,'confirm'))
            result=final_report(records,expected,confirmation);path=STUDY/'complete_report.json';assert not path.exists();write(path,result)
            goal=read(ROOT/'analysis/current_research_goal.json');goal['study_status']=result['status'];goal['result_path']=str(path);goal['joint_improvement_confirmed']=result['joint_improvement_confirmed'];write(ROOT/'analysis/current_research_goal.json',goal)
            commit([str(path),str(ROOT/'analysis/current_research_goal.json')],'Record complete fixed-formulation500step HPO and all seed comparisons')
            command(['git','revert','--no-edit',args.resource_activation])
            command(['git','revert','--no-edit',args.prior_resource_activation])
            command(['git','revert','--no-edit',args.activation])
            write(STATE/'progress.json',dict(status=result['status'],result_path=str(path),selected_candidate=result['selected_candidate'],joint_improvement_confirmed=result['joint_improvement_confirmed'],activation_reverted=True))
            report('completed','native096-hpo500-r3-complete','Fixed native096 non-EMA500step scalar hyperparameter search and fresh-seed confirmation completed; every outcome retained and activation reverted.',dict(result_path=str(path),selected_candidate=result['selected_candidate'],joint_improvement_confirmed=result['joint_improvement_confirmed'],total_actual_GPU_seconds=result['total_actual_GPU_seconds']))
        except BaseException as exc:
            write(STATE/'failure.json',dict(status='HPO500_STOPPED_RETAINING_ALL_EVIDENCE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_retry=False))
            report('failed','native096-hpo500-r3-stopped','500step HPO stopped on a concrete operational/audit issue; all evidence retained; no automatic retry or promotion.',dict(error=repr(exc),failure_path=str(STATE/'failure.json')))
            raise
if __name__=='__main__':main()
