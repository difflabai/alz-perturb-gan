"""Run the frozen30-cell benchmark serially, retaining all outcomes and audits.

This is a batch coordinator for already qualified finite canonical admissions.
It never changes model settings, selects on test scores, retries a failed cell,
or alters a foreign process. Source changes/uncertain cleanup stop the queue.
"""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'external_core_runtime'
PROTOCOL=ROOT/'analysis/external_benchmark_frozen_candidate_v096_20261105.json'
PROTOCOL_SHA='7eb3efa1ed4d4dcc70de0fe08f94dfb18754b2d028da2b72d0dc5d59ca097856'
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
PG=PROJECT/'current_base_20261029/runtime/bin/python'
SCVI=SCIENCE.parent/'isolated/cfgen-cu124-py311/bin/python'
STATE=ROOT/'artifacts/external-core-b1-batch';ACTIVATION='ab67776189fc5f99269274a67a14910d8a9dc230'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def matrix():return [(fold,method,seed) for fold in range(3) for seed in range(5) for method in ('pg','scvi')]
def attempt(fold,method,seed):return f'CAL_GPU0_V1_external_b1_f{fold}_{method}_s{seed}'
def audit_path(value):return ROOT/'analysis'/(attempt(*value)+'-external-audit.json')
def validate_audit(q,value):
    fold,method,seed=value
    assert q['status']=='EXTERNAL_CORE_NATIVE_CHECKPOINT_COUNTS_SCORES_CLEAN_SETTLEMENT_AUDIT_PASS'
    assert q['cell']==dict(experiment='external_b1',fold=fold,method=method,seed=seed)
    assert q['benchmark_protocol_SHA256']==PROTOCOL_SHA
    for k in ('own_processes_absent','owned_registry_lease_absent','independent_CPU_saved_score_audit','native_saved_state_matches_GPU_restore'):assert q[k] is True
    c=q['completed'];assert c['status']=='completed' and c['metric_role']=='final_evaluation'
    assert c['endpoint']==(20000 if method=='pg' else 400)
    assert c['new_model_updates']==(0 if method=='scvi' and fold==seed==0 else c['endpoint'])
    assert c['PG_configuration']==read(PROTOCOL)['PG']['hyperparameters'] if method=='pg' else c['PG_configuration'] is None
    return True
def run(command,*,timeout=None):
    p=subprocess.run(list(map(str,command)),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError(f'Command failed ({p.returncode}): {command[0]}: {p.stderr[-2500:]}')
    return p
def reap_settled_guard(identity):
    """Reap an exited direct child so its zombie cannot stall a settled cell."""
    path=Path('/proc',str(identity['pid']),'stat')
    if not path.exists():return True
    stat=path.read_text().rsplit(')',1)[1].split()
    if stat[19]!=str(identity['birth']):return True
    if identity['boot']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():return True
    try:os.waitpid(identity['pid'],os.WNOHANG)
    except ChildProcessError:pass
    return not path.exists()
def commit(paths,message):
    run(['git','add','--',*map(str,paths)]);run(['git','commit','-m',message])
def report(kind,event,summary,details):
    request=dict(recipient='bits255.dots.ada_20261002',conversation_id='report_auto_a20a79ed947ac6c51b74dbd821b796e2_ember',task_ref='01a1a22b-f310-7190-a7c6-8697ecabae6c',event_ref=event,kind=kind,summary=summary,details=details)
    helper=['/mnt/nvme/ai/conda/anaconda3/bin/python3','/home/mikkel/.local/lib/agent-babel/account_session_helper.py','tool','--client','codex','--account-file','/home/mikkel/.config/255bits/agent-babel/trusted-account.json','--native-session','01a1a22b-f310-7190-a7c6-8697ecabae6c','--tool','stage_report']
    p=subprocess.run(helper,input=json.dumps(request),capture_output=True,text=True,cwd=ROOT,timeout=30)
    (STATE/(event+'-report.stdout')).write_text(p.stdout);(STATE/(event+'-report.stderr')).write_text(p.stderr)
def start(value,folder):
    dep=read(folder/'deployment.json');auth=read(dep['authorization_path']);exe=PG if value[1]=='pg' else SCVI
    assert sha(dep['authorization_path'])==dep['authorization_SHA256'] and time.time()<auth['expires_unix']-150
    assert not Path(auth['ledger_path']).exists() and not Path(auth['output_dir']).exists()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (folder/'owner.stdout').open('xb') as out,(folder/'owner.stderr').open('xb') as err:
        owner=subprocess.Popen([str(exe),'-I','-S','-B',str(RUNTIME/'launch_core_gpu0.py'),'--authorization',dep['authorization_path']],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    write(folder/'launch_receipt.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_pid=owner.pid,owner_birth=int(Path(f'/proc/{owner.pid}/stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),attempt=attempt(*value),source_commit=dep['candidate_commit'],driver_managed_post_settlement_audit=True))
def wait_and_audit(value,folder):
    dep=read(folder/'deployment.json');auth=read(dep['authorization_path']);ledger=Path(auth['ledger_path'])
    a=audit_path(value);settlement=ledger.with_name(attempt(*value)+'-settlement.json')
    while not a.exists():
        if settlement.exists():
            s=read(settlement);assert s['uncertain'] is None and s['settlement']['status']=='settled','Uncertain settlement; queue stopped'
            jobs=read(ledger)['jobs'];assert len(jobs)==1
            assert jobs[0]['status']=='completed' and jobs[0]['exit_code']==0 and jobs[0]['reservation_released'],'Scientific cell did not complete; queue stopped'
            # The separately started first-cell audit waiter owns its own audit.
            if (folder/'wait_for_audit.py').exists():
                p=folder/'audit_waiter_result.json'
                if p.exists():assert read(p)['audit_exit_code']==0,'Independent audit failed'
            else:
                exe=PG if value[1]=='pg' else SCVI
                # Allow the owned guardian to exit after writing settlement.
                if not reap_settled_guard(jobs[0]['guard_identity']):time.sleep(1);continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                with (folder/'audit.stdout').open('xb') as out,(folder/'audit.stderr').open('xb') as err:
                    p=subprocess.run([str(exe),'-I','-S','-B',str(RUNTIME/'finish.py'),'--attempt',attempt(*value)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=600)
                assert p.returncode==0,'Independent audit failed; inspect retained audit.stderr'
        if time.time()>auth['expires_unix']+660:raise TimeoutError('Cell/audit exceeded original finite deadline; queue stopped')
        time.sleep(5)
    q=read(a);validate_audit(q,value);return q
def aggregate():
    inputs=[];matrix_root=STATE/'matrix';matrix_root.mkdir(exist_ok=True)
    for value in matrix():
        path=audit_path(value);assert path.exists(),'Full matrix missing audited cell'
        q=read(path);validate_audit(q,value);inputs.append(path)
        for p,pin in q['files_SHA256'].items():assert sha(p)==pin
        f,m,s=value;link=matrix_root/f'f{f}_{m}_s{s}';target=ROOT/'runs'/attempt(*value)
        if link.exists():assert link.resolve()==target
        else:link.symlink_to(target,target_is_directory=True)
    sys.path.insert(0,str(SCIENCE));from aggregate import aggregate as original_aggregate
    report_dir=STATE/'complete-report';assert not report_dir.exists()
    q=original_aggregate(matrix_root,report_dir)
    q.update(benchmark_protocol_SHA256=PROTOCOL_SHA,ParticleGAN_version='0.9.6',all30_native_cells_independently_audited=True,
             external_CFGen_full_matrix_complete=False,scDesign3_full_matrix_complete=False,newer_comparator_coverage_complete=False,
             SOTA_claim=False,interpretation='Conditional historically exposed57donor cohort; first frozen3fold5seed PG/scVI/resampling benchmark. Additional current external methods and independent cohort validation required for a broad SOTA claim.',
             audit_SHA256={str(p):sha(p) for p in inputs})
    dest=ROOT/'analysis/external_core096_complete_report.json';assert not dest.exists();write(dest,q)
    commit([dest], 'Record complete frozen ParticleGAN096 external core benchmark')
    return q
def main():
    assert sha(PROTOCOL)==PROTOCOL_SHA
    assert len(matrix())==len(set(matrix()))==30
    STATE.mkdir(parents=True,exist_ok=True)
    lock=STATE/'driver.lock';import fcntl
    with lock.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(STATE/'driver_identity.json',dict(pid=os.getpid(),birth=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=sha(__file__),protocol_SHA256=PROTOCOL_SHA,finite_cells=30))
        completed=[]
        try:
            for value in matrix():
                folder=ROOT/'artifacts/allocations'/attempt(*value)
                if audit_path(value).exists():
                    validate_audit(read(audit_path(value)),value)
                else:
                    if not folder.exists():
                        assert not run(['git','status','--porcelain']).stdout,'Uncommitted work; queue stopped without touching it'
                        result=run([sys.executable,'-I','-S','-B',RUNTIME/'prepare.py','--fold',str(value[0]),'--method',value[1],'--seed',str(value[2])],timeout=120)
                        (STATE/(attempt(*value)+'-prepare.stdout')).write_text(result.stdout)
                    if not (folder/'launch_receipt.json').exists():start(value,folder)
                    wait_and_audit(value,folder)
                path=audit_path(value)
                if run(['git','status','--porcelain','--',path]).stdout:commit([path],f'Record independently audited external benchmark {attempt(*value)}')
                completed.append(value)
                write(STATE/'progress.json',dict(status='FROZEN_MATRIX_IN_PROGRESS',audited_cells=len(completed),matrix_cells=30,completed_cells=completed,last_audit=str(path),quality_selection_performed=False))
            q=aggregate()
            run(['git','revert','--no-edit',ACTIVATION])
            write(STATE/'progress.json',dict(status='COMPLETE_FROZEN_CORE_MATRIX_AUDITED',audited_cells=30,report='/ml2/alz/analysis/external_core096_complete_report.json',advance_unchanged_recipe=q['advance_unchanged_recipe'],SOTA_claim=False))
            report('completed','external-core096-full30-complete','Frozen ParticleGAN0.9.6 full30cell PG/scVI/resampling benchmark completed and independently audited. SOTA claim withheld pending broader comparators/cohort validation.',json.dumps(dict(report='/ml2/alz/analysis/external_core096_complete_report.json',mean_energy_score=q['mean_energy_score'],rules=q['advancement_rules'],activation_reverted=True)))
        except BaseException as exc:
            write(STATE/'failure.json',dict(status='BATCH_STOPPED_PRESERVING_ALL_EVIDENCE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),audited_cells=len(completed),failed_or_pending_cell=value if 'value' in locals() else None,automatic_retry=False,automatic_promotion=False))
            report('failed','external-core096-full30-driver-stopped','Frozen external benchmark queue stopped; retained completed cells and failure evidence. No automatic retry or model promotion.',repr(exc)+'; inspect /ml2/alz/artifacts/external-core-b1-batch/failure.json')
            raise
if __name__=='__main__':main()
