"""One finite duration experiment after the current cell, then resume frozen matrix.

Only the exact owned Python queue supervisor is stopped. Its independent
scientific guardian/worker continues to its original endpoint and deadline.
No test score is used to alter the frozen matrix or choose another checkpoint.
"""
import argparse,datetime,fcntl,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');STATE=ROOT/'artifacts/duration096-queue';RUNTIME=ROOT/'duration096_runtime'
CORE_STATE=ROOT/'artifacts/external-core-b1-batch'
ATTEMPT='CAL_GPU0_V1_duration096_f0_pg_s0'
PG=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/bin/python')
def read(p):return json.loads(Path(p).read_bytes())
def write(p,q):
    p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(q,indent=2)+'\n');tmp.replace(p)
def identity(pid):
    p=Path('/proc',str(pid),'stat')
    if not p.exists():return None
    parts=p.read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,birth=int(parts[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=parts[0])
def same(actual,expected):return actual is not None and all(actual[k]==expected[k] for k in ('pid','birth','boot'))
def stop_exact_supervisor(expected):
    # pidfd prevents PID reuse between the identity check and signal delivery.
    fd=os.pidfd_open(expected['pid'])
    try:
        assert same(identity(expected['pid']),expected),'Queue identity changed; no signal sent'
        signal.pidfd_send_signal(fd,signal.SIGTERM)
    finally:os.close(fd)
def load_core():
    path=ROOT/'analysis/external_benchmark_driver.py'
    spec=importlib.util.spec_from_file_location('frozen_core_batch',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def resume_core(core):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'Dirty worktree; do not resume'
    # Starting the original coordinator preserves its frozen order, protocol,
    # completed audits, original per-cell one-use admissions and final report.
    with (STATE/'resumed_core.stdout').open('xb') as out,(STATE/'resumed_core.stderr').open('xb') as err:
        proc=subprocess.Popen([sys.executable,'-I','-S','-B',str(ROOT/'analysis/external_benchmark_driver.py')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    write(STATE/'resumed_core_identity.json',identity(proc.pid));return proc.pid
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--activation',required=True);args=parser.parse_args()
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'driver.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (STATE/'handoff.json').exists(),'One-use handoff already recorded; inspect rather than restart'
        core=load_core();old=read(CORE_STATE/'driver_identity.json')
        assert same(identity(old['pid']),old) and old['source_SHA256']==core.sha(core.__file__)
        assert not (CORE_STATE/'failure.json').exists()
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==args.activation
        protocol=read(ROOT/'analysis/duration096/protocol.json')
        # Resolve the one active cell before taking over scheduling; it is never
        # signalled, retrained, skipped or given a new scientific deadline.
        active=[]
        for value in core.matrix():
            folder=ROOT/'artifacts/allocations'/core.attempt(*value)
            if folder.exists() and not core.audit_path(value).exists():active.append((value,folder))
        assert len(active)==1,'Expected exactly one active frozen cell'
        value,folder=active[0];auth=read(folder/'authorization.json')
        assert time.time()<auth['expires_unix'] and value==(0,'pg',1)
        assert read(RUNTIME/'plan.json')['enabled'] is True
        stop_exact_supervisor(old)
        handoff=dict(status='CURRENT_FROZEN_CELL_UNINTERRUPTED_DURATION_NEXT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_supervisor=old,active_cell=list(value),active_authorization_SHA256=core.sha(folder/'authorization.json'),activation=args.activation,no_scientific_process_signalled=True,frozen_matrix_changed=False)
        write(STATE/'handoff.json',handoff);write(STATE/'driver_identity.json',identity(os.getpid()))
        write(STATE/'progress.json',dict(status='WAITING_FOR_CURRENT_FROZEN_CELL_ORIGINAL_ENDPOINT',cell=list(value),duration_started=False))
        try:
            q=core.wait_and_audit(value,folder)
            core.commit([core.audit_path(value)],f'Record independently audited external benchmark {core.attempt(*value)}')
            write(STATE/'progress.json',dict(status='CURRENT_CELL_AUDITED_PREPARING_MATCHED_DURATION',cell=list(value)))
            result=core.run([sys.executable,'-I','-S','-B',RUNTIME/'prepare.py'],timeout=120)
            (STATE/'prepare.stdout').write_text(result.stdout)
            folder=ROOT/'artifacts/allocations'/ATTEMPT;dep=read(folder/'deployment.json');auth=read(dep['authorization_path'])
            with (folder/'owner.stdout').open('xb') as out,(folder/'owner.stderr').open('xb') as err:
                owner=subprocess.Popen([str(PG),'-I','-S','-B',str(RUNTIME/'launch_core_gpu0.py'),'--authorization',dep['authorization_path']],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            write(folder/'launch_receipt.json',dict(identity=identity(owner.pid),authorization_SHA256=dep['authorization_SHA256'],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            write(STATE/'progress.json',dict(status='MATCHED_DURATION_NATIVE_CONTINUATION_ACTIVE',attempt=ATTEMPT,initial_updates=20000,endpoint=40000))
            ledger=Path(auth['ledger_path']);settlement=ledger.with_name(ATTEMPT+'-settlement.json')
            while not settlement.exists():
                assert time.time()<auth['expires_unix']+660,'Finite duration phase exceeded settlement interval'
                time.sleep(5)
            # Reap this direct child before an independent process-absence audit.
            owner.wait(timeout=60)
            s=read(settlement);job=read(ledger)['jobs'][0]
            assert s['uncertain'] is None and s['settlement']['status']=='settled'
            assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released']
            audit=core.run([str(PG),'-I','-S','-B',RUNTIME/'finish.py','--attempt',ATTEMPT],timeout=600)
            (STATE/'audit.stdout').write_text(audit.stdout)
            path=ROOT/'analysis'/(ATTEMPT+'-duration-audit.json');result=read(path)
            core.commit([path],'Record matched native096 20k-to40k calibration duration result')
            core.run(['git','revert','--no-edit',args.activation])
            pid=resume_core(core)
            write(STATE/'progress.json',dict(status='DURATION_COMPLETE_AUDITED_ORIGINAL_BENCHMARK_RESUMED',result=str(path),difference40k_minus20k=result['difference40k_minus20k'],joint_improvement=result['joint_improvement'],core_supervisor_pid=pid,frozen_matrix_changed=False))
            core.report('completed','matched-native096-duration-completed','Matched non-EMA native096 duration experiment completed; same-model20k/40k calibration scores audited. Original frozen external benchmark resumed unchanged.',json.dumps(dict(result=str(path),difference40k_minus20k=result['difference40k_minus20k'],joint_improvement=result['joint_improvement'],activation_reverted=True)))
        except BaseException as exc:
            write(STATE/'failure.json',dict(status='DURATION_QUEUE_STOPPED_EVIDENCE_RETAINED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_retry=False,no_foreign_processes_signalled=True))
            core.report('failed','matched-native096-duration-stopped','Matched native096 duration queue stopped; no automatic retry, model promotion or frozen benchmark change.',repr(exc)+'; inspect /ml2/alz/artifacts/duration096-queue/failure.json')
            raise
if __name__=='__main__':main()
