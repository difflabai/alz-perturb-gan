"""ALZ-owned finite parallel runner; original guard semantics plus exact owned cleanup on foreign departure."""
from __future__ import annotations
import inspect,json,math,os,signal,subprocess,sys,time
from pathlib import Path
import gpu1_admission as admission_source
from gpu1_admission import Registry,GPU0,GiB,Conflict,identity,matches,atomic,snapshot,proc_table,protected_identity,Uncertain

def gate_spawn(command,environment,lease,observe):
    """Child cannot import model/CUDA until its session is registered."""
    rd,wr=os.pipe();worker=None
    code="""import os,sys,ctypes,signal
parent=int(sys.argv[2]);libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
if os.getppid()!=parent:raise SystemExit(125)
fd=int(sys.argv[1]);ok=os.read(fd,1);os.close(fd)
if ok!=b'1':raise SystemExit(125)
os.execvpe(sys.argv[3],sys.argv[3:],os.environ)
"""
    try:
        worker=subprocess.Popen([sys.executable,'-I','-S','-B','-c',code,str(rd),str(os.getpid()),*map(str,command)],env=environment,
                                start_new_session=True,pass_fds=(rd,*getattr(lease,'fds',lambda:[])()))
        os.close(rd);rd=None
        w=identity(worker.pid)
        if w is None:raise Conflict('Worker disappeared before launch barrier')
        lease.attach(w,observe());os.write(wr,b'1')
        return worker,w
    except BaseException:
        # EOF aborts only this not-yet-admitted child; no CUDA was released.
        if worker is not None:
            os.close(wr);wr=None
            try:worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if identity(worker.pid) is not None:worker.kill();worker.wait()
        raise
    finally:
        if rd is not None:os.close(rd)
        if wr is not None:os.close(wr)

def cleanup_owned_targets(lease,obs):
    """Prove OWN targets even after foreign peer departure/reuse.

    This never admits a missing/changed peer: original registry health still
    refuses. Foreign PID numbers remain excluded, even if their births change.
    A live owned group must contain only current worker descendants; no broad
    session adoption and no reliance on foreign peer survival for cleanup.
    """
    lease.check_physical_locks()
    procs=obs['procs'];protected={p['identity']['pid'] for p in lease.registry.protected_peers}
    with lease.registry.transaction() as state:
        job=lease.own_job(state);members=lease.registry.membership(job,obs);w=job.get('worker')
        if protected & members:raise Conflict('Protected PID intersects owned cleanup tree')
        if not w:return []
        if matches(w,procs):
            group={p for p,v in procs.items() if v['pgid']==w['pid']}
            if group-members or group & protected:raise Conflict('Unowned/protected process shares owned group; no group signal')
            return [{'pgid':w['pid'],'leader':w}]
        return [{'pid':p,'identity':{k:procs[p][k] for k in ('pid','birth','boot')}} for p in sorted(members)]

def signal_owned(lease,observe,sig):
    targets=cleanup_owned_targets(lease,observe())
    protected={p['identity']['pid'] for p in lease.registry.protected_peers}
    for target in targets:
        if target.get('pid',target.get('pgid')) in protected:raise Conflict('Never signal protected PID')
        if 'leader' in target:
            w=target['leader']
            if identity(w['pid'])!=w:raise Conflict('Owned leader changed before signal')
            # Fresh process-only view immediately before signalling, independent
            # of failed GPU telemetry and departed/reused foreign identities.
            current={'procs':proc_table()}
            if cleanup_owned_targets(lease,current)!=targets:raise Conflict('Owned group targets changed before signal')
            if identity(w['pid'])!=w:raise Conflict('Owned leader changed at signal boundary')
            os.killpg(target['pgid'],sig)
        else:
            if identity(target['pid'])!=target['identity']:raise Conflict('Owned descendant changed before signal')
            os.kill(target['pid'],sig)
    return targets

def oom_count():
    return int(next(l.split()[1] for l in Path('/proc/vmstat').read_text().splitlines() if l.startswith('oom_kill ')))
def full_pressure():
    fields=dict(t.split('=') for t in next(l for l in Path('/proc/pressure/memory').read_text().splitlines() if l.startswith('full ')).split()[1:])
    return float(fields['avg10'])

def run(ledger_path,gpu_uuid,label,command,*,cap_seconds=3600,policy_path,role,protect=lambda:None,
        absolute_deadline=None,observe=None,poll_seconds=2,expected_policy_sha256=None,test_mode=False):
    """Wall-clock budget includes model startup, qualification, eval and cleanup.

    Per-job ledger lock prevents duplicate launches. Legacy guard source and
    all earlier ledgers stay unchanged. Runtime RSS over reservation is reported,
    not a kill rule; actual OOM/sustained pressure retains existing policy.
    """
    import fcntl
    if gpu_uuid!=GPU0:raise Conflict('This isolated candidate is GPU0 only')
    if not 0<cap_seconds<=21600:raise ValueError('Invalid finite guard budget')
    if type(poll_seconds) not in (int,float) or not math.isfinite(poll_seconds) or not 0<poll_seconds<=2:raise ValueError('Finite positive health polling interval<=2seconds required')
    policy_path=Path(policy_path)
    import hashlib
    raw_policy=policy_path.read_bytes()
    if not expected_policy_sha256 or hashlib.sha256(raw_policy).hexdigest()!=expected_policy_sha256:raise Conflict('Reviewed exact GPU0 policy hash required')
    cfg=json.loads(raw_policy)
    if observe is not None and not test_mode:raise Conflict('Injected telemetry is fixture-only')
    if test_mode and (observe is None or cfg.get('status')!='PREPARED_FIXTURE_ONLY'):raise Conflict('Explicit synthetic telemetry and fixture policy required')
    if not test_mode and (type(absolute_deadline) not in (int,float) or not math.isfinite(absolute_deadline) or absolute_deadline<=time.time()):raise Conflict('Finite future scheduler expiry required')
    if test_mode:
        root=policy_path.parent.resolve()
        if root.name!='fixture' or not (root.is_relative_to(Path('/tmp')) or root.is_relative_to(Path(__file__).resolve().parent)):raise Conflict('Fixture root must be task-local or tmp and explicitly named fixture')
        paths=[policy_path,Path(ledger_path),Path(ledger_path).with_suffix('.lock'),Path(cfg['registry_path']),*map(Path,cfg['locks'])]
        if any(not p.resolve().is_relative_to(root) for p in paths):raise Conflict('All fixture policy/registry/ledger/lock paths must resolve inside fixture root')
        physical=[Path('/mnt/nvme/sliders')/part for part in ['qualification/admission-gpu-1.lock','locks/gpu-1.lock','qualification/admission-gpu-1.lock','locks/gpu-1.lock']]
        forbidden={(p.stat().st_dev,p.stat().st_ino) for p in physical if p.exists()}
        if any(p.exists() and (p.stat().st_dev,p.stat().st_ino) in forbidden for p in paths):raise Conflict('Fixture alias of physical GPU lock refused')
    if not test_mode and tuple(cfg['locks'])!=('/mnt/nvme/sliders/qualification/admission-gpu-1.lock','/mnt/nvme/sliders/locks/gpu-1.lock'):raise Conflict('Exact physical GPU0 lock pair required')
    admission_path=Path('/ml2/alz/recovery_runtime/gpu1_admission.py')
    required={str(Path(__file__).resolve()),str(admission_path)}
    if not required<=set(cfg.get('source_sha256',{})):raise Conflict('Both active GPU0 source files must be pinned')
    if cfg['gpu']!=gpu_uuid:raise Conflict('Policy GPU mismatch')
    if Path(admission_source.__file__).resolve()!=admission_path or any(Path(inspect.getfile(value)).resolve()!=admission_path for value in [Registry,identity,atomic,snapshot,proc_table,protected_identity]):raise Conflict('Imported active admission source origin differs')
    original_protect=protect
    def protect():
        if hashlib.sha256(policy_path.read_bytes()).hexdigest()!=expected_policy_sha256:raise Conflict('Reviewed policy changed during owned campaign')
        for path,sha in cfg.get('source_sha256',{}).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise Conflict('Pinned source changed: '+path)
        original_protect()
    protect()
    role_cfg=cfg['roles'][role]
    registry=Registry(cfg['registry_path'],locks=cfg['locks'],roles=set(cfg['roles']),driver_headroom=cfg['driver_headroom_bytes'],gpu=GPU0,protected_peers=cfg['protected_peers'],host_free_floor=cfg['host_free_floor_bytes'],policy_sha256=expected_policy_sha256)
    native_observe=observe is None
    if native_observe and cfg.get('status')!='REVIEWED_GPU0_DEPLOYED':raise Conflict('Cooperative policy is prepared, not deployed; coordinate migration first')
    if observe is None:
        def observe():
            o=snapshot(gpu_uuid);o['protected_pids']=cfg.get('protected_host_peak_pids',[]);return o
    ledger_path=Path(ledger_path);ledger_path.parent.mkdir(parents=True,exist_ok=True)
    with ledger_path.with_suffix('.lock').open('a+') as guard_lock:
        fcntl.flock(guard_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'schema':1,'gpu_uuid':gpu_uuid,'cap_seconds':cap_seconds,'jobs':[]}
        if ledger['gpu_uuid']!=gpu_uuid or ledger['cap_seconds']!=cap_seconds:raise Conflict('Ledger GPU/cap mismatch')
        if any(j['status']=='active' for j in ledger['jobs']):raise Conflict('Uncertain previous run; reconcile before retry')
        if ledger['jobs']:raise Conflict('Prior attempt exists; preserve ledger and require fresh reviewed allocation/output identity')
        used=sum(j['charged_seconds'] for j in ledger['jobs']);remaining=cap_seconds-used
        if remaining<=1:raise Conflict('Wall-clock budget exhausted')
        protect();owner=identity(os.getpid())
        lease=registry.reserve(role,owner,role_cfg,observe())
        started=time.monotonic();wall_start=time.time();deadline=started+remaining-1
        if absolute_deadline is not None:deadline=min(deadline,started+absolute_deadline-wall_start-1)
        worker=None;requested=[];terminated=False;reason='completed';code=1;unreleased=False
        job={'label':label,'role':role,'status':'active','started_unix':wall_start,'guard_pid':os.getpid(),
             'guard_identity':owner,'allowed_seconds':remaining,'charged_seconds':None,'lease_token':lease.token,
             'command':list(map(str,command)),'cooperative_registry':str(registry.path),'reservation':role_cfg}
        ledger['jobs'].append(job);atomic(ledger_path,ledger)
        handlers={s:signal.signal(s,lambda signum,frame:requested.append(signum)) for s in [signal.SIGINT,signal.SIGTERM]}
        class BudgetExpired(TimeoutError):pass
        def expire(*_):raise BudgetExpired('Finite wall-clock guard cleanup boundary')
        old_alarm=signal.signal(signal.SIGALRM,expire)
        initial_oom=None;pressure_since=None;last_observe=0;last_health=None;unknown_since=None
        try:
            signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-started-min(15,remaining/4)))
            initial_oom=oom_count() if native_observe else None
            if deadline-started<10:raise TimeoutError('Original deadline leaves no safe startup/cleanup window')
            env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='' if test_mode else gpu_uuid,
                     ALZ_BUDGET_LEDGER=str(ledger_path),ALZ_GPU_JOB_LABEL=label,
                     COOPERATIVE_GPU_ROLE=role,COOPERATIVE_GPU_TOKEN=lease.token)
            protect();worker,w=gate_spawn(command,env,lease,observe)
            job.update(child_pid=worker.pid,child_identity=w,process_group=worker.pid);atomic(ledger_path,ledger)
            while True:
                now=time.monotonic()
                if worker.poll() is not None:
                    # Descendants may outlive the worker leader. Reconcile only
                    # this job; a healthy registered peer is never teardown work.
                    o=observe()
                    if not cleanup_owned_targets(lease,o):break
                    if not terminated:reason='owned_descendant_cleanup'
                if now-last_observe>=poll_seconds:
                    protect();last_observe=now
                    try:
                        o=observe();last_health=lease.health(o);unknown_since=None
                    except (subprocess.TimeoutExpired,Uncertain) as exc:
                        lease.note_telemetry_unknown();unknown_since=unknown_since or now
                        atomic(ledger_path.with_name('telemetry_unknown.json'),{'checked_unix':time.time(),'error':repr(exc),'no_new_admission':True})
                        if now-unknown_since>90:raise Uncertain('Telemetry unknown beyond existing90-second grace') from exc
                    atomic(ledger_path.with_name('cooperative_live.json'),{'checked_unix':time.time(),'elapsed_seconds':now-started,**(last_health or {})})
                    if last_health is not None and last_health['own_gpu_budget_exceeded']:raise MemoryError('Owned GPU budget exceeded')
                    if native_observe:
                        if oom_count()!=initial_oom:raise MemoryError('Actual host OOM observed')
                        psi=full_pressure();pressure_since=(pressure_since or now) if psi>=10 else None
                        if pressure_since is not None and now-pressure_since>=30:raise MemoryError('Actual full memory PSI>=10 sustained30s')
                if not terminated and (requested or now>=deadline-10):
                    reason='interrupted' if requested else 'budget_exhausted'
                    signal_owned(lease,observe,signal.SIGTERM);terminated=True
                if now>=deadline:
                    signal_owned(lease,observe,signal.SIGKILL);break
                time.sleep(min(.1,max(.001,deadline-now)))
            code=worker.wait(timeout=1)
            if reason=='completed' and code:reason='failed'
        except BaseException as exc:
            reason='budget_exhausted' if isinstance(exc,BudgetExpired) else 'failed';job['error']=repr(exc)
            if worker is not None and worker.poll() is None:
                # A telemetry timeout cannot prevent owned-session teardown.
                # Fast process-only identity checks use no NVML subprocess.
                def process_only():return {'procs':proc_table()}
                signal_owned(lease,process_only,signal.SIGTERM)
                try:worker.wait(timeout=max(.01,min(3,deadline-time.monotonic())))
                except subprocess.TimeoutExpired:signal_owned(lease,process_only,signal.SIGKILL);worker.wait(timeout=2)
            raise
        finally:
            # Retain a finite cleanup alarm, including bounded telemetry helpers.
            signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.monotonic()))
            # GPU cleanup is verified against this session only. Registry
            # reservations survive uncertain outcomes for explicit recovery.
            try:
                for _ in range(5):
                    o=observe()
                    try:lease.release(o);break
                    except Conflict:
                        if time.monotonic()>=deadline:raise
                        signal_owned(lease,observe,signal.SIGKILL);time.sleep(.1)
                else:raise Conflict('Owned cleanup remains uncertain')
            except BaseException as exc:
                unreleased=True;job['release_error']=repr(exc);reason='cleanup_uncertain'
            finally:
                signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm)
            elapsed=time.monotonic()-started
            job.update(status=reason,exit_code=code,ended_unix=time.time(),charged_seconds=elapsed,
                       reservation_released=not unreleased,last_health=last_health)
            ledger['aggregate_charged_seconds']=used+elapsed;atomic(ledger_path,ledger)
            for s,h in handlers.items():signal.signal(s,h)
        return code if reason=='completed' else (code or 124)
