"""Resource-based shared GPU leases, interoperable with existing exclusive guards.

ISOLATED GPU1 CANDIDATE; NOT DEPLOYED. Standard library only.
Pure observations are injectable for CPU tests. Unknown
allocations, uncertain telemetry and live orphan workers are never silently
adopted or evicted. A guardian may signal only its own verified session.
"""
from __future__ import annotations
import contextlib,ctypes,fcntl,hashlib,json,os,stat,subprocess,sys,time,uuid
from pathlib import Path

GiB=1024**3
GPU1='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'
LOCKS=('/mnt/nvme/sliders/qualification/admission-gpu-1.lock','/mnt/nvme/sliders/locks/gpu-1.lock')
class Conflict(RuntimeError):pass
class Capacity(Conflict):pass
class Uncertain(Conflict):pass

def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
        if s[0]=='Z':return None
        return {'pid':int(pid),'birth':int(s[19]),'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (FileNotFoundError,ProcessLookupError):return None

def proc_table():
    result={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            s=(p/'stat').read_text().rsplit(')',1)[1].split();ident=identity(int(p.name))
            if ident is None:continue
            rss=hwm=0
            for line in (p/'status').read_text().splitlines():
                k,_,v=line.partition(':')
                if k=='VmRSS':rss=int(v.split()[0])*1024
                if k=='VmHWM':hwm=int(v.split()[0])*1024
            detail=protected_identity(int(p.name))
            if detail is None:continue  # Ordinary process exit during observation.
            if any(detail[k]!=ident[k] for k in ('pid','birth','boot')):continue
            result[int(p.name)]={**detail,'pgid':int(s[2]),'sid':int(s[3]),'rss':rss,'hwm':hwm}
        except (FileNotFoundError,ProcessLookupError):continue
        except (PermissionError,OSError):
            # Kernel/foreign-user entries need not be readable. GPU allocation
            # PIDs must be observable individually or snapshot() fails closed.
            continue
    return result

def matches(saved,procs):
    return saved is not None and all(procs.get(saved['pid'],{}).get(k)==saved[k] for k in ('pid','birth','boot'))

def descendants(pid,procs):
    result={pid}
    while True:
        more={p for p,v in procs.items() if v['ppid'] in result}
        if more<=result:return result
        result|=more

def snapshot(gpu=GPU1,*,query_timeout=2):
    """Read both NVML process kinds; never create a CUDA context."""
    if gpu!=GPU1:raise Conflict('This separate candidate is GPU1 only')
    def query(args):return subprocess.check_output(['nvidia-smi',*args,'--format=csv,noheader,nounits'],text=True,timeout=query_timeout).strip()
    row=query(['-i',gpu,'--query-gpu=uuid,memory.total,memory.used,utilization.gpu'])
    observed,total,used,utilization=[x.strip() for x in row.split(',')]
    if observed!=gpu:raise Uncertain('GPU identity mismatch')
    # Isolate potentially blocking NVML FFI in a bounded read-only helper.
    helper='import importlib.util,json,sys;s=importlib.util.spec_from_file_location("gpu0_probe",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print(json.dumps(m.nvml_allocations(sys.argv[2])))'
    raw=json.loads(subprocess.check_output([sys.executable,'-I','-S','-B','-c',helper,str(Path(__file__).resolve()),gpu],text=True,timeout=query_timeout))
    allocations={int(pid):entry for pid,entry in raw.items()}
    procs=proc_table()
    if set(allocations)-set(procs):raise Uncertain('GPU PID identity not observable')
    mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:'))}
    return {'gpu':gpu,'total':int(total)*1024**2,'used':int(used)*1024**2,
            'utilization_percent_informational':utilization,
            'gpu_memory':{p:v['memory'] for p,v in allocations.items()},
            'gpu_kinds':{p:v['kinds'] for p,v in allocations.items()},
            'procs':procs,'available_ram':mem['MemAvailable'],'total_ram':mem['MemTotal'],'complete':True}

class NVMLProcess(ctypes.Structure):
    _fields_=[('pid',ctypes.c_uint),('memory',ctypes.c_ulonglong),('gpu_instance',ctypes.c_uint),('compute_instance',ctypes.c_uint)]

def nvml_allocations(gpu):
    lib=ctypes.CDLL('libnvidia-ml.so.1');handle=ctypes.c_void_p()
    if lib.nvmlInit_v2():raise Uncertain('NVML initialization failed')
    try:
        if lib.nvmlDeviceGetHandleByUUID(gpu.encode('ascii'),ctypes.byref(handle)):raise Uncertain('NVML UUID lookup failed')
        observed=ctypes.create_string_buffer(96)
        if lib.nvmlDeviceGetUUID(handle,observed,ctypes.c_uint(96)) or observed.value.decode('ascii')!=gpu:raise Uncertain('NVML UUID differs')
        result={}
        for kind,name in [('compute','nvmlDeviceGetComputeRunningProcesses_v2'),('graphics','nvmlDeviceGetGraphicsRunningProcesses_v2')]:
            query=getattr(lib,name);count=ctypes.c_uint(0);rc=query(handle,ctypes.byref(count),None)
            if rc==0 and count.value==0:continue
            if rc not in (0,7):raise Uncertain('NVML '+kind+' count failed')
            for attempt in range(3):
                if not 0<count.value<=4096:raise Uncertain('NVML count invalid')
                rows=(NVMLProcess*count.value)();capacity=len(rows);rc=query(handle,ctypes.byref(count),rows)
                if rc==7:continue
                if rc or count.value>capacity:raise Uncertain('NVML '+kind+' query failed')
                seen=set()
                for row in rows[:count.value]:
                    if not row.pid or row.pid in seen or row.memory==2**64-1 or row.gpu_instance!=0xffffffff or row.compute_instance!=0xffffffff:raise Uncertain('Ambiguous NVML process/memory/partition')
                    seen.add(row.pid);entry=result.setdefault(int(row.pid),{'memory':0,'kinds':[]})
                    entry['memory']=max(entry['memory'],int(row.memory));entry['kinds'].append(kind)
                break
            else:raise Uncertain('NVML process list did not stabilize')
        return result
    finally:
        if lib.nvmlShutdown():raise Uncertain('NVML shutdown failed')

IDENTITY_KEYS=('pid','birth','boot','uid','exe','argv_sha256','ppid')
def protected_identity(pid):
    value=identity(pid)
    if value is None:return None
    root=Path('/proc')/str(pid)
    status=(root/'status').read_text();value['uid']=int(next(l.split()[1] for l in status.splitlines() if l.startswith('Uid:')))
    value['exe']=os.readlink(root/'exe')
    value['argv_sha256']=hashlib.sha256((root/'cmdline').read_bytes()).hexdigest()
    value['ppid']=int((root/'stat').read_text().rsplit(')',1)[1].split()[1])
    if identity(pid)!={k:value[k] for k in ('pid','birth','boot')}:raise Uncertain('Protected identity changed during observation')
    return value

class Registry:
    def __init__(self,path,*,locks=LOCKS,gpu=GPU1,roles=None,driver_headroom=GiB,protected_peers=None,host_free_floor=0,policy_sha256,sharing=None,host_peak_identities=None):
        self.path=Path(path);self.locks=tuple(map(Path,locks));self.gpu=gpu
        self.roles=set(roles or ());self.driver_headroom=driver_headroom
        self.protected_peers=protected_peers or [];self.host_free_floor=host_free_floor
        if not isinstance(policy_sha256,str) or len(policy_sha256)!=64 or set(policy_sha256)-set('0123456789abcdef'):raise Conflict('Exact reviewed policy digest required')
        self.policy_sha256=policy_sha256
        self.sharing=dict(sharing) if sharing is not None else None
        self.host_peak_identities=list(host_peak_identities or [])
        if type(driver_headroom) is not int or driver_headroom<=0 or type(host_free_floor) is not int or host_free_floor<=0:raise Conflict('Explicit positive GPU driver and host admission headroom required')
        pids=[p['identity']['pid'] for p in self.protected_peers]
        if gpu!=GPU1 or not pids or len(pids)!=len(set(pids)):raise Conflict('Exact nonempty protected GPU1 identities required')
        if any(set(p['identity'])!=set(IDENTITY_KEYS) or not p.get('gpu_kinds') or len(p['gpu_kinds'])!=len(set(p['gpu_kinds'])) or set(p['gpu_kinds'])-{'compute','graphics'} or any(type(p.get(k)) is not int or p[k]<=0 for k in ('gpu_budget_bytes','ram_budget_bytes')) for p in self.protected_peers):raise Conflict('Explicit reviewed peer identities/kinds/growth budgets required')
        if self.sharing is not None:
            cfg=self.sharing
            if type(cfg.get('external_compute_budget_bytes')) is not int or not 0<cfg['external_compute_budget_bytes']<=15*GiB//4:raise Conflict('Shared external COMPUTE limit must have margin below4GiB')
            if cfg.get('minimum_available_ram_bytes')!=8*GiB or cfg.get('enforce_own_rss') is not True or cfg.get('enforce_host_growth') is not True or self.host_free_floor<8*GiB:raise Conflict('Explicit shared RSS and8GiB host-floor enforcement required')
            moss=cfg.get('moss_compute_pids',[])
            mode=cfg.get('admission_mode','with_moss')
            if mode=='free_card_exact_peers':
                if moss or any('compute' in p['gpu_kinds'] for p in self.protected_peers):raise Conflict('Free-card profile requires no protected COMPUTE peer or MOSS identity')
            elif mode=='with_moss':
                if not moss or len(moss)!=len(set(moss)) or not set(moss)<=set(pids) or any('compute' not in p['gpu_kinds'] for p in self.protected_peers if p['identity']['pid'] in moss):raise Conflict('Exact protected MOSS COMPUTE identities required')
            else:raise Conflict('Explicit supported GPU1 peer admission mode required')
        if any(set(v)!=set(IDENTITY_KEYS) for v in self.host_peak_identities) or len({v['pid'] for v in self.host_peak_identities})!=len(self.host_peak_identities):raise Conflict('Exact distinct host-peak identities required')
    def host_peak_growth(self,obs,own=()):
        protected={p['identity']['pid'] for p in self.protected_peers};growth=0
        for saved in self.host_peak_identities:
            current=obs['procs'].get(saved['pid'],{})
            if any(current.get(k)!=saved[k] for k in IDENTITY_KEYS):raise Conflict('Protected host-peak identity changed')
            if saved['pid'] not in own and saved['pid'] not in protected:growth+=max(0,current.get('hwm',current['rss'])-current['rss'])
        return growth
    def external_compute(self,state,obs,actual,prospective=0):
        if self.sharing is None:return {'sharing_active':False,'external_compute_budget_exceeded':False}
        moss=set(self.sharing['moss_compute_pids'])
        current=sum(used for pid,used in obs['gpu_memory'].items() if pid not in moss and 'compute' in obs['gpu_kinds'].get(pid,()))
        growth=sum(max(0,p['gpu_budget_bytes']-obs['gpu_memory'].get(p['identity']['pid'],0)) for p in self.protected_peers if p['identity']['pid'] not in moss and 'compute' in p['gpu_kinds'])
        pending=sum(max(0,j['request']['gpu_bytes']-actual[token]) for token,j in state['jobs'].items())
        projected=current+growth+pending+prospective;limit=self.sharing['external_compute_budget_bytes']
        return {'sharing_active':True,'external_compute_actual_bytes':current,'external_compute_projected_bytes':projected,
                'external_compute_limit_bytes':limit,'external_compute_budget_exceeded':current>limit or projected>limit}
    @contextlib.contextmanager
    def transaction(self):
        self.path.mkdir(parents=True,exist_ok=True)
        with (self.path/'registry.lock').open('a+') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise Uncertain('Registry transaction busy; no blocking cleanup or silent adoption')
            p=self.path/'registry.json'
            state=json.loads(p.read_text()) if p.exists() else {'schema':2,'gpu':self.gpu,'policy_sha256':self.policy_sha256,'jobs':{},'retired':[]}
            if state.get('schema')!=2 or state.get('gpu')!=self.gpu or state.get('policy_sha256')!=self.policy_sha256:raise Conflict('Registry identity/schema/reviewed policy differs')
            yield state
            atomic(p,state)
    def membership(self,job,obs):
        procs=obs['procs'];worker=job.get('worker')
        if not worker:return set()
        if matches(worker,procs):return descendants(worker['pid'],procs)
        # A dead leader does not authorize adoption of a reused PID/session.
        return {v['pid'] for v in job.get('known_members',[]) if matches(v,procs)}
    def reconcile(self,state,obs):
        if not obs.get('complete'):raise Uncertain('Incomplete resource/process telemetry')
        for token,job in list(state['jobs'].items()):
            if matches(job['owner'],obs['procs']):continue
            members=self.membership(job,obs)
            # Also retain a reservation for any surviving original session;
            # this blocks reaping, but does not authorize signalling it.
            w=job.get('worker')
            session=set() if not w else {p for p,v in obs['procs'].items() if v['sid']==w['pid'] and v['birth']>=w['birth']}
            if members or session:
                job['status']='orphaned';continue
            if any(p in obs['gpu_memory'] for p in members):raise Uncertain('Stale CUDA accounting')
            state['retired'].append({**job,'token':token,'retired_unix':time.time(),'reason':'owner and bound session gone'})
            del state['jobs'][token]
    def protected_processes(self,obs):
        """Process-only proof remains available when GPU telemetry is unavailable."""
        protected=set()
        for peer in self.protected_peers:
            saved=peer['identity'];pid=saved['pid'];current=obs['procs'].get(pid,{})
            if any(current.get(k)!=saved[k] for k in IDENTITY_KEYS):raise Conflict('Protected process identity changed')
            protected.add(pid)
        return protected
    def protected(self,obs):
        if not obs.get('complete'):raise Uncertain('Complete compute/graphics/process observation required')
        protected=self.protected_processes(obs);growth_gpu=growth_ram=0
        for peer in self.protected_peers:
            pid=peer['identity']['pid'];current=obs['procs'][pid]
            if set(obs.get('gpu_kinds',{}).get(pid,()))!=set(peer['gpu_kinds']):raise Conflict('Protected compute/graphics membership changed')
            used=obs['gpu_memory'].get(pid,0);rss=current['rss']
            growth_gpu+=max(0,peer['gpu_budget_bytes']-used)
            growth_ram+=max(0,peer['ram_budget_bytes']-rss,current.get('hwm',rss)-rss)
        return protected,growth_gpu,growth_ram
    def accounting(self,state,obs):
        protected,_,_=self.protected(obs)
        known=set(protected);actual={};ram={}
        for token,job in state['jobs'].items():
            members=self.membership(job,obs)
            if known&members:raise Conflict('Overlapping job ownership or protected process trees')
            known|=members
            actual[token]=sum(n for pid,n in obs['gpu_memory'].items() if pid in members)
            owners=descendants(job['owner']['pid'],obs['procs']) if matches(job['owner'],obs['procs']) else members
            ram[token]=sum(obs['procs'][pid]['rss'] for pid in owners if pid in obs['procs'])
        unknown=set(obs['gpu_memory'])-known
        if unknown:raise Conflict('Unregistered GPU allocation(s): '+str(sorted(unknown)))
        return actual,ram
    def reserve(self,role,owner,request,obs):
        if role not in self.roles:raise Conflict('Role is outside this approved deployment')
        if obs['gpu']!=self.gpu:raise Conflict('Wrong GPU')
        if not matches(owner,obs['procs']):raise Uncertain('Guardian identity not live')
        protected,_,_=self.protected(obs)
        if descendants(owner['pid'],obs['procs']) & protected:raise Conflict('Guardian tree overlaps protected peer; no adoption')
        if any(type(request.get(k)) is not int or request[k]<=0 for k in ('gpu_bytes','ram_bytes')):raise ValueError('Positive resource reservations required')
        stack=contextlib.ExitStack();handles=[]
        try:
            # Shared leases retain interoperability: legacy exclusive jobs
            # cannot overlap, and no existing exclusive owner is bypassed.
            for path in self.locks:
                fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
                f=stack.enter_context(os.fdopen(fd,'rb'));handles.append(f)
                opened,current=os.fstat(f.fileno()),path.lstat()
                if not stat.S_ISREG(current.st_mode) or (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino):raise Conflict('Physical GPU lock identity/type differs')
                try:fcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB)
                except BlockingIOError:raise Conflict('Existing exclusive guardian requires coordinated migration')
            for path,f in zip(self.locks,handles):
                opened,current=os.fstat(f.fileno()),path.lstat()
                if not stat.S_ISREG(current.st_mode) or (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino):raise Conflict('Physical GPU lock changed during acquisition')
            with self.transaction() as state:
                self.reconcile(state,obs)
                if any(j['role']==role for j in state['jobs'].values()):raise Conflict('Role already reserved; inspect before retry')
                if any(j.get('telemetry_unknown_since') is not None for j in state['jobs'].values()):raise Uncertain('Peer telemetry uncertain; no new admission')
                if any(j['status']=='orphaned' for j in state['jobs'].values()):raise Conflict('Live orphan reservation requires reconciliation')
                if any(owner['pid'] in self.membership(j,obs) for j in state['jobs'].values()):raise Conflict('A new guardian cannot nest inside another job worker')
                actual,ram=self.accounting(state,obs)
                shared=self.external_compute(state,obs,actual,request['gpu_bytes'])
                if shared['external_compute_budget_exceeded']:raise Capacity('Shared external COMPUTE budget exceeds margin below MOSS4GiB')
                # Current usage plus unconsumed peer budgets plus new budget;
                # driver/library allowance includes observed unattributed use.
                overhead=max(0,obs['used']-sum(obs['gpu_memory'].values()))
                observed_charge=max(obs['used'],sum(obs['gpu_memory'].values()))
                gpu_need=observed_charge+max(0,self.driver_headroom-overhead)+request['gpu_bytes']+sum(max(0,j['request']['gpu_bytes']-actual[t]) for t,j in state['jobs'].items())
                protected,protected_gpu_growth,protected_ram_growth=self.protected(obs)
                gpu_need+=protected_gpu_growth
                ram_need=request['ram_bytes']+self.host_free_floor+protected_ram_growth+sum(max(0,j['request']['ram_bytes']-ram[t]) for t,j in state['jobs'].items())
                # Protect observed previous peak growth of unrelated processes
                # (including GPU1 sampler/Nano) without a free-RAM threshold.
                own=set().union(*(descendants(j['owner']['pid'],obs['procs']) for j in state['jobs'].values())) if state['jobs'] else set()
                own|=descendants(owner['pid'],obs['procs'])
                protected_growth=sum(max(0,v['hwm']-v['rss']) for pid,v in obs['procs'].items() if pid not in own and pid in obs.get('protected_pids',[]) and pid not in protected)
                if self.host_peak_identities:protected_growth=self.host_peak_growth(obs,own)
                if gpu_need>obs['total']:raise Capacity('GPU resource reservations plus driver headroom exceed physical capacity')
                if ram_need+protected_growth>obs['available_ram']:raise Capacity('Declared remaining job growth plus observed protected growth exceeds available RAM')
                token=uuid.uuid4().hex
                state['jobs'][token]={'role':role,'owner':owner,'request':dict(request),'worker':None,'known_members':[],
                  'status':'reserved','created_unix':time.time(),'heartbeat_unix':time.time()}
            return Lease(self,token,owner,stack,handles)
        except BaseException:stack.close();raise

class Lease:
    def __init__(self,registry,token,owner,stack,handles):self.registry=registry;self.token=token;self.owner=owner;self.stack=stack;self.handles=handles
    def own_job(self,state):
        job=state['jobs'].get(self.token)
        if job is None or job['owner']!=self.owner:raise Conflict('Lease token/guardian ownership mismatch')
        return job
    def check_physical_locks(self):
        for path,f in zip(self.registry.locks,self.handles):
            opened,current=os.fstat(f.fileno()),path.lstat()
            if not stat.S_ISREG(current.st_mode) or (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino):raise Conflict('Physical GPU lock path replaced; retain uncertainty')
    def attach(self,worker,obs):
        self.check_physical_locks()
        self.registry.protected(obs)
        if not matches(worker,obs['procs']):raise Uncertain('Worker identity not observable')
        p=obs['procs'][worker['pid']]
        if p['ppid']!=self.owner['pid'] or p['pgid']!=worker['pid'] or p['sid']!=worker['pid']:raise Conflict('Worker must be a direct child in its own new session')
        with self.registry.transaction() as state:
            j=self.own_job(state)
            if j['worker'] is not None:raise Conflict('Worker already attached')
            j.update(worker=worker,status='running',known_members=[worker],heartbeat_unix=time.time())
    def health(self,obs):
        self.check_physical_locks()
        with self.registry.transaction() as state:
            self.registry.reconcile(state,obs);j=self.own_job(state)
            actual,ram=self.registry.accounting(state,obs)
            _,protected_gpu_growth,protected_ram_growth=self.registry.protected(obs)
            overhead=max(0,obs['used']-sum(obs['gpu_memory'].values()))
            observed_charge=max(obs['used'],sum(obs['gpu_memory'].values()))
            gpu_need=observed_charge+max(0,self.registry.driver_headroom-overhead)+protected_gpu_growth+sum(max(0,v['request']['gpu_bytes']-actual[t]) for t,v in state['jobs'].items())
            if gpu_need>obs['total']:raise Capacity('Protected/registered GPU growth no longer fits; stop owned worker only')
            ram_need=self.registry.host_free_floor+protected_ram_growth+sum(max(0,v['request']['ram_bytes']-ram[t]) for t,v in state['jobs'].items())
            own=set().union(*(descendants(v['owner']['pid'],obs['procs']) for v in state['jobs'].values())) if state['jobs'] else set()
            ram_need+=self.registry.host_peak_growth(obs,own)
            shared=self.registry.external_compute(state,obs,actual)
            # Legacy CPU fixtures preserve report-only RAM; explicit native
            # sharing policy enforces ownedRSS,8GiBfreefloor andgrowth below.
            ram_forecast_fits=ram_need<=obs['available_ram']
            members=self.registry.membership(j,obs)
            j['known_members']=[{k:obs['procs'][p][k] for k in ('pid','birth','boot')} for p in sorted(members) if p in obs['procs']]
            j['heartbeat_unix']=time.time();j.pop('telemetry_unknown_since',None)
            return {'own_gpu_bytes':actual[self.token],'own_ram_bytes':ram[self.token],
              'own_gpu_budget_exceeded':actual[self.token]>j['request']['gpu_bytes'],
              'own_ram_above_reservation':ram[self.token]>j['request']['ram_bytes'],
              'own_ram_budget_exceeded':self.registry.sharing is not None and ram[self.token]>j['request']['ram_bytes'],
              'shared_host_floor_breached':self.registry.sharing is not None and obs['available_ram']<self.registry.sharing['minimum_available_ram_bytes'],
              'shared_host_growth_budget_exceeded':self.registry.sharing is not None and not ram_forecast_fits,
              **shared,
              'host_growth_forecast_fits':ram_forecast_fits,'host_growth_required_bytes':ram_need,
              'registered_peers':[x['role'] for t,x in state['jobs'].items() if t!=self.token],
              'members':sorted(members),'free_gpu_bytes':obs['total']-obs['used']}
    def note_telemetry_unknown(self):
        with self.registry.transaction() as state:
            j=self.own_job(state);j.setdefault('telemetry_unknown_since',time.time())
    def cleanup_targets(self,obs):
        with self.registry.transaction() as state:
            self.registry.protected_processes(obs)
            j=self.own_job(state);members=self.registry.membership(j,obs)
            # Never signal a reused leader or an unverified foreign session.
            w=j.get('worker')
            if not w:return []
            protected={p['identity']['pid'] for p in self.registry.protected_peers}
            if protected & members:raise Conflict('Protected PID intersects owned cleanup tree; retain uncertain lease')
            if matches(w,obs['procs']):
                if any(p in protected and v.get('pgid')==w['pid'] for p,v in obs['procs'].items()):raise Conflict('Protected peer shares owned process group; never signal it')
                return [{'pgid':w['pid'],'leader':w}]
            return [{'pid':p,'identity':{k:obs['procs'][p][k] for k in ('pid','birth','boot')}} for p in members if p in obs['procs']]
    def release(self,obs):
        self.check_physical_locks()
        with self.registry.transaction() as state:
            j=self.own_job(state)
            w=j.get('worker')
            session=set() if w is None else {p for p,v in obs['procs'].items() if v['sid']==w['pid'] and v['birth']>=w['birth']}
            if self.registry.membership(j,obs) or session:raise Conflict('Owned worker/session still alive; do not release reservation')
            state['retired'].append({**j,'token':self.token,'retired_unix':time.time(),'reason':'owner verified cleanup'})
            del state['jobs'][self.token]
        self.stack.close()
    def fds(self):
        # Inherited shared file descriptions retain the device claim if the
        # guardian dies before its bound worker has actually exited.
        return [f.fileno() for f in self.handles]
