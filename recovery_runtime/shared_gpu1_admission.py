"""Observe and budget same-user foreign GPU occupants without owning them.

Only ALZ registry members are controllable. Foreign identities are observations,
refreshed on each complete sample, with explicit growth allowances. This is a
sampled coexistence forecast, not a reservation or allocator cap on other jobs.
"""
import os
import gpu1_admission as base

GPU0=base.GPU0
GiB=base.GiB
LOCKS=base.LOCKS
IDENTITY_KEYS=base.IDENTITY_KEYS
Conflict=base.Conflict
Capacity=base.Capacity
Uncertain=base.Uncertain

def identity(pid):return base.identity(pid)
def protected_identity(pid):return base.protected_identity(pid)
def atomic(path,value):return base.atomic(path,value)
def snapshot(gpu=GPU0,**kwargs):return base.snapshot(gpu,**kwargs)
def proc_table():return base.proc_table()
def matches(saved,procs):return base.matches(saved,procs)
def descendants(pid,procs):return base.descendants(pid,procs)

class Registry(base.Registry):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.observed_foreign=[]
        self._registered_members=set()

    def _owned(self,obs,state=None):
        # Current guard descendants include its spawn-barrier worker before attach.
        own=descendants(os.getpid(),obs['procs']) if os.getpid() in obs['procs'] else set()
        if state is not None:
            self._registered_members=set()
            for job in state['jobs'].values():
                members=self.membership(job,obs)
                self._registered_members.update((p,obs['procs'][p]['birth'],obs['procs'][p]['boot']) for p in members)
                own|=members
        else:
            own|={p for p,birth,boot in self._registered_members if obs['procs'].get(p,{}).get('birth')==birth and obs['procs'][p].get('boot')==boot}
        return own

    def refresh_foreign(self,obs,state=None):
        if not obs.get('complete'):raise Uncertain('Complete GPU/process observation required')
        if obs.get('gpu')!=self.gpu:raise Conflict('Wrong GPU observation')
        if set(obs['gpu_memory'])!=set(obs.get('gpu_kinds',{})):raise Uncertain('Incomplete GPU membership')
        own=self._owned(obs,state);peers=[]
        for pid in sorted(set(obs['gpu_memory'])-own):
            current=obs['procs'].get(pid)
            if current is None or any(k not in current for k in IDENTITY_KEYS):raise Uncertain('Foreign GPU identity not observable')
            if current['uid']!=os.getuid():raise Conflict('Foreign GPU UID outside owner-authorized account')
            if type(current['birth']) is not int or current['birth']<=0 or not current['boot'] or not current['exe'] or len(current['argv_sha256'])!=64:raise Uncertain('Malformed foreign GPU identity')
            kinds=obs['gpu_kinds'][pid];used=obs['gpu_memory'][pid]
            if not kinds or len(kinds)!=len(set(kinds)) or set(kinds)-{'compute','graphics'}:raise Uncertain('Invalid foreign GPU process kinds')
            if type(used) is not int or used<0 or any(type(current.get(k)) is not int or current[k]<0 for k in ('rss','hwm')):raise Uncertain('Invalid foreign memory observation')
            allowance=GiB//2 if 'compute' in kinds else GiB//4
            peers.append({'identity':{k:current[k] for k in IDENTITY_KEYS},'gpu_kinds':list(kinds),
                          'gpu_budget_bytes':used+allowance,'ram_budget_bytes':max(current['rss'],current['hwm'])+2*GiB,
                          'observed_gpu_bytes':used,'observed_rss_bytes':current['rss'],
                          'GPU_growth_allowance_bytes':allowance,'foreign_control':'NONE'})
        self.protected_peers=peers
        self.observed_foreign=peers
        return {p['identity']['pid'] for p in peers}

    def accounting(self,state,obs):
        self.refresh_foreign(obs,state)
        return super().accounting(state,obs)

    def protected(self,obs):
        self.refresh_foreign(obs)
        return super().protected(obs)

    def protected_processes(self,obs):
        if obs.get('complete'):self.refresh_foreign(obs)
        # Cleanup can operate on a process-only view after telemetry failure.
        # No foreign survivor is adopted; runner independently proves own targets.
        return {p['identity']['pid'] for p in self.protected_peers if p['identity']['pid'] in obs['procs']}
