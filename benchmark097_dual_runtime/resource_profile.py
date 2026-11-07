"""Bounded shared profiles for two fixed-device ALZ benchmark lanes."""
import copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
from dual_coordination import GPUS,RESOURCE_PROTOCOL,assigned_gpu,validate_resource_protocol
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
BASE=ROOT/'benchmark097_runtime/resource_profile.py'
PREPARED=HERE/'policy.prepared.json'
REVIEW=ROOT/'analysis/benchmark097_dual_owner_execution_review.json'
SCOPE='NATIVE097_FULL_MATRIX_ALL_FRESH_THREE_METHODS_DUAL_SHARED_GPUS'
OPS=('external097_benchmark_cell',)
_BASE=None
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def base():
 global _BASE
 assert sha(BASE)==read(RESOURCE_PROTOCOL)['reference_profile_SHA256'],'Original shared resource source changed'
 if _BASE is None:
  spec=importlib.util.spec_from_file_location('unchanged_serial_resource_reference',BASE);_BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(_BASE)
 return _BASE

def prepared_path(gpu):return HERE/f'policy.gpu{GPUS.index(gpu)}.prepared.json'
def deployed_path(gpu):return ROOT/f'artifacts/gpu{GPUS.index(gpu)}-benchmark097-dual-20261107/policy.deployed.json'
def registry_path(gpu):return deployed_path(gpu).parent/'live_registry'
def deployed_bytes(gpu):
 p=read(prepared_path(gpu));assert p['status']=='PREPARED_EXACT_GPU0_PARALLEL_PROFILE';p['status']='REVIEWED_GPU0_DEPLOYED'
 return (json.dumps(p,indent=2)+'\n').encode()
def policy_sha(gpu=None):
 if gpu is not None:return hashlib.sha256(deployed_bytes(gpu)).hexdigest()
 return hashlib.sha256(json.dumps({g:policy_sha(g) for g in GPUS},sort_keys=True).encode()).hexdigest()
def validate_proposal(gpu=None):
 validate_resource_protocol();old=base().validate_proposal();bundle=read(PREPARED)
 assert bundle==dict(status='PREPARED_TWO_DEVICE_BUNDLE_NOT_ACTIVE',policies=[str(prepared_path(g)) for g in GPUS])
 policies={}
 for g in GPUS:
  p=read(prepared_path(g));expected=copy.deepcopy(old);expected['gpu']=g;expected['registry_path']=str(registry_path(g));expected['locks']=[s.replace('gpu-1.lock',f'gpu-{GPUS.index(g)}.lock') for s in old['locks']]
  expected['resource_basis']['foreign_control']='No foreign control. One bounded ALZ cell per assigned card, joint peer-ALZ host-growth reserve, unchanged physical locks, native methods, endpoints and scoring.'
  assert p==expected,'Only the physical lane and new scoped registry may differ from original shared policy'
  policies[g]=p
 return policies if gpu is None else policies[gpu]
def previous_trial_released():
 base().previous_trial_released()
 old=base().REGISTRY/'registry.json'
 assert not old.exists() or not read(old)['jobs'],'Serial benchmark namespace active'
def review_sources():
 import preset_scope as s
 paths=[*HERE.glob('*.py'),PREPARED,*[prepared_path(g) for g in GPUS],RESOURCE_PROTOCOL,BASE,*s.MODEL_DIR.glob('*.py'),s.PROTOCOL,s.MODEL_PROOF,s.NATIVE_PROOF,s.ENGINE_PROOF,s.WINDOW,*s.CFGEN_DIR.glob('*.py'),*s.STUDY.glob('*.py'),ROOT/'analysis/run_benchmark097_dual.py',s.RUNTIME_PROOF]
 return {str(p):sha(p) for p in paths if p!=s.PLAN}
def reviewed():
 validate_proposal();previous_trial_released();q=read(REVIEW)
 assert q['status']=='OWNER_AUTHORIZED_SOURCE_AND_RESOURCE_REVIEW_PASS' and q['reviewer_session']=='01a1a22b-f310-7190-a7c6-8697ecabae6c' and q['independent_review_claimed'] is False
 assert q['owner_authorization']==read(ROOT/'analysis/faithful097_owner_execution_review.json')['owner_authorization']
 assert q['scope']==SCOPE and q['policy_SHA256']==policy_sha() and q['source_SHA256']==review_sources()
 return q

def configure(auth,*,check_only=False):
 if not check_only:previous_trial_released()
 g=assigned_gpu(auth['cell']);p=validate_proposal(g);path=prepared_path(g) if check_only else deployed_path(g)
 pin=sha(prepared_path(g)) if check_only else policy_sha(g)
 auth.update(gpu_uuid=g,policy_path=str(path),policy_SHA256=pin,protected_peers=p['protected_peers'],registry_path=str(registry_path(g)),parallel_profile=True,dynamic_shared_foreign=True,dynamic_shared_gpu0=g==GPUS[0])
 auth['files_SHA256'].update({str(PREPARED):sha(PREPARED),str(prepared_path(g)):sha(prepared_path(g)),str(path):pin,str(BASE):sha(BASE),str(RESOURCE_PROTOCOL):sha(RESOURCE_PROTOCOL)})
 if REVIEW.exists():auth['files_SHA256'][str(REVIEW)]=sha(REVIEW)
 return p

def validate(auth):
 import preset_scope as s
 assert auth['operation'] in OPS and auth.get('parallel_profile') is True
 g=assigned_gpu(auth['cell']);p=validate_proposal(g);reviewed()
 assert auth['gpu_uuid']==g and auth.get('dynamic_shared_foreign') is True and auth['dynamic_shared_gpu0']==(g==GPUS[0])
 assert read(s.PLAN)==s.ENABLED and auth['files_SHA256'].get(str(s.PLAN))==sha(s.PLAN)
 assert auth['policy_path']==str(deployed_path(g)) and auth['registry_path']==str(registry_path(g))
 assert sha(deployed_path(g))==auth['policy_SHA256']==policy_sha(g) and auth['protected_peers']==p['protected_peers']
 assert auth['files_SHA256'].get(str(REVIEW))==sha(REVIEW)
 return p

def free_floor(gpu):
 raw=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=2).strip().split(',')
 assert len(raw)==2 and raw[0].strip()==gpu
 free=int(raw[1].strip())*2**20;available=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))*1024
 assert free>=3.5*2**30,'ALZ early stop: actual GPU free below3.5GiB'
 assert available>=12*2**30,'ALZ early stop: host available below12GiB'
 return free,available

def capacity(obs,admission,gpu):
 p=validate_proposal(gpu)
 reg=admission.Registry(registry_path(gpu),gpu=gpu,locks=p['locks'],roles=p['roles'],driver_headroom=p['driver_headroom_bytes'],host_free_floor=p['host_free_floor_bytes'],protected_peers=p['protected_peers'],policy_sha256=policy_sha(gpu))
 reg.accounting({'jobs':{}},obs);_,gg,gr=reg.protected(obs)
 req=p['roles']['alz_core_pg'];overhead=max(0,obs['used']-sum(obs['gpu_memory'].values()))
 required=max(obs['used'],sum(obs['gpu_memory'].values()))+max(0,p['driver_headroom_bytes']-overhead)+gg+req['gpu_bytes']
 peer=read(RESOURCE_PROTOCOL)['peer_ALZ_host_reserve_bytes']
 host=p['host_free_floor_bytes']+gr+req['ram_bytes']+peer
 assert required<=obs['total'],'Exact parallel GPU growth forecasts do not fit'
 assert host<=obs['available_ram'],'Exact parallel host growth forecasts do not fit'
 return dict(GPU_required_bytes=required,GPU_total_bytes=obs['total'],host_remaining_growth_required_bytes=host,host_available_bytes=obs['available_ram'],peer_ALZ_host_reserve_bytes=peer,GPU_fits=True,host_fits=True,allocation_or_lease_created=False)
