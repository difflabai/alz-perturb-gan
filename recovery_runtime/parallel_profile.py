"""Exact ALZ-owned parallel proposal; independent review gates native deployment.

The scheduler's old deployed policy/registry and guard are never modified.
Forecasts reserve foreign growth; they do not constrain or signal foreign work.
"""
import hashlib,json,sys,subprocess
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
PREPARED=HERE/'parallel_policy.prepared.json'
DEPLOYED=ROOT/'artifacts/gpu0-parallel-20261104/policy.deployed.json'
REGISTRY=DEPLOYED.parent/'live_registry'
REVIEW=ROOT/'analysis/gpu0_parallel_independent_review.json'
OLD=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/integration/policy.GPU0.CORE.PREPARED.json')
OLD_SHA='5a4dfd1d5288bde24964ee680c003dd75f13fefe1a83ac55b27b49e6a05d125e'
MOSS=Path('/ml2/particlesliders-working/moss_training/train_concept.py')
MOSS_SHA='ff27a2c749be2221181b25158ca59bf26e4c8e0520e44a80899629031a9babcb'
OPS=('qualify_interrupted_pg8','resume_pure_ensemble','confirm_frozen_ensemble_seed2')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def deployed_bytes():
 p=read(PREPARED);assert p['status']=='PREPARED_EXACT_GPU0_PARALLEL_PROFILE'
 p['status']='REVIEWED_GPU0_DEPLOYED'
 return (json.dumps(p,indent=2)+'\n').encode()
def policy_sha():return hashlib.sha256(deployed_bytes()).hexdigest()
def review_sources():
 paths=[*HERE.glob('*.py'),PREPARED,* (ROOT/'runtime').glob('*.py'),ROOT/'analysis/exact_seed1_recovery_protocol.json']
 return {str(p):sha(p) for p in paths}
def validate_proposal():
 assert sha(OLD)==OLD_SHA and sha(MOSS)==MOSS_SHA,'Original policy or exact MOSS source changed'
 old=read(OLD);p=read(PREPARED)
 assert p['gpu']==old['gpu'] and p['locks']==old['locks'] and p['registry_path']==str(REGISTRY)
 assert p['roles']=={'alz_core_pg':old['roles']['alz_core_pg']}
 assert p['driver_headroom_bytes']==3*2**30 and p['host_free_floor_bytes']==12*2**30
 expected={str(OLD.parent.parent/'v2'/name):sha(OLD.parent.parent/'v2'/name) for name in ['gpu0_admission.py','gpu0_runner.py']}
 expected[str(HERE/'parallel_runner.py')]=sha(HERE/'parallel_runner.py')
 assert p['source_sha256']==expected,'Complete exact active guard and original reference sources required'
 assert len(p['protected_peers'])==9
 assert [v['identity'] for v in p['protected_peers'][:8]]==[v['identity'] for v in old['protected_peers']]
 assert [v['gpu_kinds'] for v in p['protected_peers'][:8]]==[v['gpu_kinds'] for v in old['protected_peers']]
 assert p['protected_peers'][-1]['identity']['pid']==2156154
 assert not read(Path(old['registry_path'])/'registry.json')['jobs'],'Old resource registry has active work; no parallel namespace bypass'
 return p
def free_floor():
 """Read-only earlier stop for owned ALZ, respecting MOSS's own 2/8GiB stops.

 Sampling cannot guarantee against abrupt foreign growth between observations.
 No foreign signal, allocator setting or source mutation occurs here.
 """
 raw=subprocess.check_output(['nvidia-smi','-i',read(PREPARED)['gpu'],'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=2).strip().split(',')
 assert len(raw)==2 and raw[0].strip()==read(PREPARED)['gpu']
 free=int(raw[1].strip())*2**20
 available=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))*1024
 assert free>=3*2**30,'ALZ early stop: actual GPU free below3GiB, preserve MOSS2GiB reserve'
 assert available>=12*2**30,'ALZ early stop: host available below12GiB, preserve MOSS8GiB reserve'
 return free,available
def reviewed():
 validate_proposal();q=read(REVIEW)
 assert q['status']=='INDEPENDENT_SOURCE_REVIEW_PASS_FOR_EXACT_ALZ_PARALLEL_PROFILE'
 assert q['authenticated_sender'] in ('bits255.dots.ada_20261002','bits255.grokbot.grok_20261002') and q['message_id']
 assert q['policy_SHA256']==policy_sha() and q['source_SHA256']==review_sources(),'Review does not cover exact current source/profile'
 return q
def configure(auth,*,check_only=False):
 p=validate_proposal();path=PREPARED if check_only else DEPLOYED;pin=sha(PREPARED) if check_only else policy_sha()
 auth.update(policy_path=str(path),policy_SHA256=pin,protected_peers=p['protected_peers'],registry_path=str(REGISTRY),parallel_profile=True)
 auth['files_SHA256'].update({str(PREPARED):sha(PREPARED),str(path):pin,str(MOSS):MOSS_SHA})
 if REVIEW.exists():auth['files_SHA256'][str(REVIEW)]=sha(REVIEW)
 return p
def validate(auth):
 assert auth['operation'] in OPS and auth.get('parallel_profile') is True,'Parallel profile is exact recovery/seed2 only'
 p=validate_proposal();reviewed()
 assert auth['policy_path']==str(DEPLOYED) and auth['registry_path']==str(REGISTRY)
 assert sha(DEPLOYED)==auth['policy_SHA256']==policy_sha()
 assert auth['protected_peers']==p['protected_peers']
 assert auth['files_SHA256'].get(str(REVIEW))==sha(REVIEW),'Independent review evidence changed'
 for path,pin in {str(PREPARED):sha(PREPARED),str(DEPLOYED):policy_sha(),str(MOSS):MOSS_SHA}.items():
  assert auth['files_SHA256'].get(path)==pin
 return p
def capacity(obs,admission):
 """Same unchanged guard forecast arithmetic, no lock/registry/allocation write."""
 p=validate_proposal();reg=admission.Registry(REGISTRY,locks=p['locks'],roles=p['roles'],driver_headroom=p['driver_headroom_bytes'],host_free_floor=p['host_free_floor_bytes'],protected_peers=p['protected_peers'],policy_sha256=policy_sha())
 reg.accounting({'jobs':{}},obs);_,gg,gr=reg.protected(obs)
 req=p['roles']['alz_core_pg'];overhead=max(0,obs['used']-sum(obs['gpu_memory'].values()))
 gpu=max(obs['used'],sum(obs['gpu_memory'].values()))+max(0,p['driver_headroom_bytes']-overhead)+gg+req['gpu_bytes']
 ram=p['host_free_floor_bytes']+gr+req['ram_bytes']
 assert gpu<=obs['total'],'Exact parallel GPU growth forecasts do not fit'
 assert ram<=obs['available_ram'],'Exact parallel host growth forecasts do not fit'
 return {'GPU_required_bytes':gpu,'GPU_total_bytes':obs['total'],'host_remaining_growth_required_bytes':ram,'host_available_bytes':obs['available_ram'],'GPU_fits':True,'host_fits':True,'allocation_or_lease_created':False}
