"""Exact ALZ-owned parallel proposal; owner authorization and verified resource/source checks gate deployment.

The scheduler's old deployed policy/registry and guard are never modified.
Forecasts reserve foreign growth; they do not constrain or signal foreign work.
"""
import hashlib,json,sys,subprocess
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
ENGINE=ROOT/'recovery_runtime'
PREPARED=HERE/'policy.prepared.json'
DEPLOYED=ROOT/'artifacts/gpu1-released-best-extend40k-20261105/policy.deployed.json'
REGISTRY=DEPLOYED.parent/'live_registry'
REVIEW=ROOT/'analysis/released_best_extend40k_owner_execution_review.json'
OLD=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/integration/policy.GPU0.CORE.PREPARED.json')
OLD_SHA='5a4dfd1d5288bde24964ee680c003dd75f13fefe1a83ac55b27b49e6a05d125e'
MOSS=Path('/ml2/particlesliders-working/moss_training/train_concept.py')
MOSS_SHA='ff27a2c749be2221181b25158ca59bf26e4c8e0520e44a80899629031a9babcb'
PREVIOUS_REGISTRY=ROOT/'artifacts/gpu1-released-g050d050p100-20261104/live_registry/registry.json'
OPS=('extend_released_hyperparameters',)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def deployed_bytes():
 p=read(PREPARED);assert p['status']=='PREPARED_EXACT_GPU0_PARALLEL_PROFILE'
 p['status']='REVIEWED_GPU0_DEPLOYED'
 return (json.dumps(p,indent=2)+'\n').encode()
def policy_sha():return hashlib.sha256(deployed_bytes()).hexdigest()
def review_sources():
 import preset_scope as scope
 paths=[*HERE.glob('*.py'),PREPARED,*scope.MODEL_DIR.glob('*.py'),*scope.protocol_paths(scope.read(scope.PROTOCOL)),scope.RUNTIME_PROOF]
 return {str(p):sha(p) for p in paths if p!=scope.PLAN}
def validate_proposal():
 assert sha(OLD)==OLD_SHA,'Original policy changed'
 old=read(OLD);p=read(PREPARED)
 assert p['gpu']=='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69' and p['locks']==[s.replace('gpu-0.lock','gpu-1.lock') for s in old['locks']] and p['registry_path']==str(REGISTRY)
 assert p['dynamic_shared_foreign'] is True and p['foreign_growth_allowances']=={'compute_GPU_bytes':2**29,'graphics_GPU_bytes':2**28,'host_above_peak_bytes':2*2**30,'uid':__import__('os').getuid()}
 assert p['roles']=={'alz_core_pg':{'gpu_bytes':3*2**30,'ram_bytes':6*2**30}}
 assert p['driver_headroom_bytes']==3*2**30 and p['host_free_floor_bytes']==12*2**30
 expected={str(OLD.parent.parent/'v2'/name):sha(OLD.parent.parent/'v2'/name) for name in ['gpu0_admission.py','gpu0_runner.py']}
 expected[str(ENGINE/'parallel_runner.py')]=sha(ENGINE/'parallel_runner.py')
 expected.update({str(ENGINE/n):sha(ENGINE/n) for n in ['gpu1_admission.py','gpu1_runner.py','shared_gpu1_admission.py','shared_gpu1_runner.py']})
 expected.update({str(ROOT/'analysis/shared_gpu0_candidate'/n):sha(ROOT/'analysis/shared_gpu0_candidate'/n) for n in ('shared_gpu0_admission.py','shared_gpu0_runner.py')})
 for path,pin in p['resource_basis']['foreign_source_SHA256'].items():assert sha(path)==pin,'Foreign executable resource source changed'
 assert p['source_sha256']==expected,'Complete exact active guard and original reference sources required'
 assert not read(ROOT/'artifacts/gpu0-halloween-native-v1-20261104/live_registry/registry.json')['jobs'],'Previous Halloween namespace must be empty'
 assert not read(ROOT/'artifacts/gpu0-stage-prior-v1-20261104/live_registry/registry.json')['jobs'],'Previous stage-prior namespace must be empty'
 assert not read(ROOT/'artifacts/gpu0-frozen-prior-v1-20261104/live_registry/registry.json')['jobs'],'Previous fixed control must be empty'
 assert not read(ROOT/'artifacts/gpu0-ka2-ema-readout-v1-20261104/live_registry/registry.json')['jobs'],'Previous EMA evaluation must be empty'
 assert not read(ROOT/'artifacts/gpu1-mog-prior-v1-20261104/live_registry/registry.json')['jobs'],'Previous MoG namespace must be empty'
 assert not read(ROOT/'artifacts/gpu1-cell-energy-aux-v1-20261104/live_registry/registry.json')['jobs'],'Previous cellenergy namespace must be empty'
 assert not read(ROOT/'artifacts/gpu1-k3p-native-v1-20261104/live_registry/registry.json')['jobs'],'Previous K3P namespace must be empty'
 assert not read(ROOT/'artifacts/gpu1-ka2-native-v1-20261104/live_registry/registry.json')['jobs'],'Previous KA2 namespace must be empty'
 assert not read(ROOT/'artifacts/gpu1-bcap-adam-v1-20261104/live_registry/registry.json')['jobs'],'Previous BCAPAdam namespace must be empty'
 assert not read(ROOT/'artifacts/gpu1-fourier128-v1-20261104/live_registry/registry.json')['jobs'],'Previous Fourier128 namespace must be empty'
 assert not read(ROOT/'artifacts/gpu1-shared-v3-20261104/live_registry/registry.json')['jobs'],'Previous final confirmation registry must be empty'
 assert not read(ROOT/'artifacts/gpu1-resource-v2-20261104/live_registry/registry.json')['jobs'],'Prior resource namespace must be empty'
 assert not read(ROOT/'artifacts/gpu1-fiveday-v1-20261104/live_registry/registry.json')['jobs'],'Prior five-day GPU1 namespace must be empty'
 assert not read(ROOT/'artifacts/gpu1-learned-v1-20261104/live_registry/registry.json')['jobs'],'Prior shared GPU1 namespace must be empty'
 assert not read(ROOT/'artifacts/gpu0-turnover-20261104/live_registry/registry.json')['jobs'],'Latest own namespace must be empty'
 assert not read(ROOT/'artifacts/gpu0-parallel-20261104/live_registry/registry.json')['jobs'],'Prior own namespace must be empty after clean turnover'
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
 assert free>=3.5*2**30,'ALZ early stop: actual GPU free below3.5GiB, yield before foreign46000MiB whole-device stop'
 assert available>=12*2**30,'ALZ early stop: host available below12GiB, preserve MOSS8GiB reserve'
 return free,available
def previous_trial_released():
 assert not read(PREVIOUS_REGISTRY)['jobs'],'Owned predecessor must release before next registered rate trial'
def reviewed():
 validate_proposal();previous_trial_released();q=read(REVIEW)
 assert q['status']=='OWNER_AUTHORIZED_SOURCE_AND_RESOURCE_REVIEW_PASS'
 assert q['reviewer_session']=='01a1a22b-f310-7190-a7c6-8697ecabae6c' and q['independent_review_claimed'] is False
 assert q['owner_authorization']=={'available_capacity':'I authorize using the available GPU capacity.','both_cards':'You can use both gpus without permission.','sharing':'you dont need solo access','continuation':'contineue','deadline_extension':'extend it to 5 days'}
 assert q['scope']=='ONE_RELEASED_HALF_RATE_20000_TO40000_EXTENSION_ONLY'
 assert q['policy_SHA256']==policy_sha() and q['source_SHA256']==review_sources(),'Review does not cover exact current source/profile'
 return q
def configure(auth,*,check_only=False):
 if not check_only:previous_trial_released()
 p=validate_proposal();path=PREPARED if check_only else DEPLOYED;pin=sha(PREPARED) if check_only else policy_sha()
 auth.update(gpu_uuid=p['gpu'],policy_path=str(path),policy_SHA256=pin,protected_peers=p['protected_peers'],registry_path=str(REGISTRY),parallel_profile=True,dynamic_shared_foreign=True,dynamic_shared_gpu0=False)
 auth['files_SHA256'].update({str(PREPARED):sha(PREPARED),str(path):pin})
 if REVIEW.exists():auth['files_SHA256'][str(REVIEW)]=sha(REVIEW)
 return p
def validate(auth):
 assert auth['operation'] in OPS and auth.get('parallel_profile') is True,'Shared profile is one released ParticleGAN scalar configuration only'
 p=validate_proposal();reviewed()
 assert auth.get('dynamic_shared_foreign') is True and auth.get('dynamic_shared_gpu0') is False
 assert auth['files_SHA256'].get(str(HERE/'plan.json'))==sha(HERE/'plan.json')
 import preset_scope;assert read(HERE/'plan.json')==preset_scope.ENABLED
 assert auth['gpu_uuid']==p['gpu']
 assert auth['policy_path']==str(DEPLOYED) and auth['registry_path']==str(REGISTRY)
 assert sha(DEPLOYED)==auth['policy_SHA256']==policy_sha()
 assert auth['protected_peers']==p['protected_peers']
 assert auth['files_SHA256'].get(str(REVIEW))==sha(REVIEW),'Execution review evidence changed'
 for path,pin in {str(PREPARED):sha(PREPARED),str(DEPLOYED):policy_sha()}.items():
  assert auth['files_SHA256'].get(path)==pin
 return p
def capacity(obs,admission):
 """Same unchanged guard forecast arithmetic, no lock/registry/allocation write."""
 p=validate_proposal();reg=admission.Registry(REGISTRY,gpu=p['gpu'],locks=p['locks'],roles=p['roles'],driver_headroom=p['driver_headroom_bytes'],host_free_floor=p['host_free_floor_bytes'],protected_peers=p['protected_peers'],policy_sha256=policy_sha())
 reg.accounting({'jobs':{}},obs);_,gg,gr=reg.protected(obs)
 req=p['roles']['alz_core_pg'];overhead=max(0,obs['used']-sum(obs['gpu_memory'].values()))
 gpu=max(obs['used'],sum(obs['gpu_memory'].values()))+max(0,p['driver_headroom_bytes']-overhead)+gg+req['gpu_bytes']
 ram=p['host_free_floor_bytes']+gr+req['ram_bytes']
 assert gpu<=obs['total'],'Exact parallel GPU growth forecasts do not fit'
 assert ram<=obs['available_ram'],'Exact parallel host growth forecasts do not fit'
 return {'GPU_required_bytes':gpu,'GPU_total_bytes':obs['total'],'host_remaining_growth_required_bytes':ram,'host_available_bytes':obs['available_ram'],'GPU_fits':True,'host_fits':True,'allocation_or_lease_created':False,'observed_foreign':getattr(reg,'observed_foreign',[])}
