"""Inactive GPU0 sharing policy bytes and read-only forecasts; no deployment API."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PREPARED=HERE/'policy.prepared.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def deployed_bytes():
    value=read(PREPARED);assert value['status']=='PREPARED_EXACT_GPU0_PARALLEL_PROFILE'
    value['status']='REVIEWED_GPU0_DEPLOYED'
    return (json.dumps(value,indent=2)+'\n').encode()
def capacity(obs,admission):
    p=read(PREPARED)
    assert p['gpu']==obs['gpu']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
    for path,pin in p['source_sha256'].items():assert sha(path)==pin
    reg=admission.Registry(p['registry_path'],gpu=p['gpu'],locks=p['locks'],roles=p['roles'],driver_headroom=p['driver_headroom_bytes'],host_free_floor=p['host_free_floor_bytes'],protected_peers=p['protected_peers'],policy_sha256=hashlib.sha256(deployed_bytes()).hexdigest())
    reg.accounting({'jobs':{}},obs);_,gg,gr=reg.protected(obs)
    req=p['roles']['alz_core_pg'];overhead=max(0,obs['used']-sum(obs['gpu_memory'].values()))
    gpu=max(obs['used'],sum(obs['gpu_memory'].values()))+max(0,p['driver_headroom_bytes']-overhead)+gg+req['gpu_bytes']
    ram=p['host_free_floor_bytes']+gr+req['ram_bytes']
    assert gpu<=obs['total'],'GPU0 growth forecast does not fit'
    assert ram<=obs['available_ram'],'GPU0 host growth forecast does not fit'
    assert not Path(p['registry_path']).exists(),'Prototype unexpectedly created a native resource namespace'
    return {'GPU_required_bytes':gpu,'GPU_total_bytes':obs['total'],'host_remaining_growth_required_bytes':ram,'host_available_bytes':obs['available_ram'],'GPU_fits':True,'host_fits':True,'observed_foreign':reg.observed_foreign,'allocation_or_lease_created':False,'native_launcher_integrated':False}
