"""Inspect immutable native096 parent on CPU; no CUDA metadata rewriting."""
import datetime,json
from pathlib import Path
import torch
import released_adapter as model,released_update as update
from state_fingerprint import fingerprint
import preset_scope as scope

def main():
    p=scope.read(scope.PROTOCOL);e=p['extension'];path=Path(e['checkpoint_path'])
    assert scope.sha(path)==e['checkpoint_SHA256']
    a=scope.read(p['parent_audit_path']);assert a['own_processes_absent'] and a['owned_registry_lease_absent']
    q=torch.load(path,map_location='cpu',weights_only=False)
    update.configure(scope.CFG);update.configure_hyperparameters(scope.HYPER)
    from extension import validate_parent
    validate_parent(q,dict(extension=e,cell=scope.cell(0,'pg',0),cache_dir=str(scope.read(Path('/ml2/alz/artifacts/allocations/CAL_GPU0_V1_external_b1_f0_pg_s0/authorization.json'))['cache_dir']),released_hyperparameters=scope.HYPER,variant_config=scope.CFG))
    b,*_=model.build(torch.ones(2000),torch.ones(2000),'cpu',0,0)
    for name,module in zip(('generator','critic','prior'),b[1:4]):
        module.load_state_dict(q['policy']['models'][name],strict=True)
        assert all(torch.isfinite(t).all() for t in module.state_dict().values())
    from particlegan.policy import _validate_optimizer_state
    for optimizer,saved in zip(b[4:6],q['policy']['optimizers']):
        _validate_optimizer_state(optimizer,saved);optimizer.load_state_dict(saved)
        assert fingerprint(optimizer.state_dict())==fingerprint(saved)
    assert q['recipe']['ema_decay']==0 and q['usage']['steps']==20000
    assert q['particlegan_version']=='0.9.6' and not torch.cuda.is_initialized()
    proof=dict(status='ACTUAL_NATIVE096_COMPLETE20K_PARENT_CPU_COMPONENT_AUDIT_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint_SHA256=scope.sha(path),whole_parent_scientific_fingerprint=fingerprint(q),native_device_metadata_relabelled=False,CUDA_initialized=False,source_SHA256={str(x):scope.sha(x) for x in (Path(__file__),Path(model.__file__),Path(update.__file__))})
    assert not scope.PARENT_PROOF.exists();scope.PARENT_PROOF.write_text(json.dumps(proof,indent=2)+'\n')
    p['extension']['whole_parent_scientific_fingerprint']=fingerprint(q)
    scope.PROTOCOL.write_text(json.dumps(p,indent=2)+'\n')
    print(proof['status'])
if __name__=='__main__':main()
