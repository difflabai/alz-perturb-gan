"""No CUDA initialization or real payloads: reference-transfer boundaries."""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cov_common as C
import resource_common as U
import covmom_gpu as ADAPTER
import test_covmom as SYNTHETIC


def test_CPU_adapter_retains_exact_frozen_reference_and_RNG():
    ctx=SYNTHETIC.sample_context()
    original=ADAPTER.M.CovarianceArm(None,'d.covmom');ADAPTER._cpu_setup(original,ctx)
    before=C.torch.get_rng_state().clone();numpy_before=copy.deepcopy(C.np.random.get_state())
    adapted=ADAPTER.M.CovarianceArm(None,'d.covmom');adapted.setup(ctx)
    assert C.T._same(original.state(),adapted.state()) and original.info==adapted.info
    assert C.torch.equal(before,C.torch.get_rng_state())
    assert all(C.np.array_equal(a,b) for a,b in zip(numpy_before,C.np.random.get_state()))


def test_GPU_adapter_reads_reference_from_CPU_without_using_GPU_statistics(monkeypatch):
    cpu=SYNTHETIC.sample_context();source=ADAPTER.M.CovarianceArm(None,'d.covmom')
    ADAPTER._cpu_setup(source,cpu);cpu.plugins=[source];seen=[]
    def reference(cache,device,flags):
        seen.append((cache,device,flags));return cpu
    monkeypatch.setattr(C.T,'Context',reference)
    native_to=C.torch.Tensor.to
    def simulated_transfer(self,*args,**kwargs):
        if args and isinstance(args[0],C.torch.device) and args[0].type=='cuda':return self.clone()
        return native_to(self,*args,**kwargs)
    monkeypatch.setattr(C.torch.Tensor,'to',simulated_transfer)
    ctx=SimpleNamespace(cache=Path('/synthetic'),device=C.torch.device('cuda'),flags=cpu.flags)
    adapted=ADAPTER.M.CovarianceArm(None,'d.covmom');adapted.setup(ctx)
    assert seen==[(ctx.cache,'cpu',ctx.flags)]
    assert C.T._same(adapted.state(),source.state()) and adapted.info==source.info
    adapted.basis[0,0]+=1
    assert not C.np.array_equal(adapted.basis,source.basis)
    assert not C.torch.cuda.is_initialized()


def test_adapter_refuses_unregistered_device():
    ctx=SimpleNamespace(device=C.torch.device('meta'))
    with pytest.raises(ValueError,match='declared'):ADAPTER.M.CovarianceArm(None,'d.covmom').setup(ctx)


def test_GPU_setup_rejects_unapproved_selector_before_CUDA(monkeypatch):
    seen=[];monkeypatch.setattr(C,'cpu',lambda:seen.append('cpu limits'))
    with pytest.raises(ValueError,match='GPU1'):U.setup_gpu('0')
    assert seen==['cpu limits'] and not C.torch.cuda.is_initialized()


def test_owner_sharing_override_retains_source_and_CPU_proof_checks_without_scheduler(tmp_path,monkeypatch):
    monkeypatch.setattr(C,'HERE',tmp_path)
    sources={f'synthetic{i}':'a'*64 for i in range(222)}
    clearance={'status':'cleared_before_bounded_GPU_TRAIN_verification',
        'source_manifest_sha256':'b'*64,'cpu_training_verification_sha256':'c'*64}
    monkeypatch.setattr(U,'sources',lambda:sources)
    monkeypatch.setattr(U,'cpu_proof',lambda:'c'*64)
    monkeypatch.setattr(C.B,'sha',lambda p:'b'*64)
    monkeypatch.setattr(C.B.P,'read',lambda p:sources if 'source_manifest' in p.name else clearance)
    assert U.verification_ready()==sources
    clearance['cpu_training_verification_sha256']='d'*64
    with pytest.raises(ValueError,match='preflight'):U.verification_ready()
    assert not C.torch.cuda.is_initialized()
