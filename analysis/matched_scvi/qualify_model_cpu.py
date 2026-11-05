"""New full-width sampler qualification plus the retained native scVI tests."""
import hashlib,json,os,runpy,socket,subprocess,sys,time,datetime
from pathlib import Path
ROOT=Path('/ml2/alz');SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=[str(SCIENCE),str(ROOT),str(ROOT/'scvi_calibration_runtime')]
def deny(self,address):raise RuntimeError('CPU qualification network disabled')
socket.socket.connect=deny;socket.socket.connect_ex=deny
import numpy as np,scipy.sparse as sp,torch,scvi
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from scvi_adapter import make_adata,make_model
from generation import iter_scvi
from stage_law import Schedule
from types import SimpleNamespace
assert scvi.__version__=='1.3.3'
# Full output width is required independently of the small training fixture.
rng=np.random.default_rng(731);stage=np.repeat(np.arange(4),32);donor=np.array([f'SYNTH_DONOR{s}' for s in stage])
a=make_adata(sp.csr_matrix(rng.poisson(3,(128,2000)).astype(np.float32)),[f'SYNTH_CELL_{i}' for i in range(128)],donor,stage,[f'SYNTH_GENE_{i}' for i in range(2000)],expected_donors=sorted(set(donor)))
model=make_model(a,0);model.module.eval();schedule=Schedule(np.zeros(200,dtype=int),np.zeros((200,400),dtype=int),np.full((200,400),6000,dtype=np.float32))
ctx=SimpleNamespace(project=lambda counts:counts[:,:50].float())
c=np.zeros(7,dtype=np.float32);c[3]=2/3
first=next(iter_scvi(ctx,model,schedule,c,715));second=next(iter_scvi(ctx,model,schedule,c,715))
assert all(np.array_equal(x,y) if isinstance(x,np.ndarray) else x==y for x,y in zip(first,second))
assert first[0]==0 and first[1].shape==(5,400,50) and first[2].shape==(1,400,2000) and first[3].shape==(2000,)
assert np.isfinite(first[2]).all() and (first[2]>=0).all() and np.equal(first[2],np.floor(first[2])).all()
assert not torch.cuda.is_initialized()
# Native actual two-epoch fit/load tests execute in a separate process to keep
# thread configuration and optional library import state isolated.
cmd=[sys.executable,'-I','-S','-B','-c',"import sys,runpy;sys.path.extend(sys.argv[1:3]);sys.path.insert(0,sys.argv[3]);sys.argv=[sys.argv[4]];runpy.run_path(sys.argv[0],run_name='__main__')",str(Path(scvi.__file__).resolve().parents[1]),str(ROOT),str(SCIENCE),str(SCIENCE/'test_scvi.py')]
result=subprocess.run(cmd,capture_output=True,text=True,env=os.environ,timeout=90)
assert result.returncode==0,result.stderr
assert 'Ran 2 tests' in result.stderr and result.stderr.rstrip().endswith('OK')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();protocol=json.loads((ROOT/'analysis/matched_scvi/protocol.json').read_bytes())
paths=[Path(__file__),ROOT/'analysis/matched_scvi/protocol.json',*[Path(p) for p in protocol['source_SHA256']]]
q={'status':'MATCHED_SCVI_NATIVE_CPU_MODEL_AND_FULLWIDTH_SAMPLER_QUALIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scvi_version':scvi.__version__,'native_scvi_tests_passed':2,'native_test_stdout':result.stdout,'native_test_stderr':result.stderr,'full2000gene_prior_sampling_replay_passed':True,'fullwidth_sample_exemplar_shape':list(first[2].shape),'fullwidth_generated_chunk_cells':2000,'CUDA_initialized':torch.cuda.is_initialized(),'GPU_job_created':False,'real_donor_model_fitted':False,'source_SHA256':{str(p):sha(p) for p in paths},'scope':'Actual native synthetic two-epoch fit/freshload/category/count tests plus full2000gene prior generation/replay via unchanged iter_scvi. No real donor fitting or quality score; CUDA runtime resource/performance still requires bounded actual admission.'}
(ROOT/'analysis/matched_scvi/CPU_model_qualification.json').write_text(json.dumps(q,indent=2)+'\n')
print(json.dumps({k:v for k,v in q.items() if k not in ('source_SHA256','native_test_stdout','native_test_stderr')},indent=2))
