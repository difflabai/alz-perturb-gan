"""Actual full2000gene synthetic two-stage native fit/restore/count checks."""
import datetime,json,os,sys,time,hashlib,tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ROOT=Path('/ml2/alz');SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=[str(ROOT/'analysis/matched_cfgen'),str(ROOT/'cfgen_gpu_qualification_runtime'),str(SCIENCE)]
from fit_adapter import fit_arrays,restore,sample,seed,torch,np,read,validate_complete
import fit_adapter
from stage_law import Schedule
from state_fingerprint import fingerprint
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rng=np.random.default_rng(784);counts=rng.poisson(3,(256,2000)).astype(np.float32);stages=np.arange(4).repeat(64);names=np.array([f'SYNTH_CELL_{i}' for i in range(256)]);genes=np.array([f'SYNTH_GENE_{i}' for i in range(2000)])
with tempfile.TemporaryDirectory(prefix='alz-cfgen2000-CPU-') as tmp:
 out=Path(tmp)/'fit';model,receipt=fit_arrays(counts,stages,names,genes,out,0,'cpu',time.monotonic()+120,synthetic=True)
 assert receipt['AE_updates']==receipt['FM_updates']==2 and receipt['total_optimizer_updates']==4
 assert receipt['frozen_encoder_parameters_buffers_and_modes_preserved'] is True
 before=fingerprint(model.state_dict());features=fingerprint({n:f.state_dict() for n,f in model.feature_embeddings.items()});checkpoint=Path(receipt['FM_checkpoint_path']);pin=sha(checkpoint)
 replay=restore(checkpoint,'cpu');assert fingerprint(replay.state_dict())==before
 assert fingerprint({n:f.state_dict() for n,f in replay.feature_embeddings.items()})==features
 labels=torch.arange(4).repeat_interleave(2);exposure=torch.full((8,1),float(np.log(6000)))
 seed(912);left=sample(model,labels,exposure);seed(912);right=sample(replay,labels,exposure)
 assert torch.equal(left,right) and left.shape==(8,2000) and bool(torch.isfinite(left).all() and (left>=0).all() and (left==left.floor()).all())
 assert sha(checkpoint)==pin
 assert not torch.cuda.is_initialized()
 # Exercise native256cellchunks across400cellbag boundaries and the same
 # method-specific fixed seed. This checks the new streaming wrapper itself.
 schedule=Schedule(np.zeros(200,dtype=int),np.zeros((200,400),dtype=int),np.full((200,400),6000,dtype=np.float32))
 ctx=SimpleNamespace(project=lambda x:x[:,:50].float());condition=np.zeros(7,dtype=np.float32);condition[3]=1/3
 calls=[]
 def tracked(m,labels,exposure):
  calls.append((labels.detach().cpu().clone(),exposure.detach().cpu().clone()))
  return sample(m,labels,exposure)
 with patch.object(fit_adapter,'sample',side_effect=tracked):first=next(fit_adapter.iter_counts(ctx,replay,schedule,condition,72))
 second=next(fit_adapter.iter_counts(ctx,replay,schedule,condition,72))
 assert len(calls)==8 and [len(x[0]) for x in calls]==[256]*7+[208]
 assert all(bool((labels==1).all()) and torch.allclose(exposure,torch.full_like(exposure,float(np.log(6000)))) for labels,exposure in calls)
 assert first[0]==0 and first[1].shape==(5,400,50) and first[2].shape==(1,400,2000) and first[3].shape==(2000,)
 assert all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b for a,b in zip(first,second))
 try:validate_complete(receipt,SCIENCE/'caches/fold0',checkpoint)
 except AssertionError:pass
 else:raise AssertionError('Synthetic short fit passed the real300/1500epoch endpoint gate')
 try:fit_arrays(counts,stages,names,genes,Path(tmp)/'forbidden',0,'cpu',time.monotonic()+1,synthetic=False)
 except AssertionError:pass
 else:raise AssertionError('Real fitting allowed CPU fallback')
 actual={'AE_optimizer_updates':receipt['AE_updates'],'FM_optimizer_updates':receipt['FM_updates'],'frozen_encoder_exact':True,'native_saved_model_reloaded_exact':True,'native_registered_tensors_after_sampling':len(replay.state_dict()),'native_count_replay_exact':True,'native_count_shape':list(left.shape),'native_trace_chunk_to_bag_replay_passed':True,'streamed_native_chunk_sizes':[len(x[0]) for x in calls],'synthetic_endpoint_and_CPU_realfit_refused':True}
p=ROOT/'analysis/matched_cfgen/protocol.json';q=read(p);sources=[Path(__file__),ROOT/'analysis/matched_cfgen/fit_adapter.py',p,*[Path(n) for n in q['source_SHA256']]]
result={'status':'MATCHED_CFGEN_NATIVE_FULL2000GENE_TWO_STAGE_CPU_FIT_REPLAY_PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':actual,'CUDA_initialized':torch.cuda.is_initialized(),'GPU_job_created':False,'real_data_model_fitted':False,'source_SHA256':{str(p):sha(p) for p in sources},'scope':'Actual full-width synthetic2AE+2FMoptimizer updates via native Trainer/objectives. Fixedencoder parameters/BatchNormbuffers/modes preserved; native sealedcheckpoint strictreload andcountreplay pass. Real33-donor CUDA300/1500epoch training, finite performance and calibration still required.'}
(ROOT/'analysis/matched_cfgen/CPU_model_qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_SHA256','scope')},indent=2))
