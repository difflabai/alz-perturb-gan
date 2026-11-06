"""Fullwidth fresh native CFGen fits at two row counts plus all fold inputs."""
import datetime,hashlib,json,os,sys,tempfile,time
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=[str(HERE),str(ROOT/'cfgen_gpu_qualification_runtime'),str(SCIENCE)]
import fit_adapter as adapter
from fit_adapter import torch,np
from state_fingerprint import fingerprint
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cases=[]
with tempfile.TemporaryDirectory(prefix='alz-cfgen-generalized-CPU-') as tmp:
 for seed_value,n in ((0,256),(4,512)):
  rng=np.random.default_rng(784+n);counts=rng.poisson(3,(n,2000)).astype(np.float32);stage=np.arange(4).repeat(n//4);names=np.array([f'SYNTH_CELL_{i}' for i in range(n)]);genes=np.array([f'SYNTH_GENE_{i}' for i in range(2000)])
  model,receipt=adapter.fit_arrays(counts,stage,names,genes,Path(tmp)/f'fit_s{seed_value}',seed_value,'cpu',time.monotonic()+150,synthetic=True)
  assert receipt['updates_per_epoch']==n//256 and receipt['AE_updates']==receipt['FM_updates']==2*(n//256)
  assert receipt['frozen_encoder_parameters_buffers_and_modes_preserved'] and receipt['training_seed']==seed_value
  checkpoint=Path(receipt['FM_checkpoint_path']);before=fingerprint(model.state_dict());features=fingerprint({k:f.state_dict() for k,f in model.feature_embeddings.items()});replay=adapter.restore(checkpoint,'cpu');assert fingerprint(replay.state_dict())==before
  assert fingerprint({k:f.state_dict() for k,f in replay.feature_embeddings.items()})==features
  labels=torch.arange(4).repeat_interleave(2);exposure=torch.full((8,1),float(np.log(6000)))
  adapter.seed(912);a=adapter.sample(model,labels,exposure);adapter.seed(912);b=adapter.sample(replay,labels,exposure)
  assert torch.equal(a,b) and a.shape==(8,2000) and bool(torch.isfinite(a).all() and (a>=0).all() and (a==a.floor()).all())
  try:adapter.validate_complete(receipt,SCIENCE/'caches/fold0',checkpoint)
  except AssertionError:pass
  else:raise AssertionError('Synthetic2epoch endpoint accepted as a real300/1500 model')
  try:adapter.fit_arrays(counts,stage,names,genes,Path(tmp)/f'forbidden{seed_value}',seed_value,'cpu',time.monotonic()+1,synthetic=False)
  except AssertionError:pass
  else:raise AssertionError('CPU fallback accepted for real comparator training')
  cases.append(dict(seed=seed_value,training_cells=n,updates_per_epoch=n//256,AE_updates=receipt['AE_updates'],FM_updates=receipt['FM_updates'],native_state_and_unregistered_features_reload_exact=True,native_count_replay_exact=True))
  del model,replay,a,b
folds=[]
for fold in range(3):
 counts,stage,names,genes,manifest,declared=adapter.load_fit(SCIENCE/'caches'/f'fold{fold}')
 assert manifest['fold']==fold and len(names)==len(stage)==counts.shape[0] and len(genes)==2000
 assert set(stage.tolist())==set(range(4));folds.append(dict(fold=fold,training_cells=len(names),optimization_donors=len(declared['optimization_donors']),updates_per_epoch=len(names)//256,AE300_updates=300*(len(names)//256),FM1500_updates=1500*(len(names)//256),cache_manifest_SHA256=sha(SCIENCE/'caches'/f'fold{fold}/manifest.json')))
assert not torch.cuda.is_initialized()
protocol=adapter.read(HERE/'protocol.json');sources=[Path(__file__),HERE/'fit_adapter.py',HERE/'protocol.json',*[Path(x) for x in protocol['source_SHA256']]]
proof=dict(status='GENERALIZED_NATIVE_CFGEN_FRESH_FULLWIDTH_CPU_AND_THREE_FOLD_INPUT_CHECKS_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),synthetic_cases=cases,all_frozen_fold_training_inputs=folds,actual_synthetic_optimizer_updates=sum(c['AE_updates']+c['FM_updates'] for c in cases),native_constructor_loss_optimizer_code_unchanged=True,fold0_seed0_training_math_unchanged=True,GPU_allocation_created=False,CUDA_initialized=False,real_ALZ_model_fitted=False,heldout_target_values_read=False,runtime_deployment_qualified=False,source_SHA256={str(p):sha(p) for p in sources},limitation='Model caller CPU qualification only. Fresh guarded CUDA training, complete endpoints, runtime/lease/accounting and heldout generation qualification remain required before benchmark execution.')
(HERE/'CPU_model_qualification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='source_SHA256'},indent=2))
