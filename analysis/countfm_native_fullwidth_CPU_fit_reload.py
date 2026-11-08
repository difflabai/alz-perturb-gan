"""One fresh synthetic public count-FM fit and native inference restoration.

This qualifies an API path, not an ALZ baseline or inference convergence.
"""
import datetime,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');UPSTREAM=ROOT/'artifacts/comparator-source-review-20261108/count-flow-matching';OUT=ROOT/'artifacts/countfm-native-fullwidth-CPU-fit-reload-20261108';SITE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/isolated/cfgen-cu124-py311/lib/python3.11/site-packages')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists();OUT.mkdir();sys.path[:0]=[str(UPSTREAM),str(SITE)]
 import numpy as np,torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261108)
 from count_fm.experiments.scrna_countfm import CountFMModel,save_countfm,load_countfm
 def fingerprint(model):
  h=hashlib.sha256()
  for key,value in sorted(model.net.state_dict().items()):h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip();assert commit=='7ce372db78581451103baf1afdb8fecd50e12230'
 pins={str(p):sha(p) for p in [*UPSTREAM.glob('count_fm/**/*.py'),Path(__file__).resolve()]};start=time.monotonic();data=np.random.default_rng(20261108).poisson(.5,size=(32,2000)).astype(np.int64);assert data.min()>=0
 model=CountFMModel(device='cpu',num_epochs=1,batch_size=32);model.fit(data,resume=False)
 steps=[int(state['step']) for state in model._opt.state.values() if 'step' in state];assert steps and set(steps)=={1};assert model.d==2000 and math.isfinite(float(model.train_loss))
 assert all(torch.isfinite(p).all() for p in model.net.state_dict().values());before=fingerprint(model);save_countfm(model,str(OUT/'native'));restored=load_countfm(str(OUT/'native'),device='cpu');assert fingerprint(restored)==before
 x=torch.ones((2,2001));x[:,-1]=.5;model.net.eval();restored.net.eval()
 with torch.no_grad():torch.testing.assert_close(model.net(x),restored.net(x),rtol=0,atol=0)
 torch.manual_seed(20261108);a=model.sample(2,n_step=8,batch_eval=2);torch.manual_seed(20261108);b=restored.sample(2,n_step=8,batch_eval=2);np.testing.assert_array_equal(a,b)
 assert a.shape==(2,2000) and a.dtype==np.int64 and (a>=0).all();assert fingerprint(model)==fingerprint(restored)==before;np.savez_compressed(OUT/'counts.npz',original=a,reloaded=b)
 for p,pin in pins.items():assert sha(p)==pin,p
 assert not torch.cuda.is_initialized()
 result=dict(status='UNMODIFIED_COUNTFM_PUBLIC_FULL2000GENE_SYNTHETIC_FRESH_FIT_NATIVE_RELOAD_INTEGER_SAMPLING_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_repository_commit=commit,source_SHA256=pins,python=sys.version,torch_version=torch.__version__,CPU_seconds=time.monotonic()-start,fullwidth_genes=2000,synthetic_training_cells=32,synthetic_training_epochs=1,synthetic_optimizer_steps=1,model_parameters=sum(p.numel() for p in model.net.parameters()),native_architecture=dict(d_model=model.d_model,depth=model.depth,n_heads=model.n_heads,mlp_ratio=model.mlp_ratio,chunk_size=model.chunk_size,dropout=model.dropout),native_training_loss=float(model.train_loss),native_saved_state_fingerprint=before,native_checkpoint_path=str(OUT/'native/countfm.pt'),native_checkpoint_SHA256=sha(OUT/'native/countfm.pt'),native_rates_bitwise_restored=True,seeded_integer_samples_bitwise_restored=True,retained_counts_path=str(OUT/'counts.npz'),retained_counts_SHA256=sha(OUT/'counts.npz'),inference_only_grid_steps=8,inference_convergence_claimed=False,synthetic_only=True,real_ALZ_data_read=False,real_ALZ_fits=0,GPU_allocations=0,CUDA_initialized=False,upstream_modified=False,PG_or_frozen45_protocol_changed=False,remaining='This public scRNA adapter is unconditional. Stage/library conditioning, fresh ALZ fit recipe, converged count-process inference, exact scoring traces and guarded GPU resource/timing qualification remain unexecuted. Native save/load is model-state inference restoration; optimizer-resume fidelity was not claimed or used.',full_goal_complete=False,SOTA_claim=False)
 (OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ('status','CPU_seconds','model_parameters','synthetic_optimizer_steps','real_ALZ_fits','GPU_allocations')}),flush=True)
if __name__=='__main__':main()
