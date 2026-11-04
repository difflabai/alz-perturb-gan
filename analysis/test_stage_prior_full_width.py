"""Full2000gene native update and unchanged iter_pg with stage prior view, CPU only."""
import json,datetime,sys,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
import stage_pg_adapter as a
import stage_pg_update as u
import stage_prior as sp
from generation import iter_pg
from contracts import sha
ROOT=Path('/ml2/alz')
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1)
 u.configure({'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.})
 x=torch.randint(1,8,(24,2000),generator=torch.Generator().manual_seed(20261104101)).float();ctx=object.__new__(a.Context);ctx.device=torch.device('cpu');ctx.fold=0;ctx.ids=np.arange(6);ctx.by=[np.arange(4*i,4*i+4) for i in ctx.ids];ctx.X=x;ctx.conditions=torch.zeros(6,7);ctx.conditions[:,3]=torch.tensor([0.,1/3,2/3,1.,1.,1.]);ctx.fitc=ctx.conditions.numpy().copy();ctx.mean=torch.zeros(2000);ctx.sd=torch.ones(2000);ctx.W=torch.randn(2000,50,generator=torch.Generator().manual_seed(20261104102))/np.sqrt(2000);ctx.trainpcs=ctx.project(x);ctx.mu=x.mean(0);ctx.theta=(ctx.mu.square()/(x.var(0,unbiased=False)-ctx.mu).clamp_min(1e-4)).clamp(.1,10000)
 calls=[];original_batch=ctx.batch
 def observe_batch(rng):
  result=original_batch(rng);calls.append(result[1].detach().clone());return result
 ctx.batch=observe_batch
 b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0);before=b[3].z.detach().clone();row=a.advance(ctx,b,g,r,usage,execution_limit=20000)
 assert row['G_gradient_norm']>0 and row['prior_gradient_norm']>0 and row['D_gradient_norm']>0
 assert not torch.equal(before,b[3].z) and int(usage.unsampled_moved_steps.sum())==0
 assert torch.equal(usage.draws.reshape(4,128).sum(1),torch.bincount(sp.stages(calls[-1]),minlength=4))
 floats=0
 for model in b[1:4]:
  for v in model.state_dict().values():
   if v.is_floating_point():assert torch.isfinite(v).all();floats+=v.numel()
 assert floats==371177
 c=np.zeros(7,dtype=np.float32);c[3]=2/3;view=sp.view_for_condition(b[3],c);mode=b[3].training
 iterator=iter_pg(ctx,b[1],view,SimpleNamespace(libraries=np.full((200,400),8000,dtype=np.float32)),c,20261104103)
 try:
  start,pcs,counts,totals=next(iterator)
  assert start==0 and pcs.shape==(5,400,50) and counts.shape==(1,400,2000) and np.isfinite(pcs).all() and np.isfinite(counts).all() and (counts>=0).all() and np.equal(counts,np.floor(counts)).all()
 finally:iterator.close()
 assert b[3].training==mode and not torch.cuda.is_initialized()
 result={'status':'PASSED_FULL2000GENE_STAGE_PRIOR_UPDATE_AND_UNCHANGED_NATIVE_GENERATION_VIEW_CPU_CHECK','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_model_floating_values':floats,'native_G_D_dispersion_and_P_gradients_verified':True,'actual_generator_draw_group_counts_match_G_conditions':True,'unsampled_raw_particle_movement_zero':True,'unchanged_iter_pg_first_five_bags_PC_and_integer_count_contract_passed':True,'native_prior_training_mode_restored':True,'CUDA_initialized':False,'quality_measured':False,'source_SHA256':{str(p):sha(p) for p in [Path(__file__).resolve(),* (ROOT/'stage_model').glob('*.py'),Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103/generation.py')]},'scope':'Synthetic fullwidth CPU optimizerpair and first five nativegenerationbags only. No realdata/calibration/evaluationtargets, GPUcontext, qualitymeasurement or nativeallocation. Grouped index law is taskextension, notnewnativepreset.'}
 p=ROOT/'analysis/stage_prior_full_width_CPU_qualification.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
