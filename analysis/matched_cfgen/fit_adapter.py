"""Unmodified native CFGen constructors with matched data and fixed endpoints."""
import gc,json,sys,time,random
from pathlib import Path
ROOT=Path('/ml2/alz');TARGET=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/execution-control-20261103')
sys.path.insert(0,str(TARGET))
from target_model import dataset,ae,fm,restore,sample,seed,torch,np,RECIPE,NETWORK
from pytorch_lightning import Trainer,Callback
from state_fingerprint import fingerprint
from contracts import checked_cache,validate_train_donors,sha
import scipy.sparse as sp
PROTOCOL=ROOT/'analysis/matched_cfgen/protocol.json'
def read(p):return json.loads(Path(p).read_bytes())
def dump(p,q):Path(p).write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
def load_fit(cache):
 manifest,fold=checked_cache(cache,'train');cache=Path(cache);m=np.load(cache/'train/metadata.npz',allow_pickle=False)
 validate_train_donors(m['donor'],fold['optimization_donors'])
 assert list(m['donors'])==fold['optimization_donors'] and len(m['donor'])==14762
 raw=dict(zip(m['donors'],m['raw_c']));stage=np.array([int(round(raw[d][3]*3)) for d in m['donor']])
 counts=sp.load_npz(cache/'train/counts_hvg.npz').astype(np.float32)
 assert counts.shape==(14762,2000) and np.isfinite(counts.data).all() and (counts.data>=0).all() and np.equal(counts.data,np.floor(counts.data)).all()
 assert (np.asarray(counts.sum(1)).ravel()>0).all()
 return counts,stage,m['cell'],m['genes'],manifest,fold
class Audit(Callback):
 def __init__(self,out,phase,end,offset,frozen=None):
  self.out=Path(out);self.phase=phase;self.end=end;self.offset=offset;self.frozen=frozen;self.completed_epochs=0
  self.last_loss=None;self.optimizer_pairs=0;self.frozen_checks=[]
 def on_train_batch_start(self,trainer,model,*args):
  if time.monotonic()>=self.end:raise TimeoutError('Finite matchedCFGen allocation ending')
  if self.frozen is not None:
   model.encoder_model.eval()
   assert not any(p.requires_grad for p in model.encoder_model.parameters())
 def on_before_optimizer_step(self,trainer,model,optimizer):
  grads=[p.grad for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
  assert grads and bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()),'Nonfinite nativeCFGen gradient'
 def check_frozen(self,model):
  if self.frozen is not None:
   current=fingerprint(model.encoder_model.state_dict());assert current==self.frozen,'Fixed AE parameters or BatchNorm buffers changed'
   assert not any(m.training for m in model.encoder_model.modules())
   self.frozen_checks.append({'flow_updates':self.optimizer_pairs,'encoder_fingerprint':current})
 def on_train_batch_end(self,trainer,model,outputs,*args):
  loss=outputs['loss'] if isinstance(outputs,dict) else outputs;assert bool(torch.isfinite(loss))
  self.optimizer_pairs=int(trainer.global_step)
  if self.optimizer_pairs%250==0 or self.optimizer_pairs==1:
   self.last_loss=float(loss.detach().cpu());q={'phase':self.phase,'phase_optimizer_updates':self.optimizer_pairs,'total_optimizer_updates':self.offset+self.optimizer_pairs,'completed_epochs':self.completed_epochs,'native_loss':self.last_loss}
   dump(self.out.parent/'progress.json',q)
   with (self.out.parent/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(q)+'\n')
  if self.optimizer_pairs%5000==0:self.check_frozen(model)
 def on_train_epoch_end(self,trainer,model):
  self.completed_epochs+=1
  if self.completed_epochs%10==0:
   self.check_frozen(model);path=self.out/'recovery.ckpt';temporary=path.with_suffix('.partial.ckpt');trainer.save_checkpoint(temporary);temporary.replace(path)
 def on_save_checkpoint(self,trainer,model,checkpoint):
  checkpoint['matched_CFGen_RNG']={'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,'completed_epochs':self.completed_epochs,'optimizer_pairs':int(trainer.global_step),'resume_exact_replay_qualified':False}
def train_phase(model,loader,out,phase,epochs,device,end,offset=0,frozen=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);cb=Audit(out,phase,end,offset,frozen)
 trainer=Trainer(accelerator='cpu' if device=='cpu' else 'gpu',devices=1,max_epochs=epochs,min_epochs=epochs,limit_val_batches=0,num_sanity_val_steps=0,logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,default_root_dir=out,deterministic=True,precision='32-true',callbacks=[cb],gradient_clip_val=1.,detect_anomaly=True,log_every_n_steps=1)
 trainer.fit(model,train_dataloaders=loader)
 assert cb.completed_epochs==epochs and trainer.global_step==epochs*len(loader)
 # Native teardown moves registered tensors toCPU. Keep the native sealed
 # checkpoint intact; fresh inference explicitly binds all devices together.
 cb.check_frozen(model);path=out/'final.ckpt';trainer.save_checkpoint(path)
 receipt={'phase':phase,'completed_epochs':cb.completed_epochs,'optimizer_updates':int(trainer.global_step),'checkpoint_path':str(path),'checkpoint_SHA256':sha(path),'fixed_endpoint':epochs,'frozen_encoder_checks':cb.frozen_checks,'native_losses_finite':True,'no_validation_or_early_selection':True}
 dump(out/'receipt.json',receipt);return model,receipt
def fit_arrays(counts,stage,names,genes,out,seed_value,device,end,*,synthetic=False):
 p=read(PROTOCOL);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 if synthetic:
  assert device=='cpu' and counts.shape==(256,2000) and all(str(n).startswith('SYNTH_') for n in names)
  epochs=(2,2)
 else:
  assert device=='cuda:0' and counts.shape==(14762,2000) and seed_value==0
  epochs=(p['AE_epochs'],p['FM_epochs'])
 assert set(np.asarray(stage).tolist())=={0,1,2,3};seed(seed_value)
 data,loader=dataset(counts,stage,names,genes);assert len(loader)==(1 if synthetic else 57)
 assert data.id2cov['stage']=={0:0,1:1,2:2,3:3}
 encoder=ae();encoder,ae_receipt=train_phase(encoder,loader,out/'AE','AE',epochs[0],device,end)
 for parameter in encoder.parameters():parameter.requires_grad=False
 encoder.eval();fixed=fingerprint(encoder.state_dict());model=fm(encoder,data,device)
 model,flow_receipt=train_phase(model,loader,out/'FM','FM',epochs[1],device,end,offset=ae_receipt['optimizer_updates'],frozen=fixed)
 assert fingerprint(model.encoder_model.state_dict())==fixed
 receipt={'status':'completed','method':'cfgen','AE_epochs':epochs[0],'FM_epochs':epochs[1],'AE_updates':ae_receipt['optimizer_updates'],'FM_updates':flow_receipt['optimizer_updates'],'total_optimizer_updates':ae_receipt['optimizer_updates']+flow_receipt['optimizer_updates'],'encoder_fingerprint':fixed,'FM_checkpoint_path':flow_receipt['checkpoint_path'],'FM_checkpoint_SHA256':flow_receipt['checkpoint_SHA256'],'AE_receipt_SHA256':sha(out/'AE/receipt.json'),'FM_receipt_SHA256':sha(out/'FM/receipt.json'),'frozen_encoder_parameters_buffers_and_modes_preserved':True,'training_seed':seed_value,'training_cells':len(names),'training_information':['stage'],'validation_cells':0,'early_stopping':False,'native_recipe':RECIPE,'synthetic':synthetic,'protocol_SHA256':sha(PROTOCOL),'network_attempts':NETWORK}
 dump(out/'fit_receipt.json',receipt);return model,receipt
def fit(cache,out,seed_value,device,end):
 counts,stage,names,genes,manifest,fold=load_fit(cache);model,receipt=fit_arrays(counts,stage,names,genes,out,seed_value,device,end)
 receipt.update(fold=manifest['fold'],cache_manifest_SHA256=sha(Path(cache)/'manifest.json'),training_donors=fold['optimization_donors'])
 dump(Path(out)/'fit_receipt.json',receipt);return model,receipt
def validate_complete(receipt,cache,checkpoint):
 p=read(PROTOCOL)
 assert receipt['status']=='completed' and receipt['synthetic'] is False and receipt['method']=='cfgen'
 assert receipt['AE_epochs']==300 and receipt['FM_epochs']==1500 and receipt['AE_updates']==17100 and receipt['FM_updates']==85500
 assert receipt['total_optimizer_updates']==102600 and receipt['training_seed']==0 and receipt['fold']==0
 assert receipt['training_cells']==14762 and len(receipt['training_donors'])==33 and receipt['training_information']==['stage']
 assert receipt['validation_cells']==0 and receipt['early_stopping'] is False and receipt['frozen_encoder_parameters_buffers_and_modes_preserved'] is True
 assert receipt['protocol_SHA256']==sha(PROTOCOL) and receipt['cache_manifest_SHA256']==sha(Path(cache)/'manifest.json')
 assert receipt['FM_checkpoint_path']==str(checkpoint) and receipt['FM_checkpoint_SHA256']==sha(checkpoint)
def iter_counts(ctx,model,schedule,c,s):
 from generation import validate_counts
 stage=int(round(float(c[3])*3));draw=s+520000000;seed(draw);device=next(model.parameters()).device;model.eval()
 for start in range(0,200,5):
  bags=min(5,200-start);libraries=schedule.libraries[start:start+bags].reshape(-1);pcs=[];totals=[];exemplar=[];kept=0
  for lo in range(0,len(libraries),256):
   lib=torch.as_tensor(libraries[lo:lo+256],device=device);labels=torch.full((len(lib),),stage,device=device,dtype=torch.long)
   counts=sample(model,labels,lib.log()[:,None]);validate_counts(counts,len(lib),2000)
   pcs.append(ctx.project(counts).cpu().numpy());totals.append(counts.sum(-1).cpu().numpy())
   if kept<400:
    take=min(400-kept,len(counts));exemplar.append(counts[:take].cpu().numpy());kept+=take
  yield start,np.concatenate(pcs).reshape(bags,400,50),np.concatenate(exemplar)[None],np.concatenate(totals)
