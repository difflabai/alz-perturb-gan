"""Streaming, inference-only disease contrasts from five frozen pure PG models.

No intervention prediction, optimizer, checkpoint selection or evaluation targets.
Synthetic replication measures Monte Carlo stability, never biological sample size.
"""
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/ml2/alz')
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
MODEL=ROOT/'analysis/faithful097/model'
PLAN=ROOT/'analysis/primamente_inference_hypothesis_plan_20261108.json'
STATE=ROOT/'artifacts/primamente-t005-gene-inference-20261108'
sys.path[:0]=list(map(str,[ROOT,MODEL,SCIENCE,PROJECT,PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import numpy as np
import torch
from released_adapter import Context
from particlegan import Recipe
from pg_model import CountGenerator

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):
    p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def fingerprint(module):
    h=hashlib.sha256()
    for k,v in sorted(module.state_dict().items()):
        a=v.detach().cpu().contiguous();h.update(k.encode()+str(a.dtype).encode()+repr(tuple(a.shape)).encode()+a.numpy().tobytes())
    return h.hexdigest()
def normalized(counts):
    return torch.log1p(counts*(10000/counts.sum(-1,keepdim=True).clamp_min(1)))
def aggregate(values,labels,n_types=6):
    out=np.zeros((n_types+1,values.shape[1]),np.float64);n=np.zeros(n_types+1,np.int64)
    a=values.detach().numpy();labels=np.asarray(labels)
    out[0]=a.sum(0,dtype=np.float64);n[0]=len(a)
    for t in range(n_types):
        idx=labels==t;n[t+1]=idx.sum();out[t+1]=a[idx].sum(0,dtype=np.float64)
    return out,n
def load_generation(ctx,item):
    assert sha(item['checkpoint'])==item['checkpoint_SHA256']
    q=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)
    assert q['particlegan_version']=='0.9.7' and q['completed_updates']==q['policy']['completed_steps']==q['fixed_endpoint']==20000
    assert q['seed']==item['seed'] and q['fold']==0 and q['status']=='completed'
    assert q['cache_manifest_SHA256']==sha(ctx.cache/'manifest.json')
    assert q['released_hyperparameters']==read(PLAN)['hyperparameters']
    assert q['variant_config']==dict(z_dim=8,stage_balance=False,cell_z_dim=8,theta_init_scale=1.,fourier_feature_scale=1.)
    rec=Recipe.from_dict(q['recipe']);assert rec.num_particles==512 and rec.ema_decay==0
    G=CountGenerator(ctx.mu.cpu(),ctx.theta.cpu(),zd=rec.z_dim,zc=8);P=rec.make_prior();rec.apply_prior_policy(P)
    for name,module in [('generator',G),('prior',P)]:
        module.load_state_dict(q['policy']['models'][name],strict=True);module.eval()
        for k,v in module.state_dict().items():assert torch.equal(v,q['policy']['models'][name][k])
    del q
    return G,P
def self_test():
    x=torch.tensor([[0.,2.],[3.,1.],[0.,0.]])
    v=normalized(x);np.testing.assert_allclose(v.numpy(),np.log1p(x.numpy()*10000/np.maximum(x.numpy().sum(1,keepdims=True),1)),rtol=1e-6)
    a,n=aggregate(v,[0,1,1]);assert n.tolist()==[3,1,2,0,0,0,0]
    np.testing.assert_allclose(a[0],v.numpy().sum(0),rtol=1e-6)
    np.testing.assert_allclose(a[2],v.numpy()[1:].sum(0),rtol=1e-6)
    assert not torch.cuda.is_initialized()
    return dict(normalization_and_zero_libraries=True,streaming_group_sums_and_empty_types=True,no_CUDA=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_grad_enabled(False);assert not torch.cuda.is_initialized()
    if args.check:print(json.dumps(self_test()));return
    assert args.run
    plan=read(PLAN)
    for p,pin in plan['input_SHA256'].items():assert sha(p)==pin,p
    assert sha(__file__)==plan['source_SHA256'] and not STATE.exists();STATE.mkdir()
    started=time.monotonic()
    def timeout(signum,frame):raise TimeoutError('Finite inference-only CPU allowance ended')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['CPU_wall_cap_seconds'])
    write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=sha(__file__),new_optimizer_updates=0))
    try:
        self_test();ctx=Context(Path(plan['fitting_cache']),'cpu')
        with np.load(ctx.cache/'train/metadata.npz',allow_pickle=False) as z:
            genes=z['genes'].copy();labels=z['type'].copy();donors=z['donor'].copy();stages=np.rint(z['raw_c'][:,3]*3).astype(int)
        assert set(ctx.ids)==set(plan['optimization_donors']) and len(genes)==2000
        actual=normalized(ctx.X);real=np.zeros((len(ctx.ids),7,len(genes)),np.float64);sizes=np.zeros((len(ctx.ids),7),np.int64)
        for i,d in enumerate(ctx.ids):real[i],sizes[i]=aggregate(actual[donors==d],labels[donors==d])
        np.savez_compressed(STATE/'real_training_donor_summaries.npz',genes=genes,donors=ctx.ids,stages=stages,sums=real,n=sizes)
        del actual
        libraries=ctx.X.sum(1).numpy();by=[np.flatnonzero(donors==d) for d in ctx.ids]
        completed=[];generated=0
        for item in plan['models']:
            G,P=load_generation(ctx,item);before=(fingerprint(G),fingerprint(P));seed=item['seed']
            for rep in range(plan['generation_replicates']):
                rng=np.random.default_rng(np.random.SeedSequence([20261108,710,rep]));template_donors=rng.integers(len(by),size=plan['bags_per_stage_replicate'])
                lib=torch.as_tensor(np.stack([libraries[rng.choice(by[d],plan['cells_per_bag'],replace=True)] for d in template_donors]))
                z,_=P.sample(plan['bags_per_stage_replicate'],generator=torch.Generator().manual_seed(20261108+rep*1000))
                for stage in range(4):
                    c=torch.zeros(plan['chunk_bags'],7);c[:,3]=stage/3;sums=np.zeros((7,len(genes)),np.float64);n=np.zeros(7,np.int64);det=np.zeros_like(sums);bagmeans=[];confidence=[]
                    for start in range(0,len(lib),plan['chunk_bags']):
                        stop=min(len(lib),start+plan['chunk_bags']);count=stop-start;common=20261108+rep*100000+start
                        torch.manual_seed(common);cellgen=torch.Generator().manual_seed(common+1);countgen=torch.Generator().manual_seed(common+2)
                        rate=G.rates(z[start:stop],c[:count],plan['cells_per_bag'],lib[start:stop],cellgen)
                        counts=torch.poisson(rate,generator=countgen).reshape(-1,len(genes))
                        assert torch.isfinite(counts).all() and (counts>=0).all() and (counts==counts.floor()).all()
                        log=normalized(counts);pcs=ctx.project(counts);prob=ctx.space.clf(pcs).softmax(-1);typ=prob.argmax(-1).numpy()
                        a,b=aggregate(log,typ);sums+=a;n+=b;d,_=aggregate((counts>0).float(),typ);det+=d
                        bagmeans.append(log.reshape(count,plan['cells_per_bag'],len(genes)).mean(1).numpy());confidence.append(prob.max(-1).values.mean().item())
                    dest=STATE/f'seed{seed}_rep{rep}_stage{stage}.npz'
                    np.savez_compressed(dest,sums=sums,n=n,detection_counts=det,bag_mean_logCP10k=np.concatenate(bagmeans),classifier_mean_confidence=np.mean(confidence))
                    generated+=int(n[0]);completed.append(dict(seed=seed,replicate=rep,stage=stage,path=str(dest),SHA256=sha(dest),cells=int(n[0])))
                    write(STATE/'progress.json',dict(status='INFERENCE_ONLY_STAGE_ASSOCIATION_STUDY_RUNNING',completed_stage_replicates=len(completed),planned_stage_replicates=160,generated_cells=generated,planned_cells=plan['total_generated_cells'],latest=dict(seed=seed,replicate=rep,stage=stage),elapsed_seconds=time.monotonic()-started,new_optimizer_updates=0,new_training_fits=0,CUDA_initialized=False,causal_intervention_prediction=False))
                assert before==(fingerprint(G),fingerprint(P)),'Inference changed saved model state'
            assert sha(item['checkpoint'])==item['checkpoint_SHA256'];del G,P;gc.collect()
        for p,pin in plan['input_SHA256'].items():assert sha(p)==pin,p
        assert generated==plan['total_generated_cells'] and not torch.cuda.is_initialized()
        result=dict(status='FIVE_FROZEN_PG_MODELS_12P8M_CELLS_INFERENCE_ONLY_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=5,stage_replicates=completed,generated_cells=generated,new_optimizer_updates=0,new_training_fits=0,source_weights_unchanged=True,CUDA_initialized=False,evaluation_or_calibration_targets_read=False,causal_intervention_prediction=False,real_biological_donors=len(ctx.ids),synthetic_cells_are_not_additional_biological_replicates=True,elapsed_seconds=time.monotonic()-started,plan_SHA256=sha(PLAN),hypothesis_ranking_and_literature_validation_pending=True)
        write(STATE/'completed.json',result);print(json.dumps({k:v for k,v in result.items() if k!='stage_replicates'}),flush=True)
    except BaseException as exc:
        write(STATE/'failure.json',dict(error=repr(exc),new_optimizer_updates=0,new_training_fits=0,retained_partial_outputs=True,CUDA_initialized=torch.cuda.is_initialized()));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
