"""Recover an interrupted inference batch without altering frozen models or seeds."""
import importlib.util
import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time

ROOT=Path('/ml2/alz')
spec=importlib.util.spec_from_file_location('original_inference',ROOT/'analysis/infer_t005_gene_hypotheses_cpu.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
np,torch=m.np,m.torch
OLD=m.STATE
STATE=ROOT/'artifacts/primamente-t005-gene-inference-recovery-v1-20261108'

def validate(path):
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)=={'sums','n','detection_counts','bag_mean_logCP10k','classifier_mean_confidence'}
        a={k:z[k].copy() for k in z.files}
    assert a['sums'].shape==a['detection_counts'].shape==(7,2000)
    assert a['n'].shape==(7,) and a['n'][0]==80000 and a['n'][1:].sum()==80000
    assert a['bag_mean_logCP10k'].shape==(200,2000)
    assert all(np.isfinite(v).all() for v in a.values())
    assert (a['sums']>=0).all() and (a['detection_counts']>=0).all()
    assert (a['detection_counts']<=a['n'][:,None]).all()
    np.testing.assert_allclose(a['sums'][0],a['sums'][1:].sum(0),rtol=1e-12,atol=1e-8)
    assert 0<=a['classifier_mean_confidence']<=1
    return a

def case(ctx,G,z,lib,rep,stage,plan):
    c=torch.zeros(plan['chunk_bags'],7);c[:,3]=stage/3
    sums=np.zeros((7,2000),np.float64);n=np.zeros(7,np.int64);det=np.zeros_like(sums);bagmeans=[];confidence=[]
    for start in range(0,len(lib),plan['chunk_bags']):
        stop=min(len(lib),start+plan['chunk_bags']);count=stop-start;common=20261108+rep*100000+start
        torch.manual_seed(common);cellgen=torch.Generator().manual_seed(common+1);countgen=torch.Generator().manual_seed(common+2)
        rate=G.rates(z[start:stop],c[:count],plan['cells_per_bag'],lib[start:stop],cellgen)
        counts=torch.poisson(rate,generator=countgen).reshape(-1,2000)
        assert torch.isfinite(counts).all() and (counts>=0).all() and (counts==counts.floor()).all()
        log=m.normalized(counts);pcs=ctx.project(counts);prob=ctx.space.clf(pcs).softmax(-1);typ=prob.argmax(-1).numpy()
        a,b=m.aggregate(log,typ);sums+=a;n+=b;d,_=m.aggregate((counts>0).float(),typ);det+=d
        bagmeans.append(log.reshape(count,plan['cells_per_bag'],2000).mean(1).numpy());confidence.append(prob.max(-1).values.mean().item())
    return dict(sums=sums,n=n,detection_counts=det,bag_mean_logCP10k=np.concatenate(bagmeans),classifier_mean_confidence=np.mean(confidence))

def main():
    torch.set_num_threads(1);torch.set_grad_enabled(False);assert not torch.cuda.is_initialized()
    plan=m.read(m.PLAN);assert m.sha(m.__file__)==plan['source_SHA256']
    for p,pin in plan['input_SHA256'].items():assert m.sha(p)==pin,p
    old=m.read(OLD/'identity.json');p=Path('/proc')/str(old['pid'])/'stat'
    assert not p.exists() or int(p.read_text().rsplit(')',1)[1].split()[19])!=old['birth'],'Original owner still exists'
    assert not STATE.exists();STATE.mkdir()
    started=time.monotonic()
    m.write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=m.sha(__file__),original_source_SHA256=m.sha(m.__file__),original_owner=old,new_optimizer_updates=0))
    def timeout(*args):raise TimeoutError('Finite recovery CPU allowance ended')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(7200)
    try:
        existing=sorted(OLD.glob('seed*_rep*_stage*.npz'));assert len(existing)==87
        reused=[]
        for p in existing:
            validate(p);dest=STATE/p.name;shutil.copyfile(p,dest);assert m.sha(p)==m.sha(dest)
            reused.append(dict(path=str(p),recovered_path=str(dest),SHA256=m.sha(p)))
        real=OLD/'real_training_donor_summaries.npz';shutil.copyfile(real,STATE/real.name)
        m.write(STATE/'handoff.json',dict(original_identity=old,original_owner_confirmed_absent=True,reused_cases=reused,real_summary_SHA256=m.sha(real),original_plan_SHA256=m.sha(m.PLAN),recovery_source_SHA256=m.sha(__file__),new_training_fits=0,new_optimizer_updates=0,unique_target_cells=12800000,replay_cells_planned=240000))
        ctx=m.Context(Path(plan['fitting_cache']),'cpu')
        with np.load(ctx.cache/'train/metadata.npz',allow_pickle=False) as z:
            donors=z['donor'].copy();genes=z['genes'].copy()
        assert len(genes)==2000 and set(ctx.ids)==set(plan['optimization_donors'])
        libraries=ctx.X.sum(1).numpy();by=[np.flatnonzero(donors==d) for d in ctx.ids]
        completed=[];replayed=[];new_cases=0
        for item in plan['models']:
            seed=item['seed'];G,P=m.load_generation(ctx,item);before=(m.fingerprint(G),m.fingerprint(P))
            for rep in range(8):
                rng=np.random.default_rng(np.random.SeedSequence([20261108,710,rep]));template_donors=rng.integers(len(by),size=plan['bags_per_stage_replicate'])
                lib=torch.as_tensor(np.stack([libraries[rng.choice(by[d],plan['cells_per_bag'],replace=True)] for d in template_donors]))
                z,_=P.sample(plan['bags_per_stage_replicate'],generator=torch.Generator().manual_seed(20261108+rep*1000))
                for stage in range(4):
                    dest=STATE/f'seed{seed}_rep{rep}_stage{stage}.npz';reuse=dest.exists()
                    if reuse and rep==stage==0:
                        a=validate(dest);b=case(ctx,G,z,lib,rep,stage,plan)
                        for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=f'Replay mismatch {seed}/{k}')
                        replayed.append(dict(seed=seed,replicate=rep,stage=stage,cells=80000,all_saved_arrays_bitwise_equal=True))
                        m.write(STATE/'replay_verification.json',dict(cases=replayed,physical_replay_cells=80000*len(replayed)))
                    elif not reuse:
                        a=case(ctx,G,z,lib,rep,stage,plan);np.savez_compressed(dest,**a);validate(dest);new_cases+=1
                    completed.append(dict(seed=seed,replicate=rep,stage=stage,path=str(dest),SHA256=m.sha(dest),cells=80000,reused_from_original=reuse))
                    m.write(STATE/'progress.json',dict(status='RECOVERED_INFERENCE_ONLY_RUNNING',completed_stage_replicates=len(completed),planned_stage_replicates=160,reused_stage_replicates=87,new_stage_replicates_completed=new_cases,generated_cells=80000*len(completed),unique_saved_cells=6960000+80000*new_cases,planned_cells=12800000,latest=dict(seed=seed,replicate=rep,stage=stage),elapsed_seconds=time.monotonic()-started,new_optimizer_updates=0,new_training_fits=0,CUDA_initialized=False))
                assert before==(m.fingerprint(G),m.fingerprint(P))
            assert m.sha(item['checkpoint'])==item['checkpoint_SHA256'];del G,P;gc.collect()
        for p,pin in plan['input_SHA256'].items():assert m.sha(p)==pin,p
        assert new_cases==73 and len(replayed)==3 and not torch.cuda.is_initialized()
        result=dict(status='FIVE_FROZEN_PG_MODELS_12P8M_CELLS_RECOVERED_INFERENCE_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=5,stage_replicates=completed,generated_cells=12800000,reused_cases=87,new_cases=73,physical_replay_cells=240000,new_optimizer_updates=0,new_training_fits=0,source_weights_unchanged=True,CUDA_initialized=False,evaluation_or_calibration_targets_read=False,causal_intervention_prediction=False,real_biological_donors=len(ctx.ids),synthetic_cells_are_not_additional_biological_replicates=True,elapsed_seconds=time.monotonic()-started,plan_SHA256=m.sha(m.PLAN),recovery_source_SHA256=m.sha(__file__),replay_verification_SHA256=m.sha(STATE/'replay_verification.json'),hypothesis_ranking_and_literature_validation_pending=True)
        m.write(STATE/'completed.json',result);print(json.dumps({k:v for k,v in result.items() if k!='stage_replicates'}),flush=True)
    except BaseException as exc:
        m.write(STATE/'failure.json',dict(error=repr(exc),retained_partial_outputs=True,new_training_fits=0,new_optimizer_updates=0));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
