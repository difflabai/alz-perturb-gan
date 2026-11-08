"""Synthetic data-sampler probe, not a deployable or trained new PG arm.

The frozen configure() guard must reject stage_balance=True. Only this isolated
CPU fixture temporarily exercises the existing dormant Context.batch branch.
No production source, authorization, model, optimizer or running process changes.
"""
import ast
import copy
import datetime
import hashlib
import importlib.metadata
import json
import os
import signal
import sys
import urllib.request
from pathlib import Path

ROOT=Path('/ml2/alz')
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
MODEL=ROOT/'analysis/faithful097/model'
sys.path[:0]=list(map(str,[MODEL,ROOT,SCIENCE,PROJECT,
    PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',
    PROJECT/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import numpy as np
import torch
import released_adapter as adapter
import released_update as update


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sample(ctx, balanced):
    update.CONFIG['stage_balance']=balanced  # CPU fixture only; never configure a real study.
    rng=np.random.default_rng(20261108)
    h=hashlib.sha256();stage_counts=np.zeros(4,dtype=np.int64)
    donor_counts=np.zeros(len(ctx.by),dtype=np.int64)
    for _ in range(100):
        real,c,libs=ctx.batch(rng)
        assert real.shape==(32,150) and c.shape==(32,7) and libs.shape==(32,128)
        assert all(torch.isfinite(x).all() for x in (real,c,libs))
        ds=libs[:,0].to(torch.int64).numpy()-1
        assert np.all((ds>=0)&(ds<len(ctx.by)))
        np.testing.assert_array_equal(libs.numpy(),np.broadcast_to(ds[:,None]+1,(32,128)))
        stages=np.rint(c[:,3].numpy()*3).astype(int)
        np.testing.assert_array_equal(stages,np.rint(ctx.fitc[ds,3]*3).astype(int))
        counts=np.bincount(stages,minlength=4)
        if balanced:np.testing.assert_array_equal(counts,np.full(4,8))
        stage_counts+=counts;donor_counts+=np.bincount(ds,minlength=len(ctx.by))
        for x in (real,c,libs):h.update(x.contiguous().numpy().tobytes())
    return dict(batches=100,bags=3200,stages=stage_counts.tolist(),
        stage_frequencies=(stage_counts/3200).tolist(),donor_bag_counts=donor_counts.tolist(),
        data_stream_SHA256=h.hexdigest(),final_numpy_rng_state=rng.bit_generator.state)


def function_pin(path,name):
    source=Path(path).read_text();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()


def main():
    output=ROOT/'analysis/stagebalanced_sampler_CPU_diagnostic_20261108.json'
    assert not output.exists()
    def timeout(*_):raise TimeoutError('Finite300s synthetic sampler probe ended')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    source_paths=[Path(__file__).resolve(),MODEL/'released_adapter.py',MODEL/'released_update.py',
        MODEL/'release_env.py',SCIENCE/'critic.py',SCIENCE/'stage_law.py']
    source_pins={str(p):sha(p) for p in source_paths}
    native=read(ROOT/'analysis/distribution097/native_release.json')
    for p,pin in native['official_wheel_source_SHA256'].items():assert sha(p)==pin,p
    assert importlib.metadata.version('particlegan')=='0.9.7'
    with urllib.request.urlopen('https://pypi.org/pypi/particlegan/json',timeout=25) as stream:release=json.load(stream)
    wheel=next(p for p in release['urls'] if p['filename']=='particlegan-0.9.7-py3-none-any.whl')
    assert release['info']['version']=='0.9.7' and wheel['digests']['sha256']=='8e6e748c84984e9fe979261c262e0a1300fe1225d1f3bc63c6d6a2e2f94c8189'
    metadata=SCIENCE/'caches/fold0/train/metadata.npz'
    with np.load(metadata,allow_pickle=False) as m:
        stage_counts=np.bincount(np.rint(m['raw_c'][:,3]*3).astype(int),minlength=4)
    assert stage_counts.tolist()==[3,4,8,18]
    # Synthetic cohort has only the same donor-count imbalance. It contains
    # no ALZ cell counts, PCs, identities or fitted model parameters.
    stages=np.repeat(np.arange(4),stage_counts);n=len(stages)
    ctx=adapter.Context.__new__(adapter.Context)
    ctx.device=torch.device('cpu')
    ctx.ids=np.arange(n)
    ctx.fitc=np.zeros((n,7),dtype=np.float32);ctx.fitc[:,3]=stages/3
    ctx.conditions=torch.from_numpy(ctx.fitc.copy())
    ctx.by=[np.array([2*d,2*d+1],dtype=int) for d in range(n)]
    ctx.trainpcs=torch.from_numpy(np.random.default_rng(274).normal(size=(2*n,50)).astype(np.float32))
    ctx.X=torch.zeros((2*n,2000));ctx.X[:,0]=torch.from_numpy(np.repeat(np.arange(1,n+1),2)).float()
    config_before=copy.deepcopy(update.CONFIG);hyper_before=copy.deepcopy(update.HYPER)
    hyper=read(ROOT/'analysis/dispersion097/protocol.json')['hyperparameters']
    assert hyper==dict(generator_lr=.012,critic_lr=.018,particle_lr=.06,reg_coeff=1.,reg_kappa=1.,prior_reg=.01)
    try:
        rejected=False
        try:update.configure(dict(config_before,stage_balance=True))
        except ValueError as exc:rejected='matched fixed PG8 architecture' in str(exc)
        assert rejected,'Frozen production scope must continue rejecting the proposed data-weight arm'
        update.configure(config_before);update.configure_hyperparameters(hyper)
        rows={};recipes={}
        for balanced in (False,True):
            first=sample(ctx,balanced);second=sample(ctx,balanced)
            assert first==second,'Synthetic sampler must replay exactly from its own fresh RNG'
            rows['balanced' if balanced else 'uniform_donor']=first
            rec=update.recipe();recipes[str(balanced)]=rec.to_dict()
            assert rec.name=='bispf_optimized' and rec.loss=='non_saturating' and rec.num_particles==512
            assert rec.total_steps==20000 and rec.ema_decay==0 and rec.lr==.012 and rec.d_lr_mult==1.5 and rec.prior_lr_mult==5.
        assert recipes['False']==recipes['True']
        assert rows['balanced']['stages']==[800]*4
        assert rows['uniform_donor']['data_stream_SHA256']!=rows['balanced']['data_stream_SHA256']
    finally:
        update.CONFIG.clear();update.CONFIG.update(config_before)
        update.HYPER.clear();update.HYPER.update(hyper_before)
    assert update.CONFIG==config_before and update.HYPER==hyper_before
    assert not torch.cuda.is_initialized()
    for p,pin in source_pins.items():assert sha(p)==pin,p
    record=dict(status='SYNTHETIC_STAGE_SAMPLER_BALANCING_AND_NATIVE_RECIPE_IDENTITY_CHECK_PASS',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_official_PyPI_release='0.9.7',
        official_wheel_SHA256=wheel['digests']['sha256'],real_fitting_donor_counts_by_stage=stage_counts.tolist(),
        real_fitting_donor_proportions=(stage_counts/stage_counts.sum()).tolist(),
        source_metadata_only=True,real_ALZ_cell_counts_or_PCs_read=False,
        synthetic_context_cells=2*n,synthetic_context_genes=2000,conditions=7,critic_features=150,
        actual_synthetic_batch_calls=400,unique_batches_per_arm=100,exact_repeat_per_arm=True,
        sampler_rows=rows,native_recipe_identical_for_data_weight_flags=True,native_recipe=recipes['False'],
        unchanged_advance_function_SHA256=function_pin(MODEL/'released_update.py','advance'),
        unchanged_Context_class_SHA256=function_pin(MODEL/'released_adapter.py','Context'),
        production_configure_guard_still_rejects_balanced_arm=True,
        in_memory_override_only_in_disposable_CPU_fixture=True,production_sources_unchanged=True,
        no_model_or_optimizer_constructed=True,new_native_training_pairs=0,new_real_ALZ_fits=0,
        no_GPU_allocation=True,CUDA_initialized=False,future_trial_qualified=False,queued=False,activated=False,
        remaining='A real future test requires an isolated explicit application allowlist, fresh training-seed namespace, native update-prefix qualification for the new arm, matched from-zero full20k controls, resource/accounting qualification and sufficient authorized window. Do not use this fixture to admit an arm under current frozen studies.',
        interpretation='The existing data-sampler branch can make every32-bag batch exactly8bags per stage without changing the resolved public native PG recipe. This does not demonstrate convergence, an ED/ES gain, exact native update parity of a future new application wrapper, or generalization.',
        source_SHA256=source_pins,input_SHA256={str(metadata):sha(metadata),str(ROOT/'analysis/distribution097/native_release.json'):sha(ROOT/'analysis/distribution097/native_release.json')},
        full_goal_complete=False,SOTA_claim=False)
    output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');signal.alarm(0)
    print(json.dumps({k:record[k] for k in ('status','actual_synthetic_batch_calls',
        'native_recipe_identical_for_data_weight_flags','production_configure_guard_still_rejects_balanced_arm','new_real_ALZ_fits')}))


if __name__=='__main__':main()
