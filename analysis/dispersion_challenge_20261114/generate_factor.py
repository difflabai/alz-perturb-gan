"""Fresh matched references and one fixed TRAIN donor-dispersion recipe."""
import argparse
import gc
import math
import sys
import time
from pathlib import Path
import common as B
import factor_model as F

np=B.np;torch=B.torch;G=B.G;ZC=B.ZC
FACTOR_MODE='factor32@cellshape0.5:biascal:stagehalf'


def interval_sanity(row):
    for split in ('train','heldout'):
        for method in ('percentile','rank'):
            for summary in ('mean','sd','combined'):
                block=row['intervals'][split][method][summary]
                B.require(all(math.isfinite(block[k]) for k in ('coverage','below','above','width','interval_score'))
                    and all(0<=block[k]<=1 for k in ('coverage','below','above'))
                    and abs(sum(block[k] for k in ('coverage','below','above'))-1)<1e-10
                    and block['width']>=0 and block['interval_score']+1e-12>=block['width'],
                    'invalid interval score metadata')


def validate_law(info,cache,fold,replicate):
    law=info['factor_law'];donors=cache['train_donors'];records=law['donor_fits']
    B.require(law['fit_donors']==donors and law['n_train_targets']==len(donors)
              and [r['donor'] for r in records]==donors,'factor fit donor scope changed')
    values=[]
    for record in records:
        u=record['u'];values.append(u)
        B.require(math.isfinite(u) and -1<=u<=1 and record['generated_bags']==32
                  and record['generated_cells']==32*256 and record['generated_gene_values']==32*256*2000,
                  'wrong factor fitting dimensions/bound')
        B.require(all(math.isfinite(record[k]) for k in ('observed_nonzero','F_low','F_high','expected_nonzero_residual'))
                  and 0<=record['observed_nonzero']<=1 and 0<=record['F_low']<=record['F_high']<=1,
                  'invalid factor fit diagnostics')
        side=record['clipped']
        B.require((side=='low' and u==-1 and record['bisection_steps']==0 and record['observed_nonzero']<=record['F_low'])
                  or (side=='high' and u==1 and record['bisection_steps']==0 and record['observed_nonzero']>=record['F_high'])
                  or (side is None and record['bisection_steps']==12 and record['F_low']<record['observed_nonzero']<record['F_high']),
                  'wrong bounded scalar fitting procedure')
    h=np.asarray(law['h'],dtype=np.float64)
    B.require(np.array_equal(h,F.normalized_factors(values)) and np.max(np.abs(h))<=2.000001
              and abs(np.exp(-h).mean()-1)<1e-12 and abs(law['mean_inverse_precision_factor']-1)<1e-12,
              'factor noise-moment normalization changed')
    B.require(law['generation_h_float32']==h.astype(np.float32).tolist()
              and abs(law['mean_inverse_precision_factor_float32']-1)<1e-6
              and law['normalization']=='h=u+log(mean(exp(-u)))'
              and law['fitting']=='12-step bounded bisection; u in[-1,1]', 'wrong factor law metadata')
    assignments=law['target_bag_indices']
    B.require(set(assignments)==set(donors+cache['heldout_donors']),'wrong factor assignment targets')
    for donor,indices in assignments.items():
        expected=np.random.default_rng(B.factor_seed(fold,donor,replicate)).integers(len(h),size=32).tolist()
        B.require(indices==expected and all(type(x) is int for x in indices),'private factor assignments changed')


def finalized(recipe,fold,replicate,sources):
    path=B.ROOT/recipe/f'fold{fold}/seed{replicate}.json'
    output=B.MODES/recipe/f'fold{fold}/seed{replicate}.npz'
    receipt=output.with_suffix('.receipt.json')
    if not path.exists():return False
    row=B.P.read(path);info=B.P.read(receipt);meta=row.get('challenge',{})
    B.require(meta.get('recipe')==recipe and meta.get('fold')==fold and meta.get('replicate')==replicate
              and row['fold']==fold and row['M']==32 and meta.get('sources')==info['sources']==sources,
              'existing result source/identity mismatch')
    B.require(meta['receipt_sha256']==B.sha(receipt) and not output.exists() and Path(row['source'])==output.resolve()
              and row['pruned_source_sha256']==row['fingerprint']['generated_sha256']==info['generated_SHA256'],
              'receipt/archive/pruning mismatch')
    B.require(info['challenge_recipe']==recipe and info['computational_replicate']==replicate
              and info['generation_streams']==B.NAMESPACE and info['M']==32 and info['K']==256
              and info['targets']==37 and meta['K']==256 and meta['PC_CELLS']==64
              and meta['generation_device']==info['generation_device']=='cpu','wrong generation dimensions/streams/device')
    B.validate_fingerprints(info['array_fingerprints'])
    cache=B.A.cache_metadata(fold,B.P.verify_cache(fold,payloads=False))
    B.require(row['fingerprint']['cache_sha256']==info['cache_manifest_sha256']==cache['manifest_sha256'],
              'wrong frozen cache')
    B.require(row['fingerprint']['score_code_sha256']==sources[str(B.W.R5/'architecture_score.py')]
              and row['fingerprint']['interval_code_sha256']==sources[str(B.W.R5/'interval_metrics.py')],
              'wrong score math source')
    if recipe!='empirical32':
        fit=B.W.RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{replicate}'
        B.require(Path(info['fit']).resolve()==fit.resolve() and info['fit_artifacts_sha256']==B.P.verify_fit(fold,replicate),
                  'wrong original fit lineage')
    if recipe=='factor32':
        B.require(info['mode']==FACTOR_MODE,'wrong factor recipe mode')
        # Validate the shared cellshape/joint-calibration contract separately
        # from the additional factor law, whose actual mode is checked above.
        B.H.validate_reference('current32',dict(info,mode=B.W.MODES_BY_NAME['current32']),cache,replicate)
        calibration=B.MODES/'current32'/f'fold{fold}/seed{replicate}.receipt.json'
        B.require(info['factor_fit_calibration_receipt_sha256']==B.sha(calibration),'factor fit replay link changed')
        validate_law(info,cache,fold,replicate)
        verify_fidelity(sources)
        B.require(info['zero_fidelity_sha256']==B.sha(B.ROOT/'zero_factor_fidelity.json'),'zero-control link changed')
    else:B.H.validate_reference(recipe,info,cache,replicate)
    interval_sanity(row)
    return True


def layout(ctx,fit,cfg,targets):
    states=G.available_states(fit,cfg)
    plans,_,_=G.mode_plans(ctx,'stagehalf',targets,states,updates=cfg['updates'])
    used=[s for s in states if any(s in names for _,names in plans)]
    B.require(used==B.A.STATES,'unexpected selected model states')
    return plans,used


def context(fit,cache):
    cfg=B.P.read(fit/'config.json');ctx=B.T.Context(cache,'cpu',tuple(cfg['flags']))
    train=ZC.train_targets(ctx);plans,states=layout(ctx,fit,cfg,train)
    return ctx,cfg,plans,states


@torch.no_grad()
def refit(ctx,fit,cfg,plans,states,route):
    models=G.load_models(ctx,fit,cfg,states)
    for model,_ in models.values():route.patch(model);B.CS.patch_generator(model,.5)
    with route.installed():calibration=ZC.calibrate(ctx,models,dict(enumerate(plans)),joint=True)
    return models,calibration


@torch.no_grad()
def generate(ctx,fit,cfg,models,route,mode):
    targets=G.targets(ctx);plans,states=layout(ctx,fit,cfg,targets)
    with route.installed():arrays,stats=G.generate_arrays(ctx,models,plans,T=targets)
    arrays.update(mode=np.array(mode),state_names=np.array(states),
                  state_index=np.array([[states.index(s) for s in names] for _,names in plans],dtype=np.int8),
                  conditions=np.stack([c for c,_ in plans]).astype(np.float32))
    return arrays,stats


def verify_fidelity(sources):
    proof=B.P.read(B.ROOT/'zero_factor_fidelity.json')
    current=B.MODES/'current32/fold0/seed0.receipt.json'
    reference=B.P.read(current)
    B.validate_fingerprints(reference['array_fingerprints'])
    B.validate_fingerprints(proof['zero_array_fingerprints'])
    B.require(B.valid_digest(proof.get('pruned_control_sha256')),'missing/invalid pruned zero-control digest')
    B.require(proof['passed'] is True and proof['sources']==sources and proof['fold']==proof['replicate']==0
              and proof['current_receipt_sha256']==B.sha(current)
              and proof['zero_array_fingerprints']==reference['array_fingerprints']
              and proof['zero_bias_delta']==reference['mode_extra']['biascal']['delta']
              and proof['zero_gene_s']==reference['mode_extra']['zcal']['s']
              and not (B.MODES/'zero_fidelity/fold0/seed0.npz').exists(),
              'full zero-factor calibration fidelity not proven')


@torch.no_grad()
def ensure_fidelity(sources,allow_create=False):
    path=B.ROOT/'zero_factor_fidelity.json'
    if path.exists():verify_fidelity(sources);return
    B.require(allow_create,'initial fold0/rep0 zero control must be completed first')
    B.require(finalized('current32',0,0,sources),'fresh first current32 reference required')
    fit=B.W.RUNS/'fits_r3/pbulk/hvg2000/fold0/seed0';cache=B.W.RUNS/'caches/hvg2000/fold0'
    output=B.MODES/'zero_fidelity/fold0/seed0.npz'
    B.require(not output.exists(),'partial zero-fidelity archive exists')
    reference=B.P.read(B.MODES/'current32/fold0/seed0.receipt.json')
    ctx,cfg,plans,states=context(fit,cache)
    route=F.FactorRoute([0.],lambda donor:B.factor_seed(0,donor,0))
    models,info=refit(ctx,fit,cfg,plans,states,route)
    arrays,stats=generate(ctx,fit,cfg,models,route,B.W.MODES_BY_NAME['current32'])
    B.AS.E.save_npz(output,**arrays)
    fingerprints=B.array_fingerprints(arrays)
    B.require(fingerprints==reference['array_fingerprints'] and
              info['biascal']['delta']==reference['mode_extra']['biascal']['delta'] and
              info['s']==reference['mode_extra']['zcal']['s'],'full zero-factor calibration differs from frozen current32')
    B.write(path,{'passed':True,'fold':0,'replicate':0,'sources':sources,
                  'current_receipt_sha256':B.sha(B.MODES/'current32/fold0/seed0.receipt.json'),
                  'zero_array_fingerprints':fingerprints,'zero_bias_delta':info['biascal']['delta'],
                  'zero_gene_s':info['s'],'pruned_control_sha256':B.sha(output),
                  'scope':'Exact common generated arrays and full joint calibration; not numerical superiority'})
    output.unlink();verify_fidelity(sources)


@torch.no_grad()
def factor(fit,cache,fold,replicate,sources):
    ensure_fidelity(sources,allow_create=(fold,replicate)==(0,0))
    gc.collect()
    reference_path=B.MODES/'current32'/f'fold{fold}/seed{replicate}.receipt.json'
    reference=B.P.read(reference_path)
    ctx,cfg,plans,states=context(fit,cache)
    baseline=G.load_models(ctx,fit,cfg,states)
    for model,_ in baseline.values():
        B.BC.head(model).add_(torch.tensor(reference['mode_extra']['biascal']['delta'],dtype=torch.float32))
        B.CS.patch_generator(model,.5)
        B.ZC.patch_generator(model,'cpu')['s'].copy_(torch.tensor(reference['mode_extra']['zcal']['s'],dtype=torch.float32))
    h,law=F.fit_law(ctx,baseline,plans)
    del baseline;gc.collect()
    route=F.FactorRoute(h,lambda donor:B.factor_seed(fold,donor,replicate))
    models,cal=refit(ctx,fit,cfg,plans,states,route)
    arrays,stats=generate(ctx,fit,cfg,models,route,FACTOR_MODE)
    cal['inner_mode']='biascal@stagehalf'
    extra={'biascal':cal['biascal'],'zcal':{k:v for k,v in cal.items() if k!='biascal'},
           'cellshape':reference['mode_extra']['cellshape']}
    law.update(target_bag_indices=route.assignments,generation_h_float32=route.values.tolist(),
               mean_inverse_precision_factor_float32=float(np.exp(-route.values.astype(np.float64)).mean()))
    stats.update(mode=FACTOR_MODE,states_used=states,mode_extra=extra,factor_law=law,
                 fit=str(fit.resolve()),generation_device='cpu',
                 factor_fit_calibration_receipt_sha256=B.sha(reference_path),
                 zero_fidelity_sha256=B.sha(B.ROOT/'zero_factor_fidelity.json'))
    return arrays,stats


@torch.no_grad()
def run(recipe,fold,replicate):
    B.require(recipe in B.RECIPES and fold in range(9) and replicate in (0,1),'invalid fixed job')
    torch.set_num_threads(3);G.M=32
    sources=B.source_manifest()
    B.require(B.P.read(B.ROOT/'source_manifest.json')==sources,'launch pinned queue first')
    B.P.verify_cache(fold,payloads=False)
    if recipe!='empirical32':B.P.verify_fit(fold,replicate)
    if finalized(recipe,fold,replicate,sources):return
    output=B.MODES/recipe/f'fold{fold}/seed{replicate}.npz'
    score=B.ROOT/recipe/f'fold{fold}/seed{replicate}.json'
    B.require(not score.exists() and not any(output.parent.glob(output.stem+'*')),'partial job artifacts exist; inspect first')
    output.parent.mkdir(parents=True,exist_ok=True)
    cache=B.W.RUNS/f'caches/hvg2000/fold{fold}';fit=B.W.RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{replicate}'
    G.generation_streams=lambda f,j:B.streams(f,j,replicate)
    started=time.monotonic()
    if recipe=='empirical32':
        real=B.AS.E.load_real(cache);B.BL._target_rng=lambda r,t:B.empirical_rng(r,t,replicate)
        arrays,info=B.BL.generate('resample_blind',real,M=32,K=256,kpc=64)
        B.AS.E.save_npz(output,**arrays)
        info.update(baseline='resample_blind',empirical_role='reference',empirical_stream=replicate,
                    source_selection='training donors only; target excluded when in training',targets=37)
    elif recipe=='factor32':
        B.require(finalized('current32',fold,replicate,sources),'complete current32 calibration required')
        arrays,info=factor(fit,cache,fold,replicate,sources)
        B.AS.E.save_npz(output,**arrays)
    else:_,info=G.generate_fit(fit,cache,mode=B.W.MODES_BY_NAME[recipe],out=output,device='cpu')
    if recipe!='empirical32':
        info.update(fit_artifacts_sha256={name:B.sha(fit/name) for name in B.W.FIT_FILES},
                    state_stage_allocation=B.W.allocation(output))
    info.update(challenge_recipe=recipe,computational_replicate=replicate,generation_streams=B.NAMESPACE,
                sources=sources,generated_SHA256=B.sha(output),generation_device='cpu',
                seconds_total=time.monotonic()-started,cache_manifest_sha256=B.sha(cache/'manifest.json'),
                array_fingerprints=B.archive_fingerprints(output),
                runtime={'python':sys.version,'numpy':np.__version__,'torch':str(torch.__version__),'particlegan':str(B.T.VERSION)})
    B.write(output.with_suffix('.receipt.json'),info)
    B.AS.score(output,cache,score)
    B.require(B.source_manifest()==sources,'source changed during generation/scoring')
    row=B.P.read(score);row['pruned_source_sha256']=row['fingerprint']['generated_sha256']
    row['challenge']={'recipe':recipe,'fold':fold,'replicate':replicate,'sources':sources,
                      'receipt_sha256':B.sha(output.with_suffix('.receipt.json')),
                      'generation_device':'cpu','K':256,'PC_CELLS':64}
    B.write(score,row);output.unlink()
    B.require(finalized(recipe,fold,replicate,sources),'worker did not finalize')
    print(B.H.json.dumps({'recipe':recipe,'fold':fold,'replicate':replicate}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',choices=B.RECIPES,required=True)
    parser.add_argument('--fold',type=int,choices=range(9),required=True)
    parser.add_argument('--replicate',type=int,choices=(0,1),required=True)
    args=parser.parse_args();run(args.recipe,args.fold,args.replicate)
