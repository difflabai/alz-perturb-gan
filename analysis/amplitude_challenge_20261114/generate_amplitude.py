"""Fresh matched references and one fixed TRAIN composition-amplitude recipe."""
import argparse
import gc
import math
import sys
import time
from pathlib import Path
import amp_common as B
import amplitude_model as F

np=B.np;torch=B.torch;G=B.G;ZC=B.ZC
AMPLITUDE_MODE='amplitude32@jointcal:stagehalf'


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


class InfeasibleLaw(ValueError):pass


def validate_raw_record(record,cache,fold,replicate,sources,current_sha,fit_hashes):
    law=record['law'];donors=cache['train_donors'];records=law['donor_fits']
    B.require(record['fold']==fold and record['replicate']==replicate and record['sources']==sources
              and record['fit_hashes']==fit_hashes and record['cache_manifest_sha256']==cache['manifest_sha256']
              and record['current_receipt_sha256']==current_sha and law['fit_donors']==donors
              and law['n_train_targets']==len(donors) and [r['donor'] for r in records]==donors,
              'wrong raw TRAIN law provenance/scope')
    for rec in records:
        B.require(rec['generated_bags']==32 and rec['generated_cells']==32*256
                  and rec['generated_gene_values']==32*256*2000,'wrong raw amplitude fitting dimensions')
        validate_fit(rec)
    values,normal=F.normalize([r['raw_t'] for r in records])
    for key,value in normal.items():B.require(law[key]==value,'wrong raw law normalization: '+key)
    B.require(law['fitting']=='concave peak/segment roots;12 derivative and12 response bisections; exact endpoint roots'
              and law['fit_precision']=='float64 evaluation of fixed float32 fully calibrated NB parameters',
              'wrong raw law fitting method/precision')
    return values


def validate_law(info,cache,fold,replicate):
    law=info['amplitude_law'];donors=cache['train_donors'];records=law['donor_fits']
    B.require(law['fit_donors']==donors and law['n_train_targets']==len(donors)
              and [r['donor'] for r in records]==donors,'wrong TRAIN amplitude fit scope')
    raw=[]
    for rec in records:
        raw.append(rec['raw_t'])
        B.require(rec['generated_bags']==32 and rec['generated_cells']==32*256
                  and rec['generated_gene_values']==32*256*2000,'wrong amplitude fitting dimensions')
        validate_fit(rec)
    values,normal=F.normalize(raw)
    B.require(values is not None,'infeasible amplitude law used for generation')
    for key,value in normal.items():B.require(law[key]==value,'wrong normalized amplitude law: '+key)
    assignments=law['target_bag_indices']
    B.require(set(assignments)==set(donors+cache['heldout_donors']),'wrong amplitude target set')
    for donor,indices in assignments.items():
        expected=np.random.default_rng(B.amplitude_seed(fold,donor,replicate)).integers(len(values),size=32).tolist()
        B.require(indices==expected and all(type(x) is int for x in indices),'private amplitude assignments changed')
    path=B.ROOT/'raw_laws'/f'fold{fold}/seed{replicate}.json';record=B.P.read(path)
    validate_raw_record(record,cache,fold,replicate,info['sources'],
                        info['amplitude_fit_calibration_receipt_sha256'],info['fit_artifacts_sha256'])
    B.require(info['raw_law_sha256']==B.sha(path) and record['fold']==fold and record['replicate']==replicate
              and record['sources']==info['sources'] and record['fit_hashes']==info['fit_artifacts_sha256']
              and record['cache_manifest_sha256']==cache['manifest_sha256']
              and record['current_receipt_sha256']==info['amplitude_fit_calibration_receipt_sha256']
              and {k:v for k,v in law.items() if k!='target_bag_indices'}==record['law'],
              'raw TRAIN law provenance differs')


def validate_fit(rec):
    keys=('raw_t','observed_nonzero','F_zero','F_one','F_peak','peak','dF_zero','dF_one',
          'peak_resolution','expected_nonzero_residual')
    B.require(all(math.isfinite(rec[k]) for k in keys) and all(0<=rec[k]<=1 for k in
              ('raw_t','observed_nonzero','F_zero','F_one','F_peak','peak')),'invalid amplitude fit values')
    d0,d1=rec['dF_zero'],rec['dF_one'];peak=rec['peak'];lo,hi=rec['peak_bracket']
    if d0<=0:expected=(peak==lo==hi==0 and rec['peak_bisection_steps']==0)
    elif d1>=0:expected=(peak==lo==hi==1 and rec['peak_bisection_steps']==0)
    else:expected=(0<=lo<hi<=1 and hi-lo==2**-12 and peak==(lo+hi)/2 and rec['peak_bisection_steps']==12)
    B.require(expected and rec['peak_resolution']==hi-lo and d0+1e-12>=d1,'wrong concave peak procedure')
    fs={0.:rec['F_zero'],peak:rec['F_peak'],1.:rec['F_one']};target=rec['observed_nonzero']
    degenerate=max(fs.values())-min(fs.values())<=1e-12
    B.require(rec['degenerate'] is degenerate,'wrong degeneracy flag')
    roots=rec['roots'];reachable=[]
    if not degenerate:
        for left,right in ((0.,peak),(peak,1.)):
            if left!=right and min(fs[left],fs[right])<=target<=max(fs[left],fs[right]):reachable.append([left,right])
    B.require([root['segment'] for root in roots]==reachable,'missing or spurious reachable segment')
    for root in roots:
        left,right=root['segment'];a,b=root['bracket'];t=root['t']
        if target==fs[left]:valid=(t==a==b==left and root['bisection_steps']==0)
        elif target==fs[right]:valid=(t==a==b==right and root['bisection_steps']==0)
        else:valid=(left<=a<b<=right and t==(a+b)/2 and root['bisection_steps']==12
                    and math.isclose(b-a,(right-left)/4096,rel_tol=0,abs_tol=1e-16))
        B.require(valid and root['resolution']==b-a,'wrong segment root/bracket or exact endpoint')
    if degenerate:chosen=.5;clipped=False
    elif roots:chosen=min((r['t'] for r in roots),key=lambda t:(abs(t-.5),t));clipped=False
    else:chosen=min((0.,peak,1.),key=lambda t:(abs(fs[t]-target),abs(t-.5),t));clipped=True
    B.require(rec['raw_t']==chosen and rec['clipped'] is clipped,'wrong closest root/extremum selection')


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
    affinity=info['cpu_affinity']
    B.require(isinstance(affinity,list) and 1<=len(affinity)<=3 and len(set(affinity))==len(affinity)
              and all(type(cpu) is int and cpu>=0 for cpu in affinity),'wrong CPU affinity bound')
    if recipe=='amplitude32':
        B.require(info['mode']==AMPLITUDE_MODE and info['states_used']==B.A.STATES
                  and info['heldout_targets']==len(cache['heldout_donors'])
                  and info['runtime']['particlegan']=='0.9.7','wrong amplitude mode/states/runtime')
        extra=info['mode_extra'];bias=extra['biascal'];z=extra['zcal']
        B.require(set(extra)=={'biascal','zcal','amplitude'} and
                  extra['amplitude']=={'kind':'independent TRAIN empirical law','constant_fidelity_control':False}
                  and bias['n_train_targets']==z['n_train_targets']==len(cache['train_donors'])
                  and sorted(bias['states'])==sorted(z['states'])==sorted(B.A.STATES)
                  and len(bias['delta'])==len(z['s'])==2000 and z['joint_biascal']
                  and all(math.isfinite(x) for x in bias['delta'])
                  and all(math.isfinite(x) and -6.000001<=x<=6.000001 for x in z['s']),
                  'wrong full pooled joint amplitude calibration')
        B.A.allocation(info['state_stage_allocation'],cache,32,'current32')
        calibration=B.MODES/'current32'/f'fold{fold}/seed{replicate}.receipt.json'
        B.require(info['amplitude_fit_calibration_receipt_sha256']==B.sha(calibration),'amplitude fit replay link changed')
        validate_law(info,cache,fold,replicate)
        verify_fidelity(sources)
        B.require(info['constant_fidelity_sha256']==B.sha(B.ROOT/'constant_amplitude_fidelity.json'),'constant control link changed')
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
    for model,_ in models.values():route.patch(model)
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
    proof=B.P.read(B.ROOT/'constant_amplitude_fidelity.json')
    current=B.MODES/'current32/fold0/seed0.receipt.json'
    reference=B.P.read(current)
    B.validate_fingerprints(reference['array_fingerprints'])
    B.validate_fingerprints(proof['control_array_fingerprints'])
    B.require(B.valid_digest(proof.get('pruned_control_sha256')),'missing/invalid pruned constant-control digest')
    B.require(proof['passed'] is True and proof['sources']==sources and proof['fold']==proof['replicate']==0
              and proof['current_receipt_sha256']==B.sha(current)
              and proof['control_array_fingerprints']==reference['array_fingerprints']
              and proof['control_bias_delta']==reference['mode_extra']['biascal']['delta']
              and proof['control_gene_s']==reference['mode_extra']['zcal']['s']
              and not (B.MODES/'constant_fidelity/fold0/seed0.npz').exists(),
              'full constant-amplitude calibration fidelity not proven')


@torch.no_grad()
def ensure_fidelity(sources,allow_create=False):
    path=B.ROOT/'constant_amplitude_fidelity.json'
    if path.exists():verify_fidelity(sources);return
    B.require(allow_create,'initial fold0/rep0 constant control must be completed first')
    B.require(finalized('current32',0,0,sources),'fresh first current32 reference required')
    fit=B.W.RUNS/'fits_r3/pbulk/hvg2000/fold0/seed0';cache=B.W.RUNS/'caches/hvg2000/fold0'
    output=B.MODES/'constant_fidelity/fold0/seed0.npz'
    B.require(not output.exists(),'partial zero-fidelity archive exists')
    reference=B.P.read(B.MODES/'current32/fold0/seed0.receipt.json')
    ctx,cfg,plans,states=context(fit,cache)
    route=F.AmplitudeRoute([.5],lambda donor:B.amplitude_seed(0,donor,0))
    models,info=refit(ctx,fit,cfg,plans,states,route)
    arrays,stats=generate(ctx,fit,cfg,models,route,B.W.MODES_BY_NAME['current32'])
    B.AS.E.save_npz(output,**arrays)
    fingerprints=B.array_fingerprints(arrays)
    B.require(fingerprints==reference['array_fingerprints'] and
              info['biascal']['delta']==reference['mode_extra']['biascal']['delta'] and
              info['s']==reference['mode_extra']['zcal']['s'],'full constant-amplitude calibration differs from frozen current32')
    B.write(path,{'passed':True,'fold':0,'replicate':0,'sources':sources,
                  'current_receipt_sha256':B.sha(B.MODES/'current32/fold0/seed0.receipt.json'),
                  'control_array_fingerprints':fingerprints,'control_bias_delta':info['biascal']['delta'],
                  'control_gene_s':info['s'],'pruned_control_sha256':B.sha(output),
                  'scope':'Exact common generated arrays and full joint calibration; not numerical superiority'})
    output.unlink();verify_fidelity(sources)


@torch.no_grad()
def amplitude(fit,cache,fold,replicate,sources):
    ensure_fidelity(sources,allow_create=(fold,replicate)==(0,0))
    gc.collect()
    reference_path=B.MODES/'current32'/f'fold{fold}/seed{replicate}.receipt.json'
    reference=B.P.read(reference_path)
    ctx,cfg,plans,states=context(fit,cache)
    baseline=G.load_models(ctx,fit,cfg,states)
    for model,_ in baseline.values():
        B.BC.head(model).add_(torch.tensor(reference['mode_extra']['biascal']['delta'],dtype=torch.float32))
        B.ZC.patch_generator(model,'cpu')['s'].copy_(torch.tensor(reference['mode_extra']['zcal']['s'],dtype=torch.float32))
    h,law=F.fit_law(ctx,baseline,plans)
    raw_path=B.ROOT/'raw_laws'/f'fold{fold}/seed{replicate}.json'
    B.write(raw_path,{'fold':fold,'replicate':replicate,'sources':sources,'law':law,
                      'current_receipt_sha256':B.sha(reference_path),'fit_hashes':B.P.verify_fit(fold,replicate),
                      'cache_manifest_sha256':B.sha(cache/'manifest.json')})
    if h is None:raise InfeasibleLaw('TRAIN amplitude normalization infeasible; see raw-law record')
    del baseline;gc.collect()
    route=F.AmplitudeRoute(h,lambda donor:B.amplitude_seed(fold,donor,replicate))
    models,cal=refit(ctx,fit,cfg,plans,states,route)
    arrays,stats=generate(ctx,fit,cfg,models,route,AMPLITUDE_MODE)
    cal['inner_mode']='biascal@stagehalf'
    extra={'biascal':cal['biascal'],'zcal':{k:v for k,v in cal.items() if k!='biascal'},
           'amplitude':{'kind':'independent TRAIN empirical law','constant_fidelity_control':False}}
    law.update(target_bag_indices=route.assignments)
    B.require(law['generation_t_float32']==route.values.tolist(),'generation amplitude cast differs')
    stats.update(mode=AMPLITUDE_MODE,states_used=states,mode_extra=extra,amplitude_law=law,raw_law_sha256=B.sha(raw_path),
                 fit=str(fit.resolve()),generation_device='cpu',
                 amplitude_fit_calibration_receipt_sha256=B.sha(reference_path),
                 constant_fidelity_sha256=B.sha(B.ROOT/'constant_amplitude_fidelity.json'))
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
    elif recipe=='amplitude32':
        B.require(finalized('current32',fold,replicate,sources),'complete current32 calibration required')
        arrays,info=amplitude(fit,cache,fold,replicate,sources)
        B.AS.E.save_npz(output,**arrays)
    else:_,info=G.generate_fit(fit,cache,mode=B.W.MODES_BY_NAME[recipe],out=output,device='cpu')
    if recipe!='empirical32':
        info.update(fit_artifacts_sha256={name:B.sha(fit/name) for name in B.W.FIT_FILES},
                    state_stage_allocation=B.W.allocation(output))
    info.update(challenge_recipe=recipe,computational_replicate=replicate,generation_streams=B.NAMESPACE,
                sources=sources,generated_SHA256=B.sha(output),generation_device='cpu',
                seconds_total=time.monotonic()-started,cache_manifest_sha256=B.sha(cache/'manifest.json'),
                array_fingerprints=B.archive_fingerprints(output),cpu_affinity=sorted(__import__('os').sched_getaffinity(0)),
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
    args=parser.parse_args()
    try:run(args.recipe,args.fold,args.replicate)
    except InfeasibleLaw as exc:print(str(exc),flush=True);sys.exit(2)
