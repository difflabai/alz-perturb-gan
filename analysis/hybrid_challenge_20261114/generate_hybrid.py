"""One fixed semiparametric whole-bag mixture; isolated from prior experiments."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='3'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
import math
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'empirical_challenge_20261114'
sys.path.insert(0,str(PREVIOUS))
import generate_recipe as W
import audit as A
import provenance as P

np=W.np;G=W.G;T=W.T;AS=W.AS;BL=W.BL
ROOT=W.RUNS/'results_r5/hybrid_challenge_20261114'
MODES=W.RUNS/'smoke_r5/hybrid_challenge_20261114'
RECIPES=('original32','current32','empirical32','hybrid32')
ARRAYS=('pb_counts','mean_log','nonzero','pcs','lib','sq_log')
IDENTITY=('targets','is_heldout','stage','genes')
NAMESPACE='hybrid20261114; see committed PROTOCOL.txt'
sha=W.sha;write=W.write


def require(condition,message):
    if not condition:raise ValueError(message)


def source_manifest():
    paths=[HERE/name for name in ('PROTOCOL.txt','generate_hybrid.py','compare_hybrid.py','run_hybrid.py',
                                  'prior_study_complete_evidence.json')]
    paths += [Path(A.__file__)]
    # Do not silently bless changed prior dependencies with a new launch hash.
    return {**A.sources(),**{str(p):sha(p) for p in paths}}


def streams(fold,index,replicate):
    children=np.random.SeedSequence([20261114,2,fold,index,replicate]).spawn(5)
    return {k:int(child.generate_state(1,dtype=np.uint64)[0])%(2**63-16)
            for k,child in zip(G.GEN_STREAMS,children)}


def empirical_rng(real,target,replicate,role):
    require(role in ('reference','component'),'unknown empirical stream role')
    return np.random.default_rng(AS.E.MX.stable_seed('hybrid-challenge',20261114,
        real.panel,real.fold,real.donors[target],replicate,role))


def combine(model,empirical):
    for key in IDENTITY:require(np.array_equal(model[key],empirical[key]),'component identity mismatch: '+key)
    output={key:model[key].copy() for key in IDENTITY}
    for key in ARRAYS:
        left,right=model[key],empirical[key]
        require(left.dtype==right.dtype and left.shape==right.shape and left.shape[1]==16,
                'component bag shape/dtype mismatch: '+key)
        output[key]=np.concatenate((left,right),axis=1)
    output['mode']=np.array('hybrid32:16current-stagehalf+16resample-blind')
    output['bag_component']=np.array(['model']*16+['empirical']*16)
    return output


def read_arrays(path):
    with np.load(path,allow_pickle=False) as arrays:
        return {k:arrays[k].copy() for k in (*ARRAYS,*IDENTITY)}


def validate_reference(recipe,info,cache,replicate):
    if recipe=='empirical32':
        require(info['baseline']=='resample_blind' and info['empirical_role']=='reference'
                and info['empirical_stream']==replicate and info['kpc']==64 and
                0<=info['frac_cells_depth_unmatched']<=1 and
                info['source_selection']=='training donors only; target excluded when in training',
                'wrong empirical reference stream/algorithm')
        return
    require(info['mode']==W.MODES_BY_NAME[recipe] and info['states_used']==A.STATES
            and info['heldout_targets']==len(cache['heldout_donors'])
            and info['runtime']['particlegan']=='0.9.7','wrong model reference mode/states/runtime')
    extra=info['mode_extra'];bias=extra['biascal']
    require(bias['n_train_targets']==len(cache['train_donors']) and sorted(bias['states'])==sorted(A.STATES)
            and len(bias['delta'])==2000 and all(math.isfinite(x) for x in bias['delta']),
            'wrong model reference bias calibration')
    if recipe=='current32':
        z=extra['zcal']
        require(z['n_train_targets']==len(cache['train_donors']) and sorted(z['states'])==sorted(A.STATES)
                and z['joint_biascal'] and len(z['s'])==2000
                and all(math.isfinite(x) and -6.000001<=x<=6.000001 for x in z['s'])
                and extra['cellshape']['alpha']==.5,'wrong model reference joint dispersion calibration')
    A.allocation(info['state_stage_allocation'],cache,32,recipe)


def finalized(recipe,fold,replicate,sources):
    score=ROOT/recipe/f'fold{fold}/seed{replicate}.json'
    output=MODES/recipe/f'fold{fold}/seed{replicate}.npz'
    receipt=output.with_suffix('.receipt.json')
    if not score.exists():return False
    row=P.read(score);meta=row.get('challenge',{})
    require(meta.get('recipe')==recipe and meta.get('fold')==fold and meta.get('replicate')==replicate
            and meta.get('sources')==sources,'existing result identity/source differs')
    info=P.read(receipt)
    require(row['fold']==fold and row['M']==32 and meta.get('K')==256 and meta.get('PC_CELLS')==64
            and meta.get('generation_device')==info['generation_device']=='cpu'
            and info['challenge_recipe']==recipe and info['computational_replicate']==replicate
            and info['M']==32 and info['K']==256 and info['targets']==37
            and info['generation_streams']==NAMESPACE and info['sources']==sources,
            'existing receipt dimensions/streams/source differ')
    require(meta['receipt_sha256']==sha(receipt) and not output.exists()
            and row['pruned_source_sha256']==row['fingerprint']['generated_sha256']==info['generated_SHA256']
            and Path(row['source'])==output.resolve(),'existing receipt/archive/pruning mismatch')
    cache=A.cache_metadata(fold,P.verify_cache(fold,payloads=False))
    require(info['cache_manifest_sha256']==row['fingerprint']['cache_sha256']==
            cache['manifest_sha256'],'wrong frozen cache')
    require(row['fingerprint']['score_code_sha256']==sources[str(W.R5/'architecture_score.py')]
            and row['fingerprint']['interval_code_sha256']==sources[str(W.R5/'interval_metrics.py')],
            'wrong score math sources')
    if recipe!='empirical32':
        fit=W.RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{replicate}'
        require(Path(info['fit']).resolve()==fit.resolve() and
                info['fit_artifacts_sha256']==P.verify_fit(fold,replicate),'wrong frozen fit lineage')
    if recipe!='hybrid32':validate_reference(recipe,info,cache,replicate)
    for split in ('train','heldout'):
        for method in ('percentile','rank'):
            for group in ('mean','sd','combined'):
                block=row['intervals'][split][method][group]
                require(all(math.isfinite(block[k]) for k in ('coverage','below','above','width','interval_score'))
                        and all(0<=block[k]<=1 for k in ('coverage','below','above'))
                        and abs(block['coverage']+block['below']+block['above']-1)<1e-10
                        and block['width']>=0 and block['interval_score']+1e-12>=block['width'],
                        'invalid interval score metadata')
    if recipe=='hybrid32':
        calibration=MODES/'current32'/f'fold{fold}/seed{replicate}.receipt.json'
        require(info['mixture']=={'model_bags':16,'empirical_bags':16,'unit':'whole_donor_bag'}
                and info['calibration_M']==32 and info['calibration_receipt_sha256']==sha(calibration)
                and info['model_component']['mode_extra']==P.read(calibration)['mode_extra'],
                'wrong fixed mixture or frozen calibration')
        model=info['model_component']
        require(model['calibration_M']==32 and model['calibration_receipt_sha256']==sha(calibration)
                and model['generation_device']=='cpu' and Path(model['fit']).resolve()==fit.resolve(),
                'wrong model component lineage/calibration')
        A.allocation(model['state_stage_allocation'],cache,16,'current32')
        for role in ('model','empirical'):
            component=info[role+'_component']
            path=output.with_name(output.stem+'.'+role+'.npz')
            require(component['M']==16 and component['K']==256 and
                    component['pruned_source_sha256']==component['generated_SHA256'] and not path.exists(),
                    'component generation/pruning mismatch')
        empirical=info['empirical_component']
        require(empirical['empirical_role']=='component' and empirical['empirical_stream']==replicate
                and empirical['baseline']=='resample_blind' and empirical['kpc']==64 and
                0<=empirical['frac_cells_depth_unmatched']<=1 and
                empirical['source_selection']=='training donors only; target excluded when in training',
                'wrong empirical component stream/algorithm')
    return True


def empirical(real,replicate,draws,role):
    BL._target_rng=lambda r,t:empirical_rng(r,t,replicate,role)
    arrays,info=BL.generate('resample_blind',real,M=draws,K=256,kpc=64)
    info.update(baseline='resample_blind',generation_device='cpu',targets=len(real.donors),
                empirical_stream=replicate,empirical_role=role,
                source_selection='training donors only; target excluded when in training')
    return arrays,info


def run(recipe,fold,replicate):
    require(recipe in RECIPES and fold in range(9) and replicate in (0,1),'invalid fixed job')
    W.torch.set_num_threads(3)
    sources=source_manifest()
    require(P.read(ROOT/'source_manifest.json')==sources,'launch pinned queue first')
    P.verify_cache(fold,payloads=False)
    if recipe!='empirical32':P.verify_fit(fold,replicate)
    if finalized(recipe,fold,replicate,sources):return
    output=MODES/recipe/f'fold{fold}/seed{replicate}.npz'
    score=ROOT/recipe/f'fold{fold}/seed{replicate}.json'
    require(not score.exists() and not any(output.parent.glob(output.stem+'*')),
            'partial job artifacts exist; inspect before resuming')
    output.parent.mkdir(parents=True,exist_ok=True)
    cache=W.RUNS/f'caches/hvg2000/fold{fold}'
    fit=W.RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{replicate}'
    G.generation_streams=lambda f,j:streams(f,j,replicate)
    started=time.monotonic()
    if recipe=='empirical32':
        arrays,receipt=empirical(AS.E.load_real(cache),replicate,32,'reference')
        AS.E.save_npz(output,**arrays)
    elif recipe=='hybrid32':
        require(finalized('current32',fold,replicate,sources),'complete current32 calibration required')
        model_path=output.with_name(output.stem+'.model.npz')
        empirical_path=output.with_name(output.stem+'.empirical.npz')
        calibration=MODES/'current32'/f'fold{fold}/seed{replicate}.receipt.json'
        model_info=W.replay(fit,cache,calibration,model_path,16)
        model_info.update(generation_streams=NAMESPACE,
                          note='M16 valid stagehalf allocation; frozen M32 coefficient replay, not first16 M32 bags')
        model_info['state_stage_allocation']=W.allocation(model_path)
        model=read_arrays(model_path)
        other,other_info=empirical(AS.E.load_real(cache),replicate,16,'component')
        AS.E.save_npz(empirical_path,**other)
        AS.E.save_npz(output,**combine(model,other))
        for path,info in ((model_path,model_info),(empirical_path,other_info)):
            info.update(generated_SHA256=sha(path),pruned_source_sha256=sha(path))
            path.unlink()
        receipt={'M':32,'K':256,'targets':37,'generation_device':'cpu','fit':str(fit.resolve()),
                 'mixture':{'model_bags':16,'empirical_bags':16,'unit':'whole_donor_bag'},
                 'calibration_M':32,'calibration_receipt_sha256':sha(calibration),
                 'model_component':model_info,'empirical_component':other_info,
                 'scientific_role':'semiparametric whole-bag simulator; observed training cells reused'}
    else:
        G.M=32
        _,receipt=G.generate_fit(fit,cache,mode=W.MODES_BY_NAME[recipe],out=output,device='cpu')
        receipt['state_stage_allocation']=W.allocation(output)
    receipt.update(challenge_recipe=recipe,computational_replicate=replicate,
                   generation_streams=NAMESPACE,sources=sources,generated_SHA256=sha(output),
                   seconds_total=time.monotonic()-started,cache_manifest_sha256=sha(cache/'manifest.json'),
                   runtime={'python':sys.version,'numpy':np.__version__,'torch':str(W.torch.__version__),
                            'particlegan':str(T.VERSION)})
    if recipe!='empirical32':receipt['fit_artifacts_sha256']={name:sha(fit/name) for name in W.FIT_FILES}
    write(output.with_suffix('.receipt.json'),receipt)
    AS.score(output,cache,score)
    require(source_manifest()==sources,'source changed during generation/scoring')
    row=P.read(score)
    row['challenge']={'recipe':recipe,'fold':fold,'replicate':replicate,'sources':sources,
                      'receipt_sha256':sha(output.with_suffix('.receipt.json')),
                      'generation_device':'cpu','K':256,'PC_CELLS':64}
    row['pruned_source_sha256']=row['fingerprint']['generated_sha256']
    write(score,row);output.unlink()
    require(finalized(recipe,fold,replicate,sources),'worker did not finalize valid output')
    print(json.dumps({'recipe':recipe,'fold':fold,'replicate':replicate}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',choices=RECIPES,required=True)
    parser.add_argument('--fold',type=int,choices=range(9),required=True)
    parser.add_argument('--replicate',type=int,choices=(0,1),required=True)
    args=parser.parse_args();run(args.recipe,args.fold,args.replicate)
