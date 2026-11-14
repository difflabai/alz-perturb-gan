"""Fresh matched references and one fixed same-donor paired-cell hybrid."""
import argparse
import json
from pathlib import Path
import sys
import time
import pair_common as B
import pair_cells as K
import generate_factor as D
np=B.np


def validate_component(info,cache,replicate,fold):
    B.require(info['baseline']=='paired_resample_blind' and info['parts']==2
              and info['empirical_role']=='component' and info['empirical_stream']==replicate
              and info['M']==16 and info['K']==256 and info['kpc']==64 and info['targets']==37
              and 0<=info['frac_cells_depth_unmatched']<=1
              and info['source_selection']=='training donors only; target excluded when in training'
              and info['count_rule']=='sum_j Binomial(X_j,min(1,L/L_j)/parts)',
              'wrong fixed pair kernel/component stream')
    assignments=info['target_bag_source_donors'];train=cache['train_donors']
    B.require(set(assignments)==set(train+cache['heldout_donors']),'wrong target source assignment keys')
    for target,donors in assignments.items():
        eligible=set(train)-{target}
        B.require(len(donors)==16 and len(set(donors))==16 and set(donors)<=eligible,
                  'wrong source donor cycle or held-out/self source')
        B.require(donors==B.component_sources(train,target,cache['donor_cell_counts'][target],fold,replicate),
                  'source cycle differs from exact fixed seed/role/replicate')


def finalized(recipe,fold,replicate,sources):
    score=B.ROOT/recipe/f'fold{fold}/seed{replicate}.json'
    output=B.MODES/recipe/f'fold{fold}/seed{replicate}.npz';receipt=output.with_suffix('.receipt.json')
    if not score.exists():return False
    row=B.P.read(score);info=B.P.read(receipt);meta=row.get('challenge',{})
    B.require(meta.get('recipe')==recipe and meta.get('fold')==fold and meta.get('replicate')==replicate
              and meta.get('sources')==info['sources']==sources and row['fold']==fold and row['M']==32
              and info['challenge_recipe']==recipe and info['computational_replicate']==replicate
              and info['M']==32 and info['K']==256 and info['targets']==37 and meta['K']==256 and meta['PC_CELLS']==64
              and meta['generation_device']==info['generation_device']=='cpu' and info['generation_streams']==B.NAMESPACE,
              'wrong artifact identity/source/dimensions/streams/device')
    B.require(meta['receipt_sha256']==B.sha(receipt) and not output.exists()
              and row['pruned_source_sha256']==row['fingerprint']['generated_sha256']==info['generated_SHA256']
              and B.valid_digest(info['generated_SHA256']) and Path(row['source'])==output.resolve(),
              'wrong generated/pruned source or receipt hash')
    B.validate_fingerprints(info['array_fingerprints'])
    affinity=info['cpu_affinity']
    B.require(isinstance(affinity,list) and 1<=len(affinity)<=3 and len(set(affinity))==len(affinity)
              and all(isinstance(cpu,int) and cpu>=0 for cpu in affinity),'wrong CPU affinity bound')
    cache=B.A.cache_metadata(fold,B.P.verify_cache(fold,payloads=False))
    manifest=B.P.read(B.W.RUNS/f'caches/hvg2000/fold{fold}/manifest.json')
    cache['donor_cell_counts']={d:row['cells_kept'] for split in ('train','heldout')
                               for d,row in manifest['cells_per_donor'][split].items()}
    B.require(row['fingerprint']['cache_sha256']==info['cache_manifest_sha256']==cache['manifest_sha256']
              and row['fingerprint']['score_code_sha256']==sources[str(B.W.R5/'architecture_score.py')]
              and row['fingerprint']['interval_code_sha256']==sources[str(B.W.R5/'interval_metrics.py')],
              'wrong frozen cache/score/interval math')
    if recipe!='empirical32':
        fit=B.W.RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{replicate}'
        B.require(Path(info['fit']).resolve()==fit.resolve() and info['fit_artifacts_sha256']==B.P.verify_fit(fold,replicate),
                  'wrong original fit lineage')
    if recipe!='paired32':B.H.validate_reference(recipe,info,cache,replicate)
    else:
        calibration=B.MODES/'current32'/f'fold{fold}/seed{replicate}.receipt.json'
        reference=B.P.read(calibration);model=info['model_component']
        B.require(info['mixture']=={'model_bags':16,'empirical_bags':16,'unit':'whole_donor_bag'}
                  and info['calibration_M']==model['calibration_M']==32
                  and info['calibration_receipt_sha256']==model['calibration_receipt_sha256']==B.sha(calibration)
                  and model['mode_extra']==reference['mode_extra'] and model['mode']==B.W.MODES_BY_NAME['current32']
                  and model['generation_device']=='cpu' and Path(model['fit']).resolve()==fit.resolve(),
                  'wrong fixed mixture/model calibration lineage')
        B.A.allocation(model['state_stage_allocation'],cache,16,'current32')
        for role in ('model','empirical'):
            component=info[role+'_component'];path=output.with_name(output.stem+'.'+role+'.npz')
            B.require(component['M']==16 and component['K']==256 and not path.exists()
                      and component['pruned_source_sha256']==component['generated_SHA256']
                      and B.valid_digest(component['generated_SHA256']),'wrong component/pruning hash')
            B.validate_fingerprints(component['array_fingerprints'])
        validate_component(info['empirical_component'],cache,replicate,fold)
        verify_identity(sources)
        B.require(info['identity_control_sha256']==B.sha(B.ROOT/'single_cell_identity.json'),'identity proof link changed')
    D.interval_sanity(row)
    return True


def verify_identity(sources):
    proof=B.P.read(B.ROOT/'single_cell_identity.json')
    receipt=B.MODES/'empirical32/fold0/seed0.receipt.json';reference=B.P.read(receipt)
    B.validate_fingerprints(proof['control_array_fingerprints']);B.validate_fingerprints(reference['array_fingerprints'])
    B.require(proof['passed'] is True and proof['parts']==1 and proof['fold']==proof['replicate']==0
              and proof['sources']==sources and proof['reference_receipt_sha256']==B.sha(receipt)
              and proof['control_array_fingerprints']==reference['array_fingerprints']
              and B.valid_digest(proof.get('pruned_control_sha256'))
              and not (B.MODES/'identity_control/fold0/seed0.npz').exists(),
              'full single-cell frozen empirical identity not proven')


def ensure_identity(sources,allow_create=False):
    proof=B.ROOT/'single_cell_identity.json'
    if proof.exists():verify_identity(sources);return
    B.require(allow_create,'first fixed f0/r0 identity control required')
    B.require(finalized('empirical32',0,0,sources),'fresh first empirical reference required')
    output=B.MODES/'identity_control/fold0/seed0.npz'
    B.require(not output.exists(),'partial identity-control archive exists')
    cache=B.W.RUNS/'caches/hvg2000/fold0'
    arrays,_=K.generate(B.AS.E.load_real(cache),0,32,'reference',parts=1)
    B.AS.E.save_npz(output,**arrays);fingerprints=B.array_fingerprints(arrays)
    receipt=B.MODES/'empirical32/fold0/seed0.receipt.json'
    B.require(fingerprints==B.P.read(receipt)['array_fingerprints'],'new parts=1 kernel differs from frozen resampler')
    B.write(proof,{'passed':True,'parts':1,'fold':0,'replicate':0,'sources':sources,
                  'reference_receipt_sha256':B.sha(receipt),'control_array_fingerprints':fingerprints,
                  'pruned_control_sha256':B.sha(output),'scope':'Exact full frozen empirical arrays, not numerical superiority'})
    output.unlink();verify_identity(sources)


def empirical(real,replicate,draws,role):
    B.BL._target_rng=lambda r,t:B.empirical_rng(r,t,replicate,role)
    arrays,info=B.BL.generate('resample_blind',real,M=draws,K=256,kpc=64)
    info.update(baseline='resample_blind',generation_device='cpu',targets=len(real.donors),
                empirical_stream=replicate,empirical_role=role,
                source_selection='training donors only; target excluded when in training')
    return arrays,info


def run(recipe,fold,replicate):
    B.require(recipe in B.RECIPES and fold in range(9) and replicate in (0,1),'invalid fixed job')
    B.torch.set_num_threads(3);B.G.M=32;sources=B.source_manifest()
    B.require(B.P.read(B.ROOT/'source_manifest.json')==sources,'launch pinned queue first')
    B.P.verify_cache(fold,payloads=False)
    if recipe!='empirical32':B.P.verify_fit(fold,replicate)
    if finalized(recipe,fold,replicate,sources):return
    output=B.MODES/recipe/f'fold{fold}/seed{replicate}.npz';score=B.ROOT/recipe/f'fold{fold}/seed{replicate}.json'
    B.require(not score.exists() and not any(output.parent.glob(output.stem+'*')),'partial job artifacts; inspect first')
    output.parent.mkdir(parents=True,exist_ok=True)
    cache=B.W.RUNS/f'caches/hvg2000/fold{fold}';fit=B.W.RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{replicate}'
    B.G.generation_streams=lambda f,j:B.streams(f,j,replicate)
    started=time.monotonic()
    if recipe=='empirical32':
        arrays,info=empirical(B.AS.E.load_real(cache),replicate,32,'reference');B.AS.E.save_npz(output,**arrays)
    elif recipe=='paired32':
        ensure_identity(sources,allow_create=(fold,replicate)==(0,0))
        B.require(finalized('current32',fold,replicate,sources),'complete current32 calibration required')
        calibration=B.MODES/'current32'/f'fold{fold}/seed{replicate}.receipt.json'
        model_path=output.with_name(output.stem+'.model.npz');empirical_path=output.with_name(output.stem+'.empirical.npz')
        model_info=B.W.replay(fit,cache,calibration,model_path,16)
        model_info.update(generation_streams=B.NAMESPACE,state_stage_allocation=B.W.allocation(model_path),
                          note='Valid M16 stagehalf allocation; rounded M32 calibration replay, not first16 reference bags')
        model=B.H.read_arrays(model_path)
        other,other_info=K.generate(B.AS.E.load_real(cache),replicate,16,'component',parts=2)
        B.AS.E.save_npz(empirical_path,**other)
        mixed=B.H.combine(model,other);mixed['mode']=np.array('paired32:16current-stagehalf+16paired-resample-blind')
        mixed['bag_component']=np.array(['model']*16+['paired_empirical']*16)
        B.AS.E.save_npz(output,**mixed)
        for path,component in ((model_path,model_info),(empirical_path,other_info)):
            component.update(generated_SHA256=B.sha(path),pruned_source_sha256=B.sha(path),
                             array_fingerprints=B.archive_fingerprints(path));path.unlink()
        info={'M':32,'K':256,'targets':37,'generation_device':'cpu','fit':str(fit.resolve()),
              'mixture':{'model_bags':16,'empirical_bags':16,'unit':'whole_donor_bag'},
              'calibration_M':32,'calibration_receipt_sha256':B.sha(calibration),
              'model_component':model_info,'empirical_component':other_info,
              'identity_control_sha256':B.sha(B.ROOT/'single_cell_identity.json'),
              'scientific_role':'semiparametric whole-bag simulator; empirical branch generates artificial pooled cells'}
    else:
        _,info=B.G.generate_fit(fit,cache,mode=B.W.MODES_BY_NAME[recipe],out=output,device='cpu')
        info['state_stage_allocation']=B.W.allocation(output)
    if recipe!='empirical32':info['fit_artifacts_sha256']={name:B.sha(fit/name) for name in B.W.FIT_FILES}
    info.update(challenge_recipe=recipe,computational_replicate=replicate,generation_streams=B.NAMESPACE,
                sources=sources,generated_SHA256=B.sha(output),generation_device='cpu',
                seconds_total=time.monotonic()-started,cache_manifest_sha256=B.sha(cache/'manifest.json'),
                array_fingerprints=B.archive_fingerprints(output),cpu_affinity=sorted(__import__('os').sched_getaffinity(0)),
                runtime={'python':sys.version,'numpy':np.__version__,'torch':str(B.torch.__version__),'particlegan':str(B.T.VERSION)})
    B.write(output.with_suffix('.receipt.json'),info);B.AS.score(output,cache,score)
    B.require(B.source_manifest()==sources,'source changed during generation/scoring')
    row=B.P.read(score);row['pruned_source_sha256']=row['fingerprint']['generated_sha256']
    row['challenge']={'recipe':recipe,'fold':fold,'replicate':replicate,'sources':sources,
                      'receipt_sha256':B.sha(output.with_suffix('.receipt.json')),
                      'generation_device':'cpu','K':256,'PC_CELLS':64}
    B.write(score,row);output.unlink()
    B.require(finalized(recipe,fold,replicate,sources),'worker did not finalize')
    print(json.dumps({'recipe':recipe,'fold':fold,'replicate':replicate}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',choices=B.RECIPES,required=True)
    parser.add_argument('--fold',type=int,choices=range(9),required=True)
    parser.add_argument('--replicate',type=int,choices=(0,1),required=True)
    args=parser.parse_args();run(args.recipe,args.fold,args.replicate)
