"""Read-only provenance audit; no count/weight deserialization or promotion."""
import argparse
import json
import math
from pathlib import Path

import generate_recipe as W
import compare as C
import provenance as P

PRELAUNCH_SHA256='1f31f30fd94d8c30bed04afd2635d2de0638643302d0aee3085e48a4b418790d'
STATES=[f'snap_{s}' for s in range(15000,20000,1000)]+['final']
PRIMARY=('original32','current32','blind32','radial32','empirical32')
CONTROLS=('replay32','current128','empirical128')


def require(condition,message):
    if not condition:raise ValueError(message)


def sources():
    path=W.HERE/'source_manifest_prelaunch.json'
    require(W.sha(path)==PRELAUNCH_SHA256,'committed prelaunch manifest changed')
    expected=P.read(path)
    require(P.read(W.ROOT/'source_manifest.json')==expected,'runtime source manifest differs from prelaunch')
    require(W.source_manifest()==expected,'source dependency changed')
    return expected


def expected_allocation(draws,stage,train_stages,blind=False):
    """Reconstruct the frozen six-state stage plan using donor metadata only."""
    sizes=[draws//6+(i<draws%6) for i in range(6)]
    mixed=draws if blind else draws//2
    # Match the declared largest-remainder rule, including lower-stage ties.
    proportions=W.np.array([train_stages.count(s) for s in range(4)],dtype=W.np.float64)/len(train_stages)
    quota=mixed*proportions
    counts=W.np.floor(quota).astype(int)
    order=sorted(range(4),key=lambda s:(-(quota[s]-counts[s]),s))
    for s in order[:mixed-int(counts.sum())]:counts[s]+=1
    assigned=[s for s in range(4) for _ in range(counts[s])]
    # Interleave mixed positions across contiguous snapshot runs, with every
    # second position eligible for stagehalf. Fill remaining odd-run slots.
    step=1 if blind else 2
    eligible=[list(range((n-1)%step,n,step)) for n in sizes]
    positions=[(i,e[j]) for j in range(max(map(len,eligible)))
               for i,e in enumerate(eligible) if j<len(e)]
    if len(positions)<mixed:
        remaining=[[j for j in range(n) if j not in eligible[i]] for i,n in enumerate(sizes)]
        positions += [(i,e[j]) for j in range(max(map(len,remaining)))
                      for i,e in enumerate(remaining) if j<len(e)]
    stages=[[stage]*n for n in sizes]
    for (i,j),s in zip(positions[:mixed],assigned):stages[i][j]=s
    return [[row.count(s) for s in range(4)] for row in stages]


def cache_metadata(fold,cache):
    manifest=P.read(W.RUNS/f'caches/hvg2000/fold{fold}/manifest.json')
    stages={}
    for group in ('train','heldout'):
        rows=manifest['cells_per_donor'][group]
        require(set(rows)==set(cache[group+'_donors']),'stage metadata donor identities differ')
        for donor,info in rows.items():
            require(type(info['stage']) is int and info['stage'] in range(4),'invalid donor stage metadata')
            stages[donor]=info['stage']
    return dict(cache,donor_stages=stages)


def allocation(info,cache,draws,recipe):
    names=info['states'];targets=info['targets'];held=info['is_heldout']
    counts=info['target_state_stage_counts']
    require(names==STATES,'wrong allocation states')
    require(targets==cache['train_donors']+cache['heldout_donors'],
            'allocation donor identities differ from frozen37')
    require(len(held)==37 and all(type(x) is bool for x in held) and
            held==[d in cache['heldout_donors'] for d in targets],
            'allocation held-out flags differ from frozen partition')
    require(len(counts)==37,'wrong allocation target count')
    train_stages=[cache['donor_stages'][d] for d in cache['train_donors']]
    for donor,matrix in zip(targets,counts):
        require(len(matrix)==6 and all(len(row)==4 for row in matrix),'wrong state/stage count shape')
        require(all(type(x) is int and x>=0 for row in matrix for x in row),'invalid allocation count')
        require(sum(sum(row) for row in matrix)==draws,'allocation does not sum toM')
        require(matrix==expected_allocation(draws,cache['donor_stages'][donor],train_stages,
                                           blind=recipe=='blind32'),
                'allocation differs from fixed snapshot weights/stage plan')


def validate(recipe,fold,replicate,row,receipt,cache):
    """Validate recorded scientific identities; file digests checked separately."""
    draws=128 if recipe.endswith('128') else 32
    meta=row['challenge'];fp=row['fingerprint']
    require(row['fold']==fold and row['M']==draws,'wrong score fold/M')
    require(meta['recipe']==recipe and meta['fold']==fold and meta['replicate']==replicate,
            'wrong computational replicate identity')
    require(meta['K']==256 and meta['PC_CELLS']==64 and meta['generation_device']=='cpu',
            'wrong score generation dimensions/device')
    require(receipt['challenge_recipe']==recipe and receipt['computational_replicate']==replicate
            and receipt['M']==draws and receipt['K']==256 and receipt['targets']==37
            and receipt['generation_device']=='cpu','wrong receipt identity/dimensions/device')
    require(receipt['cache_manifest_sha256']==cache['manifest_sha256']==fp['cache_sha256'],
            'wrong cache fingerprint')
    require(fp['generated_sha256']==row['pruned_source_sha256']==receipt['generated_SHA256'],
            'generated/receipt/pruning hashes differ')
    require(receipt['generation_streams']=='challenge20261114; see committed PROTOCOL.txt',
            'wrong challenge stream namespace')
    for group in ('train','heldout'):
        for construction in ('percentile','rank'):
            for summary in ('mean','sd','combined'):
                block=row['intervals'][group][construction][summary]
                require(all(math.isfinite(block[s]) for s in (
                    'coverage','below','above','width','interval_score')),'nonfinite interval metric')
                require(all(0<=block[k]<=1 for k in ('coverage','below','above')) and
                        block['width']>=0 and block['interval_score']+1e-12>=block['width'],
                        'invalid interval metric')
                require(abs(block['coverage']+block['below']+block['above']-1)<1e-10,
                        'interval miss partition does not sum to1')
    if recipe.startswith('empirical'):
        require(receipt['baseline']=='resample_blind' and receipt['kpc']==64 and
                receipt['empirical_stream']==replicate,'wrong empirical recipe/stream')
        require(0<=receipt['frac_cells_depth_unmatched']<=1,'invalid depth mismatch fraction')
        require(receipt['source_selection']=='training donors only; target excluded when in training',
                'wrong empirical source-pool definition')
        return
    require(receipt['heldout_targets']==len(cache['heldout_donors']) and receipt['states_used']==STATES,
            'wrong model target/state counts')
    require(receipt['runtime']['particlegan']=='0.9.7','wrong ParticleGAN version')
    require(set(receipt['fit_artifacts_sha256'])==set(W.FIT_FILES),'incomplete fit lineage')
    mode=W.MODES_BY_NAME.get(recipe,W.MODES_BY_NAME['current32'])
    require(receipt['mode']==mode,'wrong generation mode')
    extra=receipt['mode_extra'];bias=extra['biascal']
    require(bias['n_train_targets']==len(cache['train_donors']) and
            sorted(bias['states'])==sorted(STATES),'bias calibration uses wrong donors/states')
    require(len(bias['delta'])==2000 and all(math.isfinite(x) for x in bias['delta']),
            'invalid bias calibration coefficients')
    if recipe!='original32':
        z=extra['zcal']
        require(z['n_train_targets']==len(cache['train_donors']) and
                sorted(z['states'])==sorted(STATES) and z['joint_biascal'],
                'dispersion calibration uses wrong donors/states/order')
        require(len(z['s'])==2000 and all(math.isfinite(x) and -6.000001<=x<=6.000001 for x in z['s']),
                'invalid dispersion calibration coefficients')
        if recipe=='radial32':
            law=extra['radialshape']
            require(set(law)==set(STATES),'radial law missing a selected model state')
            for state in law.values():
                digest=state['sorted_radii_sha256']
                require(state['law']=='training-prior centered radial midrank; dependent on donor latent'
                        and isinstance(digest,str) and len(digest)==64 and
                        all(c in '0123456789abcdef' for c in digest),
                        'wrong radial law label/digest format')
                require(state['particles']==512 and type(state['unique_radii']) is int and
                        1<=state['unique_radii']<=512 and
                        all(math.isfinite(state[k]) for k in ('alpha_min','alpha_max','alpha_mean')) and
                        0<state['alpha_min']<=state['alpha_mean']<=state['alpha_max']<1 and
                        abs(state['alpha_mean']-.5)<1e-7,'wrong radial contraction law')
                if state['unique_radii']==512:
                    require(state['alpha_min']==.5/512 and state['alpha_max']==511.5/512,
                            'untied radial extrema differ from512-particle midranks')
                if state['unique_radii']==1:
                    require(state['alpha_min']==state['alpha_max']==.5,
                            'fully tied radii must all contract byone half')
        else:
            require(extra['cellshape']['alpha']==.5,'wrong constant contraction strength')
    allocation(receipt['state_stage_allocation'],cache,draws,recipe)


def one(recipe,fold,replicate,expected_sources,cache,fit_hashes):
    score_path=W.ROOT/recipe/f'fold{fold}/seed{replicate}.json'
    output=W.MODES/recipe/f'fold{fold}/seed{replicate}.npz'
    receipt_path=output.with_suffix('.receipt.json')
    if not score_path.exists():return None
    row=P.read(score_path)
    # A live scorer writes before the worker adds provenance and prunes.
    if not row.get('pruned_source_sha256') or not row.get('challenge') or output.exists():return None
    require(receipt_path.exists(),'finalized score lacks receipt')
    receipt=P.read(receipt_path)
    require(row['challenge']['sources']==receipt['sources']==expected_sources,'mixed generation sources')
    require(row['challenge']['receipt_sha256']==W.sha(receipt_path),'receipt changed after scoring')
    require(Path(row['source'])==output.resolve(),'score source path differs from planned artifact')
    require(row['fingerprint']['score_code_sha256']==expected_sources[str(W.R5/'architecture_score.py')]
            and row['fingerprint']['interval_code_sha256']==expected_sources[str(W.R5/'interval_metrics.py')],
            'score math source fingerprint differs')
    validate(recipe,fold,replicate,row,receipt,cache)
    if not recipe.startswith('empirical'):
        fit=W.RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{replicate}'
        require(Path(receipt['fit']).resolve()==fit.resolve(),'receipt points to another fit')
        require(receipt['fit_artifacts_sha256']==fit_hashes[fold,replicate],'frozen fit hash mismatch')
        if recipe in ('replay32','current128'):
            current=W.MODES/'current32'/f'fold{fold}/seed{replicate}.receipt.json'
            require(receipt['calibration_M']==32 and receipt['calibration_receipt_sha256']==W.sha(current),
                    'wrong M32 calibration replay link')
            require(receipt['mode_extra']==P.read(current)['mode_extra'],'frozen replay coefficients changed')
    return {'recipe':recipe,'fold':fold,'replicate':replicate,'score_sha256':W.sha(score_path),
            'receipt_sha256':W.sha(receipt_path),'pruned_generated_sha256':row['pruned_source_sha256']}


def primary_eligibility():
    # C.report also reads controls, whose scorer may still be writing.
    # Only finalized PRIMARY rows determine the conditional experiment set.
    refs={name:C.load(name,(0,1,2))[0] for name in ('current32','empirical32','original32')}
    arms={name:C.verdict(C.load(name,(0,1,2))[0],refs['current32'],refs['empirical32'],
                         refs['original32'],(0,1,2)) for name in C.CANDIDATES}
    return arms


def mode_artifact_set(required,screen_only=False):
    expected={W.MODES/name/f'fold{f}/seed{r}.receipt.json' for name,f,r in required}
    observed=set()
    for path in W.MODES.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(W.MODES)
        if screen_only and len(relative.parts)>=2 and relative.parts[1] in {
                f'fold{f}' for f in range(3,9)}:
            continue
        observed.add(path)
    require(observed==expected,'mode artifact set contains missing/unplanned receipts or orphan archives')


def audit_path(out):
    stem='artifact_audit_screen' if out['scope']=='screen_only' else 'artifact_audit'
    return W.ROOT/(stem+('' if out['complete'] else '_partial')+'.json')


def ready_artifact(name,fold,replicate):
    path=W.ROOT/name/f'fold{fold}/seed{replicate}.json'
    row=P.read(path) if path.exists() else {}
    return bool(row.get('challenge') and row.get('pruned_source_sha256') and
                not (W.MODES/name/f'fold{fold}/seed{replicate}.npz').exists())


def audit(screen_only=False):
    expected=sources()
    required=[(name,f,r) for name in (*PRIMARY,*CONTROLS) for f in range(3) for r in (0,1)]
    # Do not ask the numeric comparator to read an in-flight scorer's row.
    primary={(name,f,r) for name in PRIMARY for f in range(3) for r in (0,1)}
    accepted={key for key in primary if ready_artifact(*key)}
    ready=accepted==primary
    arms=primary_eligibility() if ready else None
    screen_complete=ready and all(x['status']=='complete' for x in arms.values())
    eligible=[name for name in C.CANDIDATES if arms[name]['passes']] if screen_complete else []
    if screen_complete and eligible and not screen_only:
        required += [(name,f,r) for name in ('original32','current32','empirical32',*eligible)
                     for f in range(3,9) for r in (0,1)]
    # Freeze acceptance before cache hashing. A row finishing later must wait
    # for the next audit, including when it changes conditional eligibility.
    accepted.update(key for key in required if key not in primary and ready_artifact(*key))
    if not screen_only:
        allowed=set(required)
        for p in W.ROOT.glob('*/fold*/seed*.json'):
            key=(p.parents[1].name,int(p.parent.name[4:]),int(p.stem[4:]))
            require(key in allowed,'unplanned result outside fixed conditional study')
    represented={f for name,f,r in accepted}
    caches={f:cache_metadata(f,P.verify_cache(f)) for f in sorted(represented)}
    fits={(f,r):P.verify_fit(f,r) for f in represented for r in (0,1)}
    artifacts=[];missing=[]
    for name,f,r in required:
        result=one(name,f,r,expected,caches.get(f),fits) if (name,f,r) in accepted else None
        if result is None:missing.append({'recipe':name,'fold':f,'replicate':r})
        else:artifacts.append(result)
    if not missing:mode_artifact_set(required,screen_only)
    return {'status':'passed' if not missing else 'incomplete','complete':not missing,
            'scope':'screen_only' if screen_only else 'full_conditional_study',
            'verified_artifacts':len(artifacts),'required_artifacts':len(required),'missing':missing,
            'prelaunch_manifest_sha256':PRELAUNCH_SHA256,'pinned_source_files':len(expected),
            'verified_cache_folds':sorted(caches),'cache_payload_hashes_checked':6*len(caches),
            'verified_original_fits':len(fits),'artifacts':artifacts,'screen_eligible':eligible,
            'confirmation_disposition':('not_triggered' if screen_complete and not eligible else
                                         'required_for_eligible' if eligible else 'eligibility_incomplete'),
            'default_changed':False,'auditor_sha256':W.sha(Path(__file__)),
            'limitations':['Hash/metadata verification, not independent rescoring of pruned arrays',
                           'Calibration counts, streams and radial digests are recorded metadata; execution rests on pinned code',
                           'Numerical superiority and actual CLI proof remain separate',
                           'Same37 development donors; no biological calibration claim']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-incomplete',action='store_true')
    p.add_argument('--screen-only',action='store_true')
    a=p.parse_args();out=audit(a.screen_only)
    path=audit_path(out)
    W.write(path,out)
    print(json.dumps({k:out[k] for k in ('status','scope','verified_artifacts','required_artifacts')}))
    if not out['complete'] and not a.allow_incomplete:raise SystemExit('complete declared artifact set required')
