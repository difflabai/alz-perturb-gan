"""Strict conditional metadata/hash audit; no array rescoring or model loading."""
import argparse
import json
from pathlib import Path
import cov_common as C
import resource_common as U
import generate_covariance as D
import compare_covariance as COMPARE


serialized_equal=D.serialized_equal


def snapshot_ready(keys):
    """Freeze readiness before potentially slow hashing; later arrivals remain missing."""
    ready=set()
    for recipe,fold,seed in keys:
        score=C.ROOT/recipe/f'fold{fold}/seed{seed}.json'
        archive=C.MODES/recipe/f'fold{fold}/seed{seed}.npz'
        if not score.exists() or archive.exists() or not archive.with_suffix('.receipt.json').exists():continue
        row=C.B.P.read(score)
        if row.get('challenge') and row.get('pruned_source_sha256'):ready.add((recipe,fold,seed))
    return ready


def expected_trees(keys,complete):
    """Exact scientific artifact names; native status metadata is retained."""
    scores={C.ROOT/name/f'fold{fold}/seed{seed}.json' for name,fold,seed in keys}
    modes={C.MODES/name/f'fold{fold}/seed{seed}.receipt.json' for name,fold,seed in keys}
    if not complete:modes|={p.with_suffix('').with_suffix('.npz') for p in modes}
    modes|={C.MODES/'orchestration_identity'/f'{name}.receipt.json' for name in ('direct','passthrough')}
    if not complete:modes|={C.MODES/'orchestration_identity'/f'{name}.npz' for name in ('direct','passthrough')}
    fits=set()
    for name,fold,seed in keys:
        if name not in ('training_control32','covariance32'):continue
        role='control' if name=='training_control32' else 'candidate';root=C.fit_path(role,fold,seed)
        fits|={root/n for n in (*D.META,'progress.json')}
        pruned=root/'control_weight_pruning.json'
        if role=='control':fits.add(pruned)
        if not (role=='control' and pruned.exists()):fits|={root/n for n in D.WEIGHTS}
        if not complete:
            fits|={root/n for n in ('recovery.pt','recovery.partial.pt','final.partial.pt','failure.json')}
            fits|={root/f'.snap_{u}.partial.pt' for u in C.T.snapshot_updates(20000)}
    return scores,modes,fits


def reject_orphans(keys,complete):
    scores,modes,fits=expected_trees(keys,complete)
    # Every retained mode file belongs to a planned receipt/control. Both
    # CPU/GPU training verification scratch trees must be empty at launch.
    for path in C.MODES.rglob('*'):
        if path.is_file():C.B.require(path in modes,f'orphan or unpruned mode file: {path}')
    for path in C.FITS.rglob('*'):
        if path.is_file():C.B.require(path in fits,f'orphan or unpruned fit file: {path}')
    reserved={'source_manifest.json','status.json','queue.lock','orchestration.lock',
        'training_verification.json','training_verification_progress.json',
        'gpu_training_verification.json','gpu_training_verification_progress.json',
        'control_orchestration_identity.json','screen_comparison.json','confirmation_comparison.json',
        'artifact_audit.json','artifact_audit_screen_partial.json','artifact_audit_screen_complete.json',
        'artifact_audit_full_partial.json','artifact_audit_full_complete.json'}
    for path in C.ROOT.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(C.ROOT)
        if len(relative.parts)==1 and relative.name in reserved:continue
        if (len(relative.parts)==2 and relative.parts[0]=='cache_verification'
                and relative.name in {f'fold{fold}.json' for _,fold,_ in keys}):continue
        C.B.require(path in scores,f'orphan result file: {path}')


def check_comparison(name,folds,confirmation,ready,required):
    path=C.ROOT/name
    if not path.exists():
        C.B.require(not required,f'missing required comparison: {name}')
        return None,None
    live=COMPARE.report(folds,confirmation,ready_keys=ready)
    C.B.require(serialized_equal(C.B.P.read(path),live),'stored numerical comparison differs')
    return live,C.B.sha(path)


def audit(folds,require_complete=True):
    folds=tuple(folds)
    C.B.require(folds in (tuple(range(3)),tuple(range(9))),'audit scope must be fixed screen or full conditional study')
    full=len(folds)==9
    C.cpu();sources=C.full_launch_ready();C.prerequisite()
    C.B.require(C.B.P.read(C.ROOT/'source_manifest.json')==sources,'runtime source manifest differs')
    defaults=C.HERE.parent/'improve_20261109'
    C.B.require(C.B.sha(defaults/'default_config.json')=='fecbfce1e92d31a5d8c3d646f0a6dfb036c975b599dbdc880c332c1023bb3c23'
        and C.B.sha(C.HERE.parents[1]/'scripts/microglia.py')=='40d6102e77fa6a4ece2c64cd431db3d907449d7bd69335cb597370d5f860b80e',
        'default/CLI changed during study')
    screen_keys=COMPARE.job_keys(range(3))
    # Freeze the entire potential tree before reading any payload hashes.
    potential=COMPARE.job_keys(range(9))
    ready=snapshot_ready(potential)
    screen_complete=screen_keys<=ready
    screen,screen_hash=check_comparison('screen_comparison.json',tuple(range(3)),False,
        ready,required=screen_complete)
    eligible=bool(screen_complete and screen is not None and screen['passes'])
    C.B.require(not full or eligible,'full study audit requires both successful screen verdicts')
    required=potential if full else screen_keys
    # A completed screen proof describes only its own phase. Successful
    # screen eligibility requires a later separate full proof, never a
    # "study complete" claim at30/90.
    artifacts=[];missing=[];caches={};fits={}
    for fold in folds:
        caches[str(fold)]=C.B.P.verify_cache(fold)
        for seed in (0,1):
            for recipe in C.RECIPES:
                key=(recipe,fold,seed)
                if key not in ready:missing.append([recipe,fold,seed]);continue
                C.B.require(D.finalized(recipe,fold,seed,sources),'invalid finalized artifact')
                score=C.ROOT/recipe/f'fold{fold}/seed{seed}.json'
                receipt=C.MODES/recipe/f'fold{fold}/seed{seed}.receipt.json'
                info=C.B.P.read(receipt)
                artifacts.append({'recipe':recipe,'fold':fold,'replicate':seed,'score_sha256':C.B.sha(score),
                    'receipt_sha256':C.B.sha(receipt),'pruned_generated_sha256':info['generated_SHA256'],
                    'array_fingerprints':info['array_fingerprints']})
                if recipe in ('training_control32','covariance32'):
                    role='control' if recipe=='training_control32' else 'candidate'
                    fit,hashes=D.new_fit(role,fold,seed,allow_pruned=True)
                    fits[f'{role}/fold{fold}/seed{seed}']={'fit_artifacts_sha256':hashes,
                        'weights_pruned':(fit/'control_weight_pruning.json').exists(),
                        'verification_scope':'Current retained weights rehashed; pruned control weights use verified preprune receipts',
                        'peak_scope':'Training receipt records the latest native execution segment if the fit resumed'}
    complete=not missing
    C.B.require(complete or not require_complete,'complete artifact audit has missing jobs')
    proof=D.verify_identity(sources) if (C.ROOT/'control_orchestration_identity.json').exists() else None
    C.B.require(proof is not None or not complete,'full calibration identity missing')
    comparisons={}
    if screen_hash:comparisons['screen_comparison.json']=screen_hash
    confirmation=None
    if full:
        confirmation,h=check_comparison('confirmation_comparison.json',tuple(range(3,9)),True,
            ready,required=complete)
        if h:comparisons['confirmation_comparison.json']=h
    else:C.B.require(not (C.ROOT/'confirmation_comparison.json').exists(),'confirmation comparison outside screen scope')
    reject_orphans(required,complete)
    C.B.require(C.sources()==sources and C.disk_bytes()<3000000000,'audit source/cap differs')
    return {'status':'passed' if complete else 'incomplete','complete':complete,
        'scope_phase':'full' if full else 'screen','folds':list(folds),
        'study_complete':bool(complete and (full or (screen_complete and not eligible))),
        'confirmation_eligible':eligible,'confirmation_triggered':full,
        'verified_artifacts':len(artifacts),'required_artifacts':len(required),'missing':missing,
        'sources':sources,'prelaunch_manifest_sha256':C.B.sha(C.HERE/'source_manifest_prelaunch.json'),
        'native_training_verification_sha256':C.B.sha(C.ROOT/'training_verification.json'),
        'gpu_training_verification_sha256':C.B.sha(C.ROOT/'gpu_training_verification.json'),
        'control_orchestration_identity_sha256':C.B.sha(C.ROOT/'control_orchestration_identity.json') if proof else None,
        'full_real_fresh_TRAIN_joint_calibration_identity_verified':proof is not None,
        'caches':caches,'new_fits':fits,'artifacts':artifacts,'comparison_sha256':comparisons,
        'default_changed':False,'logical_task_bytes':C.disk_bytes(),
        'scope':'Hash/metadata/calibration/pruning verification only; no independent rescoring of pruned arrays or biological validation'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folds',default='0,1,2')
    p.add_argument('--allow-partial',action='store_true');a=p.parse_args();folds=tuple(map(int,a.folds.split(',')))
    with C.leases():result=audit(folds,not a.allow_partial)
    phase=result['scope_phase'];state='complete' if result['complete'] else 'partial'
    C.B.write(C.ROOT/f'artifact_audit_{phase}_{state}.json',result)
    print(json.dumps({'status':result['status'],'verified':result['verified_artifacts']}),flush=True)
