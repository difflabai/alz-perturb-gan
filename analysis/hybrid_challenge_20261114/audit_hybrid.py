"""Separate read-only conditional artifact proof; never rescoring or promotion."""
import argparse
import json
from pathlib import Path
import generate_hybrid as H
import compare_hybrid as C

# Filled with the actual committed launch manifest before this auditor is used.
PRELAUNCH_SHA256='e4098a9d08d63a451c08709f4e5a9317b8441e618a1bcfd60325e3c517dbdf24'


def ready(recipe,fold,replicate):
    score=H.ROOT/recipe/f'fold{fold}/seed{replicate}.json'
    output=H.MODES/recipe/f'fold{fold}/seed{replicate}.npz'
    if not score.exists():return False
    row=H.P.read(score)
    return bool(row.get('challenge') and row.get('pruned_source_sha256') and not output.exists())


def required_set(screen_passes):
    return [(name,f,r) for name in H.RECIPES for f in range(9 if screen_passes else 3) for r in (0,1)]


def exact_artifacts(required):
    expected_scores={H.ROOT/name/f'fold{f}/seed{r}.json' for name,f,r in required}
    H.require(set(H.ROOT.glob('*/fold*/seed*.json'))==expected_scores,'unplanned or missing score artifacts')
    expected_receipts={H.MODES/name/f'fold{f}/seed{r}.receipt.json' for name,f,r in required}
    H.require({p for p in H.MODES.rglob('*') if p.is_file()}==expected_receipts,
              'unplanned/missing receipts or orphan component/archive files')


def audit():
    manifest=H.HERE/'source_manifest_prelaunch.json'
    H.require(PRELAUNCH_SHA256 and H.sha(manifest)==PRELAUNCH_SHA256,'wrong committed prelaunch pins')
    expected=H.P.read(manifest)
    H.require(expected==H.P.read(H.ROOT/'source_manifest.json')==H.source_manifest(),'changed execution sources')
    primary=required_set(False)
    accepted={key for key in primary if ready(*key)}
    primary_ready=len(accepted)==len(primary)
    screen=C.report((0,1,2)) if primary_ready else None
    screen_passes=bool(screen and screen['verdict']['passes'])
    required=required_set(screen_passes)
    accepted.update(key for key in required if key not in primary and ready(*key))
    represented={f for name,f,r in accepted}
    for fold in represented:H.P.verify_cache(fold)
    for fold in represented:
        for rep in (0,1):H.P.verify_fit(fold,rep)
    verified=[];missing=[]
    for name,fold,rep in required:
        if (name,fold,rep) not in accepted:
            missing.append({'recipe':name,'fold':fold,'replicate':rep});continue
        H.require(H.finalized(name,fold,rep,expected),'invalid finalized artifact')
        score=H.ROOT/name/f'fold{fold}/seed{rep}.json'
        receipt=H.MODES/name/f'fold{fold}/seed{rep}.receipt.json'
        row=H.P.read(score);info=H.P.read(receipt)
        if name=='hybrid32':
            for role in ('model','empirical'):
                digest=info[role+'_component']['generated_SHA256']
                H.require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),
                          'invalid pruned component digest')
        verified.append({'recipe':name,'fold':fold,'replicate':rep,'score_sha256':H.sha(score),
                         'receipt_sha256':H.sha(receipt),'pruned_generated_sha256':row['pruned_source_sha256']})
    if not missing:exact_artifacts(required)
    return {'status':'incomplete' if missing else 'passed','complete':not missing,
            'scope':'full_conditional_study','verified_artifacts':len(verified),'required_artifacts':len(required),
            'missing':missing,'artifacts':verified,'screen_passes':screen_passes if primary_ready else None,
            'confirmation_disposition':'required' if screen_passes else 'not_triggered' if primary_ready else 'eligibility_incomplete',
            'prelaunch_manifest_sha256':PRELAUNCH_SHA256,'pinned_source_files':len(expected),
            'verified_cache_folds':sorted(represented),'cache_payload_hashes_checked':6*len(represented),
            'verified_original_fits':2*len(represented),'default_changed':False,
            'auditor_sha256':H.sha(Path(__file__)),
            'limitations':['Hashes/recorded metadata; not independent rescoring of pruned component or mixture arrays',
                           'Exact bag concatenation rests on pinned code and synthetic checks',
                           'Numerical superiority, public CLI proof and biological validity are separate',
                           'Same37 reused development donors; semiparametric simulator, not patient-vector evidence']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args();out=audit()
    H.write(H.ROOT/('artifact_audit.json' if out['complete'] else 'artifact_audit_partial.json'),out)
    print(json.dumps({k:out[k] for k in ('status','verified_artifacts','required_artifacts')}))
    if not out['complete'] and not args.allow_incomplete:raise SystemExit('complete conditional artifact set required')
