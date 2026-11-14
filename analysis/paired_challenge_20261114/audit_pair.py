"""Read-only conditional artifact proof, separate from numeric/CLI validation."""
import argparse
import json
from pathlib import Path
import pair_common as B
import generate_pair as D
import compare_pair as C

PRELAUNCH_SHA256='f6feb5844ac3088677c9d7f4324a5630a35df20c98c2e3f75c7aafe887876255'


def ready(recipe,fold,replicate):
    score=B.ROOT/recipe/f'fold{fold}/seed{replicate}.json'
    output=B.MODES/recipe/f'fold{fold}/seed{replicate}.npz'
    if not score.exists():return False
    row=B.P.read(score)
    return bool(row.get('challenge') and row.get('pruned_source_sha256') and not output.exists())


def required_set(screen_passes):
    return [(name,f,r) for name in B.RECIPES for f in range(9 if screen_passes else 3) for r in (0,1)]


def exact_artifacts(required):
    expected_scores={B.ROOT/name/f'fold{f}/seed{r}.json' for name,f,r in required}
    B.require(set(B.ROOT.glob('*/fold*/seed*.json'))==expected_scores,'unplanned or missing score artifacts')
    expected_receipts={B.MODES/name/f'fold{f}/seed{r}.receipt.json' for name,f,r in required}
    B.require({p for p in B.MODES.rglob('*') if p.is_file()}==expected_receipts,
              'unplanned/missing receipts or orphan archive files')


def audit():
    manifest=B.HERE/'source_manifest_prelaunch.json'
    B.require(PRELAUNCH_SHA256 and B.sha(manifest)==PRELAUNCH_SHA256,'wrong committed prelaunch pins')
    expected=B.P.read(manifest)
    B.require(expected==B.P.read(B.ROOT/'source_manifest.json')==B.source_manifest(),'changed execution sources')
    # Freeze readiness before hashing: a row finalizing during this audit
    # remains missing until the next complete snapshot/eligibility pass.
    primary=required_set(False)
    accepted={key for key in primary if ready(*key)}
    primary_ready=len(accepted)==len(primary)
    screen=C.report((0,1,2)) if primary_ready else None
    screen_passes=bool(screen and screen['verdict']['passes'])
    required=required_set(screen_passes)
    accepted.update(key for key in required if key not in primary and ready(*key))
    represented={f for name,f,r in accepted}
    for fold in represented:B.P.verify_cache(fold)
    for fold in represented:
        for rep in (0,1):B.P.verify_fit(fold,rep)
    verified=[];missing=[]
    for name,fold,rep in required:
        if (name,fold,rep) not in accepted:
            missing.append({'recipe':name,'fold':fold,'replicate':rep});continue
        B.require(D.finalized(name,fold,rep,expected),'invalid finalized artifact')
        score=B.ROOT/name/f'fold{fold}/seed{rep}.json'
        receipt=B.MODES/name/f'fold{fold}/seed{rep}.receipt.json'
        row=B.P.read(score)
        B.require(B.valid_digest(row['pruned_source_sha256']),'invalid pruned generated digest')
        verified.append({'recipe':name,'fold':fold,'replicate':rep,'score_sha256':B.sha(score),
                         'receipt_sha256':B.sha(receipt),'pruned_generated_sha256':row['pruned_source_sha256']})
    identity=B.ROOT/'single_cell_identity.json'
    identity_verified=False
    if identity.exists():D.verify_identity(expected);identity_verified=True
    if not missing:
        B.require(identity_verified,'full single-cell identity proof required')
        exact_artifacts(required)
    return {'status':'incomplete' if missing else 'passed','complete':not missing,
            'scope':'full_conditional_study','verified_artifacts':len(verified),'required_artifacts':len(required),
            'missing':missing,'artifacts':verified,'screen_passes':screen_passes if primary_ready else None,
            'confirmation_disposition':'required' if screen_passes else 'not_triggered' if primary_ready else 'eligibility_incomplete',
            'prelaunch_manifest_sha256':PRELAUNCH_SHA256,'pinned_source_files':len(expected),
            'verified_cache_folds':sorted(represented),'cache_payload_hashes_checked':6*len(represented),
            'verified_original_fits':2*len(represented),'default_changed':False,
            'single_cell_identity_verified':identity_verified,'single_cell_identity_sha256':B.sha(identity) if identity_verified else None,
            'auditor_sha256':B.sha(Path(__file__)),
            'limitations':['Hashes/recorded metadata; not independent rescoring of pruned arrays',
                           'Identity control reproduces full frozen empirical arrays with exact common fingerprints',
                           'Numerical superiority, public CLI proof and biological validity are separate',
                           'Same37 reused development donors; paired outputs are artificial pooled cells, not native cells or patient axes']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args();out=audit()
    B.write(B.ROOT/('artifact_audit.json' if out['complete'] else 'artifact_audit_partial.json'),out)
    print(json.dumps({k:out[k] for k in ('status','verified_artifacts','required_artifacts')}))
    if not out['complete'] and not args.allow_incomplete:raise SystemExit('complete conditional artifact set required')
