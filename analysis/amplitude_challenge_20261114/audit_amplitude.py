"""Read-only conditional provenance proof, including explicit feasibility stops."""
import argparse
import json
from pathlib import Path
import amp_common as B
import generate_amplitude as D
import compare_amplitude as C
import run_amplitude as R
PRELAUNCH_SHA256='17ab33dae2bf13c1995eea5b74a2aa813ccb635643db7cb00a089118f44020e4'


def serialized_equal(saved,recomputed):
    # Frozen compare.verdict uses integer fold keys; JSON receipts use strings.
    # Compare the exact persisted representation, preserving all numeric values.
    return saved==json.loads(json.dumps(recomputed,allow_nan=False))


def ordered_jobs(folds):
    return [(name,f,r) for f in folds for r in (0,1) for name in B.RECIPES]


def ready(recipe,fold,replicate):
    score=B.ROOT/recipe/f'fold{fold}/seed{replicate}.json'
    output=B.MODES/recipe/f'fold{fold}/seed{replicate}.npz'
    if not score.exists():return False
    row=B.P.read(score)
    return bool(row.get('challenge') and row.get('pruned_source_sha256') and not output.exists())


def exact_artifacts(required,extra_raw=()):
    scores={B.ROOT/name/f'fold{f}/seed{r}.json' for name,f,r in required}
    raw={B.ROOT/'raw_laws'/f'fold{f}/seed{r}.json' for name,f,r in [*required,*extra_raw] if name=='amplitude32'}
    B.require(set(B.ROOT.glob('*/fold*/seed*.json'))==scores|raw,'unplanned or missing score/raw-law artifacts')
    receipts={B.MODES/name/f'fold{f}/seed{r}.receipt.json' for name,f,r in required}
    B.require({p for p in B.MODES.rglob('*') if p.is_file()}==receipts,'unplanned/missing receipts or orphan archives')


def verify_infeasible(status,sources):
    name,fold,rep=status['recipe'],status['fold'],status['replicate']
    B.require(name=='amplitude32' and fold in range(9) and rep in (0,1),'invalid infeasible job')
    path=B.ROOT/'raw_laws'/f'fold{fold}/seed{rep}.json';record=B.P.read(path);law=record['law']
    cache=B.A.cache_metadata(fold,B.P.verify_cache(fold))
    calibration=B.MODES/'current32'/f'fold{fold}/seed{rep}.receipt.json'
    D.validate_raw_record(record,cache,fold,rep,sources,B.sha(calibration),B.P.verify_fit(fold,rep))
    B.require(record['sources']==sources and record['fold']==fold and record['replicate']==rep
              and record['fit_hashes']==B.P.verify_fit(fold,rep)
              and record['cache_manifest_sha256']==cache['manifest_sha256']
              and record['current_receipt_sha256']==B.sha(calibration)
              and status['raw_law_sha256']==B.sha(path) and law['fit_donors']==cache['train_donors']
              and law['n_train_targets']==len(cache['train_donors'])
              and [x['donor'] for x in law['donor_fits']]==cache['train_donors'],'wrong infeasible TRAIN law lineage')
    for rec in law['donor_fits']:
        D.validate_fit(rec)
        B.require(rec['generated_bags']==32 and rec['generated_cells']==32*256
                  and rec['generated_gene_values']==32*256*2000,'wrong infeasible fitting dimensions')
    value,normal=D.F.normalize([x['raw_t'] for x in law['donor_fits']])
    B.require(value is None and all(law[k]==v for k,v in normal.items())
              and not (B.MODES/name/f'fold{fold}/seed{rep}.npz').exists()
              and not (B.ROOT/name/f'fold{fold}/seed{rep}.json').exists(),'unproven infeasibility or forbidden candidate')
    return name,fold,rep


def audit():
    manifest=B.HERE/'source_manifest_prelaunch.json'
    B.require(B.sha(manifest)==PRELAUNCH_SHA256,'wrong committed prelaunch pins')
    sources=B.P.read(manifest)
    B.require(sources==B.P.read(B.ROOT/'source_manifest.json')==B.source_manifest(),'changed execution sources')
    R.previous_completed()
    status=B.P.read(B.ROOT/'status.json');infeasible=status['status']=='scientifically_infeasible'
    screen=None;screen_passes=None;failed=None
    primary=ordered_jobs(range(3))
    # Freeze readiness before hashes; a row finalizing now is missing in this snapshot.
    accepted={key for key in primary if ready(*key)}
    if infeasible:
        failed=verify_infeasible(status,sources);all_jobs=ordered_jobs(range(9))
        required=all_jobs[:all_jobs.index(failed)]
        if failed[1]>=3:
            screen=C.report((0,1,2));screen_passes=screen['verdict']['passes']
            B.require(screen_passes,'confirmation infeasibility without a screen pass')
    else:
        if len(accepted)==24:
            screen=C.report((0,1,2));screen_passes=screen['verdict']['passes']
        required=ordered_jobs(range(9 if screen_passes else 3))
    accepted.update(key for key in required if key not in primary and ready(*key))
    represented={f for name,f,r in accepted}
    for fold in represented:B.P.verify_cache(fold)
    for fold in represented:
        for rep in (0,1):B.P.verify_fit(fold,rep)
    verified=[];missing=[]
    for name,fold,rep in required:
        if (name,fold,rep) not in accepted:
            missing.append({'recipe':name,'fold':fold,'replicate':rep});continue
        B.require(D.finalized(name,fold,rep,sources),'invalid finalized artifact')
        score=B.ROOT/name/f'fold{fold}/seed{rep}.json';receipt=B.MODES/name/f'fold{fold}/seed{rep}.receipt.json'
        row=B.P.read(score)
        B.require(B.valid_digest(row['pruned_source_sha256']),'invalid pruned generated digest')
        verified.append({'recipe':name,'fold':fold,'replicate':rep,'score_sha256':B.sha(score),
                         'receipt_sha256':B.sha(receipt),'pruned_generated_sha256':row['pruned_source_sha256']})
    proof=B.ROOT/'constant_amplitude_fidelity.json';control_verified=False
    if proof.exists():D.verify_fidelity(sources);control_verified=True
    terminal=status['status'] in ('completed','scientifically_infeasible')
    complete=not missing and terminal
    if complete:
        B.require(control_verified,'full constant amplitude calibration fidelity required')
        exact_artifacts(required,[failed] if failed else [])
        if not infeasible:
            B.require(status['screen_passes']==screen_passes and serialized_equal(B.P.read(B.ROOT/'screen_comparison.json'),screen),
                      'terminal screen comparison/status differs')
            if screen_passes:B.require(serialized_equal(B.P.read(B.ROOT/'confirmation_comparison.json'),C.report(tuple(range(3,9)),True)),
                                       'terminal confirmation comparison differs')
    B.require(R.disk_bytes()<3_000_000_000,'combined output cap exceeded')
    return {'status':'passed' if complete else 'incomplete','complete':complete,'scope':'full_conditional_study',
            'scientifically_infeasible':infeasible,'infeasible_job':list(failed) if failed else None,
            'verified_artifacts':len(verified),'required_artifacts':len(required),'missing':missing,'artifacts':verified,
            'screen_passes':screen_passes,'confirmation_disposition':'stopped_infeasible' if infeasible else
            'required' if screen_passes else 'not_triggered' if screen is not None else 'eligibility_incomplete',
            'prelaunch_manifest_sha256':PRELAUNCH_SHA256,'pinned_source_files':len(sources),
            'verified_cache_folds':sorted(represented),'cache_payload_hashes_checked':6*len(represented),
            'verified_original_fits':2*len(represented),'default_changed':False,
            'constant_amplitude_fidelity_verified':control_verified,
            'constant_amplitude_fidelity_sha256':B.sha(proof) if control_verified else None,
            'auditor_sha256':B.sha(Path(__file__)),
            'limitations':['Hashes/recorded metadata; not independent rescoring of pruned arrays',
                'Constant.5 control reproduces full current32 arrays and joint calibration; no superiority claim',
                'Scalar simulation random effect, not biological donor/style decomposition or patient features',
                'Same37 reused development donors; numeric superiority and public CLI proof remain separate']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args();out=audit()
    B.write(B.ROOT/('artifact_audit.json' if out['complete'] else 'artifact_audit_partial.json'),out)
    print(json.dumps({k:out[k] for k in ('status','verified_artifacts','required_artifacts','scientifically_infeasible')}))
    if not out['complete'] and not args.allow_incomplete:raise SystemExit('complete conditional artifact set required')
