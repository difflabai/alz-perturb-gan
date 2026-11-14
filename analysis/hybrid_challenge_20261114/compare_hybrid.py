"""Exact prior numerical gates, one fixed hybrid, fresh matched references."""
import json
import generate_hybrid as H
import compare as C


def load(recipe,folds):
    expected=H.P.read(H.ROOT/'source_manifest.json')
    H.require(H.source_manifest()==expected,'comparison source changed')
    rows={};hashes={}
    for fold in folds:
        for replicate in (0,1):
            path=H.ROOT/recipe/f'fold{fold}/seed{replicate}.json'
            if not path.exists():continue
            H.require(H.finalized(recipe,fold,replicate,expected),'unfinalized comparison input')
            rows[fold,replicate]=H.P.read(path);hashes[str(path)]=H.sha(path)
    return rows,hashes


def report(folds,confirmation=False):
    rows={};hashes={}
    for name in H.RECIPES:
        rows[name],found=load(name,folds);hashes.update(found)
    result=C.verdict(rows['hybrid32'],rows['current32'],rows['empirical32'],rows['original32'],
                     tuple(folds),confirmation)
    return {'phase':'confirmation' if confirmation else 'screen','folds':list(folds),
            'candidate':'hybrid32','verdict':result,'score_sha256':hashes,'default_changed':False,
            'scope':'Fixed semiparametric16/16 whole-bag simulator; same37 development donors',
            'note':'Numerical gates only; full artifact audit, review and actual public CLI proof remain separate'}
