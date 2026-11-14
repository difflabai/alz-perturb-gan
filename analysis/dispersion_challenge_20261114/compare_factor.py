"""The original prospective gates, unchanged, applied to one fixed factor arm."""
import common as B
import generate_factor as D
import compare as C


def load(recipe,folds):
    sources=B.P.read(B.ROOT/'source_manifest.json')
    B.require(B.source_manifest()==sources,'source changed before numeric comparison')
    rows={};hashes={}
    for f in folds:
        for r in (0,1):
            path=B.ROOT/recipe/f'fold{f}/seed{r}.json'
            if not path.exists():continue
            B.require(D.finalized(recipe,f,r,sources),'unfinalized numeric comparison input')
            rows[f,r]=B.P.read(path);hashes[str(path)]=B.sha(path)
    return rows,hashes


def report(folds,confirmation=False):
    rows={};hashes={}
    for name in B.RECIPES:
        rows[name],found=load(name,folds);hashes.update(found)
    result=C.verdict(rows['factor32'],rows['current32'],rows['empirical32'],rows['original32'],tuple(folds),confirmation)
    return {'phase':'confirmation' if confirmation else 'screen','folds':list(folds),
            'candidate':'factor32','verdict':result,'score_sha256':hashes,'default_changed':False,
            'scope':'Independent TRAIN-fitted dispersion simulation parameter; same37 development donors',
            'note':'Numeric gates only; full provenance, independent review and public CLI proof remain separate'}
