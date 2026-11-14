"""The original prospective gates, unchanged, applied to one fixed paired-cell arm."""
import pair_common as B
import generate_pair as D
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
    result=C.verdict(rows['paired32'],rows['current32'],rows['empirical32'],rows['original32'],tuple(folds),confirmation)
    return {'phase':'confirmation' if confirmation else 'screen','folds':list(folds),
            'candidate':'paired32','verdict':result,'score_sha256':hashes,'default_changed':False,
            'scope':'Artificial pooled-cell semiparametric simulator; same37 development donors',
            'note':'Numeric gates only; full provenance, independent review and public CLI proof remain separate'}
