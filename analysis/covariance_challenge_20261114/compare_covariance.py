"""Unchanged prospective gates against BOTH frozen and fresh training controls."""
from pathlib import Path
import cov_common as C
import resource_common as U
import generate_covariance as D
import compare as ORIGINAL
C.B.require(Path(ORIGINAL.__file__).resolve()==C.B.W.HERE/'compare.py','wrong frozen comparator module')


def job_keys(folds):
    return {(name,fold,seed) for name in C.RECIPES for fold in folds for seed in (0,1)}


def report(folds,confirmation=False,ready_keys=None):
    folds=tuple(folds)
    C.B.require(folds==(tuple(range(3,9)) if confirmation else tuple(range(3))),
                'comparison must use the complete prospectively fixed phase')
    sources=C.B.P.read(C.ROOT/'source_manifest.json')
    C.B.require(C.sources()==sources,'sources changed before comparison')
    rows={};hashes={}
    for name in C.RECIPES:
        rows[name]={}
        for fold in folds:
            for seed in (0,1):
                if ready_keys is not None and (name,fold,seed) not in ready_keys:continue
                p=C.ROOT/name/f'fold{fold}/seed{seed}.json'
                if not p.exists():continue
                C.B.require(D.finalized(name,fold,seed,sources),'unfinalized comparison input')
                rows[name][fold,seed]=C.B.P.read(p);hashes[str(p)]=C.B.sha(p)
    verdicts={label:ORIGINAL.verdict(rows['covariance32'],rows[control],rows['empirical32'],rows['original32'],
                                   tuple(folds),confirmation)
              for label,control in (('frozen_default_control','current32'),('fresh_training_control','training_control32'))}
    return {'phase':'confirmation' if confirmation else 'screen','folds':list(folds),'candidate':'covariance32',
            'verdicts':verdicts,'passes':all(v['passes'] for v in verdicts.values()),'score_sha256':hashes,
            'default_changed':False,'current32_label_within_verdict':{'frozen_default_control':'current32',
                'fresh_training_control':'training_control32'},
            'scope':'Same37 repeatedly selected development donors; computational replicate noise, not biological significance',
            'note':'No causal feature-content claim; native-cap matching does not hold all critic geometry fixed'}
