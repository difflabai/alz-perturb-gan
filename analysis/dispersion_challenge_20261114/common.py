"""Frozen dependencies, namespaces and fingerprints for one dispersion probe."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='3'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'hybrid_challenge_20261114'
sys.path.insert(0,str(PREVIOUS))
import generate_hybrid as H

np=H.np;torch=H.W.torch;G=H.G;T=H.T;AS=H.AS;BL=H.BL;ZC=H.W.ZC;CS=H.W.CS;BC=H.W.BC
P=H.P;A=H.A;W=H.W
sha=H.sha;write=H.write;require=H.require
ROOT=W.RUNS/'results_r5/dispersion_challenge_20261114'
MODES=W.RUNS/'smoke_r5/dispersion_challenge_20261114'
RECIPES=('original32','current32','empirical32','factor32')
NAMESPACE='dispersion20261114; see committed PROTOCOL.txt'
PRIOR_MANIFEST_SHA='e4098a9d08d63a451c08709f4e5a9317b8441e618a1bcfd60325e3c517dbdf24'


def source_manifest():
    path=PREVIOUS/'source_manifest_prelaunch.json'
    require(sha(path)==PRIOR_MANIFEST_SHA,'prior178 manifest changed')
    expected=P.read(path)
    require(H.source_manifest()==expected,'prior178 dependency changed')
    paths=[HERE/name for name in ('PROTOCOL.txt','common.py','factor_model.py','generate_factor.py',
                                  'compare_factor.py','run_factor.py','prior_study_complete_evidence.json')]
    return {**expected,**{str(p):sha(p) for p in paths}}


def streams(fold,index,replicate):
    children=np.random.SeedSequence([20261114,3,fold,index,replicate]).spawn(5)
    return {k:int(child.generate_state(1,dtype=np.uint64)[0])%(2**63-16)
            for k,child in zip(G.GEN_STREAMS,children)}


def empirical_rng(real,target,replicate):
    return np.random.default_rng(AS.E.MX.stable_seed('dispersion-challenge',20261114,
        real.panel,real.fold,real.donors[target],replicate))


def factor_seed(fold,donor,replicate):
    return AS.E.MX.stable_seed('donor-dispersion-factor',20261114,'hvg2000',fold,donor,replicate)


def array_fingerprints(arrays):
    output={}
    for key in (*H.ARRAYS,*H.IDENTITY):
        value=np.ascontiguousarray(arrays[key])
        digest=hashlib.sha256(json.dumps([str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        digest.update(value.tobytes())
        output[key]=digest.hexdigest()
    return output


def valid_digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def validate_fingerprints(value):
    require(isinstance(value,dict) and set(value)==set((*H.ARRAYS,*H.IDENTITY))
            and all(valid_digest(digest) for digest in value.values()),'invalid common array fingerprints')


def archive_fingerprints(path):
    with np.load(path,allow_pickle=False) as arrays:return array_fingerprints(arrays)
