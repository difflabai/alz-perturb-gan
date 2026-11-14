"""Fifth fixed study, preserving every predecessor execution pin."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'branch_geometry_20261114'
sys.path.insert(0,str(PREVIOUS))
import diagnose as Q
D=Q.B.D
np=D.np;torch=D.torch;G=D.G;T=D.T;AS=D.AS;BL=D.BL;P=D.P;A=D.A;W=D.W;H=D.H
ZC=D.ZC;CS=D.CS;BC=D.BC
sha=D.sha;write=D.write;require=D.require
array_fingerprints=D.array_fingerprints;archive_fingerprints=D.archive_fingerprints
valid_digest=D.valid_digest;validate_fingerprints=D.validate_fingerprints
ROOT=W.RUNS/'results_r5/amplitude_challenge_20261114'
MODES=W.RUNS/'smoke_r5/amplitude_challenge_20261114'
RECIPES=('original32','current32','empirical32','amplitude32')
NAMESPACE='amplitude20261114; see committed PROTOCOL.txt'
PRIOR_MANIFEST_SHA='55f2bbd3537dd9fe1868a717fe90d82aaad3853df93316182b58ecb415a0d49e'


def source_manifest():
    path=PREVIOUS/'source_manifest_prelaunch.json'
    require(sha(path)==PRIOR_MANIFEST_SHA,'prior195 manifest changed')
    expected=P.read(path);require(Q.sources()==expected,'prior195 dependency changed')
    names=('PROTOCOL.txt','amp_common.py','amplitude_model.py','generate_amplitude.py',
           'compare_amplitude.py','run_amplitude.py','predecessor_evidence.json')
    return {**expected,**{str(HERE/name):sha(HERE/name) for name in names}}


def streams(fold,index,replicate):
    children=np.random.SeedSequence([20261114,5,fold,index,replicate]).spawn(5)
    return {key:int(child.generate_state(1,dtype=np.uint64)[0])%(2**63-16)
            for key,child in zip(G.GEN_STREAMS,children)}


def empirical_rng(real,target,replicate):
    return np.random.default_rng(AS.E.MX.stable_seed('amplitude-challenge',20261114,
        real.panel,real.fold,real.donors[target],replicate))


def amplitude_seed(fold,donor,replicate):
    return AS.E.MX.stable_seed('donor-composition-amplitude',20261114,'hvg2000',fold,donor,replicate)
