"""Fourth-study dependencies, fixed namespaces and bounded CPU affinity."""
import os
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:3]))
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'dispersion_challenge_20261114'
sys.path.insert(0,str(PREVIOUS))
import common as D
np=D.np;torch=D.torch;G=D.G;T=D.T;AS=D.AS;BL=D.BL;P=D.P;A=D.A;W=D.W;H=D.H
sha=D.sha;write=D.write;require=D.require
valid_digest=D.valid_digest;validate_fingerprints=D.validate_fingerprints
array_fingerprints=D.array_fingerprints;archive_fingerprints=D.archive_fingerprints
ROOT=W.RUNS/'results_r5/paired_challenge_20261114'
MODES=W.RUNS/'smoke_r5/paired_challenge_20261114'
RECIPES=('original32','current32','empirical32','paired32')
NAMESPACE='paired20261114; see committed PROTOCOL.txt'
PRIOR_MANIFEST_SHA='a42a25df33429c4eb16cc2160d6f0459eec8bd30200e4641c28e39ea2044513c'


def source_manifest():
    path=PREVIOUS/'source_manifest_prelaunch.json'
    require(sha(path)==PRIOR_MANIFEST_SHA,'prior185 manifest changed')
    expected=P.read(path);require(D.source_manifest()==expected,'prior185 dependency changed')
    names=('PROTOCOL.txt','pair_common.py','pair_cells.py','generate_pair.py',
           'compare_pair.py','run_pair.py','prior_study_complete_evidence.json')
    return {**expected,**{str(HERE/name):sha(HERE/name) for name in names}}


def streams(fold,index,replicate):
    children=np.random.SeedSequence([20261114,4,fold,index,replicate]).spawn(5)
    return {k:int(child.generate_state(1,dtype=np.uint64)[0])%(2**63-16)
            for k,child in zip(G.GEN_STREAMS,children)}


def empirical_seed(panel,fold,donor,replicate,role):
    require(role in ('reference','component'),'invalid empirical stream role')
    return AS.E.MX.stable_seed('paired-challenge',20261114,panel,fold,donor,replicate,role)


def empirical_rng(real,target,replicate,role):
    return np.random.default_rng(empirical_seed(real.panel,real.fold,real.donors[target],replicate,role))


def component_sources(train,target,n_cells,fold,replicate):
    require(type(n_cells) is int and n_cells>0,'invalid frozen target cell cardinality')
    rng=np.random.default_rng(empirical_seed('hvg2000',fold,target,replicate,'component'))
    # Uniform choice consumes indices, independently of library values.
    rng.choice(n_cells,(16,256),replace=True)
    return BL._donor_cycle(rng,[d for d in train if d!=target],16).tolist()
