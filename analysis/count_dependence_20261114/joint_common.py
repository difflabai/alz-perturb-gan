"""Frozen fifth-study ancestry for one bounded TRAIN-only diagnostic."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'amplitude_challenge_20261114'
sys.path.insert(0,str(PREVIOUS))
import amp_common as B
np=B.np;torch=B.torch
ROOT=B.W.RUNS/'results_r5/count_dependence_20261114'
SMOKE=B.W.RUNS/'smoke_r5/count_dependence_20261114'
PRIOR_SHA='17ab33dae2bf13c1995eea5b74a2aa813ccb635643db7cb00a089118f44020e4'
TASKS=(*B.Q.TASKS,'amplitude_challenge_20261114','count_dependence_20261114')


def sources():
    path=PREVIOUS/'source_manifest_prelaunch.json'
    B.require(B.sha(path)==PRIOR_SHA,'prior202 manifest changed')
    prior=B.P.read(path);B.require(B.source_manifest()==prior,'prior202 dependency changed')
    return {**prior,**{str(HERE/name):B.sha(HERE/name) for name in
        ('PROTOCOL.txt','joint_common.py','gamma_head.py','diagnose_joint.py','predecessor_evidence.json')}}


def disk_bytes():
    return sum(p.stat().st_size for root in ('fits_r5','smoke_r5','results_r5','logs_r5')
               for task in TASKS for p in (B.W.RUNS/root/task).rglob('*') if p.is_file())


def groups(loadings):
    w=np.asarray(loadings,np.float64)
    B.require(w.ndim==2 and w.shape[1]==50 and np.isfinite(w).all(),'invalid frozen50PC loadings')
    pc=np.argmax(np.abs(w),axis=1)
    return (2*pc+(w[np.arange(len(w)),pc]<0)).astype(np.int64)
