"""Sixth scored hypothesis; ancestry and bounded verification resources."""
from contextlib import ExitStack,contextmanager
import fcntl
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'count_dependence_20261114'
sys.path.insert(0,str(PREVIOUS))
import joint_common as J
import diagnose_joint as DIAG
B=J.B;np=J.np;torch=J.torch;T=B.T;G=B.G
RUNS=B.W.RUNS
ROOT=RUNS/'results_r5/covariance_challenge_20261114'
MODES=RUNS/'smoke_r5/covariance_challenge_20261114'
FITS=RUNS/'fits_r5/covariance_challenge_20261114'
LOG=RUNS/'logs_r5/covariance_challenge_20261114'
TASKS=(*J.TASKS,'covariance_challenge_20261114')
ARM='d.covmom,pbulk,t.kappa@0.679975'
KAPPA=float(format((2150/4650)**.5,'.6g'))
PRIOR_SHA='103cc98aea29ce16fc8d4bec3961bc0aac7b801413da7db37305e09681c53f69'
VERIFY_FILES=('PROTOCOL.txt','cov_common.py','covmom.py','train_covariance.py',
              'verify_training.py','predecessor_evidence.json')
FULL_FILES=(*VERIFY_FILES,'generate_covariance.py','compare_covariance.py','audit_covariance.py','run_covariance.py')
RECIPES=('original32','current32','empirical32','cpu_control32','covariance32')
NAMESPACE='covariance20261114; see committed PROTOCOL.txt'


def verification_sources():
    p=PREVIOUS/'source_manifest_prelaunch.json'
    B.require(B.sha(p)==PRIOR_SHA,'wrong prior207 manifest')
    old=B.P.read(p);B.require(J.sources()==old,'prior207 source changed')
    return {**old,**{str(HERE/n):B.sha(HERE/n) for n in VERIFY_FILES}}


def sources():
    return {**verification_sources(),**{str(HERE/n):B.sha(HERE/n) for n in FULL_FILES}}


def full_launch_ready():
    expected=B.P.read(HERE/'source_manifest_prelaunch.json')
    B.require(sources()==expected,'full study sources changed')
    launch=B.P.read(HERE/'prelaunch_verification.json')
    proof=B.P.read(ROOT/'training_verification.json')
    B.require(launch['status']=='cleared_before_full_screen_execution'
        and launch['source_manifest_sha256']==B.sha(HERE/'source_manifest_prelaunch.json')
        and launch['training_verification_sha256']==B.sha(ROOT/'training_verification.json')
        and proof['status']=='passed_native_TRAIN_parity_and_candidate_smoke'
        and proof['sources']==verification_sources()
        and proof['verification_manifest_sha256']==B.sha(HERE/'verification_source_manifest_prelaunch.json'),
        'full launch preflight incomplete')
    return expected


def streams(fold,index,replicate):
    children=np.random.SeedSequence([20261114,6,fold,index,replicate]).spawn(5)
    return {key:int(child.generate_state(1,dtype=np.uint64)[0])%(2**63-16)
            for key,child in zip(G.GEN_STREAMS,children)}


def empirical_rng(real,target,replicate):
    return np.random.default_rng(B.AS.E.MX.stable_seed('covariance-challenge',20261114,
        real.panel,real.fold,real.donors[target],replicate))


def fit_path(role,fold,seed):
    return FITS/role/('pbulk' if role=='control' else ARM)/f'hvg2000/fold{fold}/seed{seed}'


def prerequisite():
    pin=B.P.read(HERE/'predecessor_evidence.json')
    for name,h in pin['runtime_sha256'].items():B.require(B.sha(J.ROOT/name)==h,'prior TRAIN evidence changed')
    x=B.P.read(J.ROOT/'report.json')
    B.require(x['status']=='completed_TRAIN_count_dependence_diagnostic_only'
        and not x['decision']['passes_TRAIN_mechanism_only'] and not x['default_changed'],'prior TRAIN stop required')
    B.require(x['compact_moments_sha256']==pin['runtime_sha256']['compact_moments.npz']
        and x['zero_replay_fidelity_sha256']==pin['runtime_sha256']['zero_replay_fidelity.json'],'wrong prior TRAIN links')
    DIAG.prerequisite()


def disk_bytes():
    return sum(p.stat().st_size for name in TASKS for folder in ('fits_r5','smoke_r5','results_r5','logs_r5')
               for p in (RUNS/folder/name).rglob('*') if p.is_file())


def cpu():
    torch.set_num_threads(3)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:3])
    B.require(not torch.cuda.is_initialized(),'CPU-only study')


@contextmanager
def leases():
    roots=[RUNS/'results_r5'/name for name in TASKS if name!='architecture_pass']
    roots.append(RUNS/'logs_r5/architecture_pass')
    with ExitStack() as stack:
        for root in roots:
            root.mkdir(parents=True,exist_ok=True)
            f=stack.enter_context((root/'queue.lock').open('a'))
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def train_scope(fold,opened):
    cache=RUNS/f'caches/hvg2000/fold{fold}'
    sys.addaudithook(B.Q.read_guard(cache,opened))
    anchor=B.P.verify_cache(fold,payloads=False)
    for name in ('space.npz','train/counts.npz','train/metadata.npz'):
        B.require(B.sha(cache/name)==anchor['payload_sha256'][name],'TRAIN payload changed')
    return cache,anchor


def check_train_scope(cache,opened):
    permitted={str((cache/n).resolve()) for n in ('manifest.json','space.npz','train/counts.npz','train/metadata.npz')}
    B.require(opened==permitted,'TRAIN exact-four-open scope differs')
