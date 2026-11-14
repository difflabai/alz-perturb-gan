"""Isolated native trainer route; automatic post-fit generation is deferred.

The frozen T.run/advance/restore implementations execute unchanged. Training
does not open target/held depth or counts. Generation is a separate phase.
"""
from contextlib import contextmanager
import argparse
import io
import os
from pathlib import Path
import cov_common as C
import covmom  # registers exactly one isolated observation plug-in


@contextmanager
def deferred_generation():
    original=C.G.generate_fit
    def defer(*args,**kwargs):return None,{'status':'deferred_to_separate_generation_phase'}
    C.G.generate_fit=defer
    try:yield
    finally:C.G.generate_fit=original


@contextmanager
def watch_checkpoint_peak(stats):
    original=C.torch.save
    def save(obj,path,*args,**kwargs):
        # Serialize once with native Torch into memory, preflight its EXACT
        # byte length, then write those bytes. Native T.run still performs
        # its own atomic rename and unlink; model/state/update calls do not
        # change. The archive root name is Torch's file-like default.
        path=Path(path);buffer=io.BytesIO();original(obj,buffer,*args,**kwargs)
        size=buffer.getbuffer().nbytes;old_size=path.stat().st_size if path.exists() else 0
        predicted=C.disk_bytes()-old_size+size
        stats['max_serialized_checkpoint_bytes']=max(stats.get('max_serialized_checkpoint_bytes',0),size)
        C.B.require(predicted<3000000000,'checkpoint preflight would exceed3GB')
        with path.open('wb') as stream:stream.write(buffer.getbuffer())
        # Temporary and old checkpoint coexist until the native rename.
        value=C.disk_bytes();stats['peak_logical_task_bytes']=max(stats.get('peak_logical_task_bytes',0),value)
        C.B.require(value<3000000000,'checkpoint temporary-file peak exceeds3GB')
        return None
    C.torch.save=save
    try:yield
    finally:C.torch.save=original


def train(fold,seed,role):
    C.cpu();C.full_launch_ready();opened=set();cache,anchor=C.train_scope(fold,opened)
    sources=C.verification_sources();pinned=C.B.P.read(C.HERE/'verification_source_manifest_prelaunch.json')
    C.B.require(sources==pinned,'verification dependencies changed')
    C.prerequisite();C.B.require(C.disk_bytes()<2850000000,'less than150MB training headroom')
    arm='pbulk' if role=='control' else C.ARM
    root=C.FITS/role
    args=C.T.parser().parse_args(['--cache',str(cache),'--arm',arm,'--seed',str(seed),'--updates','20000',
        '--out-root',str(root),'--device','cpu','--threads','3'])
    peak={'peak_logical_task_bytes':C.disk_bytes()}
    with deferred_generation(),watch_checkpoint_peak(peak):fit=C.T.run(args)
    C.check_train_scope(cache,opened)
    C.B.require(C.verification_sources()==sources and C.disk_bytes()<3000000000,'post-fit source/cap check failed')
    receipt=C.B.P.read(fit/'fit_receipt.json')
    C.B.require(receipt['status']=='training_completed' and receipt['completed_updates']==20000
        and receipt['device']=='cpu' and receipt['threads']==3,'native training endpoint differs')
    C.B.write(fit/'study_training_receipt.json',{'status':'native_training_completed_TRAIN_only','role':role,
        'fold':fold,'seed':seed,'fit':str(fit),'sources':sources,'cache_files_opened':sorted(opened),
        'cache_manifest_sha256':anchor['manifest_sha256'],'fit_receipt_sha256':C.B.sha(fit/'fit_receipt.json'),
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'default_changed':False,
        **peak,
        'weight_sha256':{p.name:C.B.sha(p) for p in fit.glob('*.pt')},'automatic_generation_deferred':True})
    return fit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold',type=int,choices=range(9),required=True)
    p.add_argument('--seed',type=int,choices=(0,1),required=True)
    p.add_argument('--role',choices=('control','candidate'),required=True)
    a=p.parse_args()
    with C.leases():train(a.fold,a.seed,a.role)
