"""Native matched GPU trainer; generation deferred, TRAIN-only four-file guard."""
import os
SELECTOR=os.environ.get('ALZ_COVARIANCE_GPU','1')
import argparse
import cov_common as C
import resource_common as U
import covmom_gpu
import train_covariance as TRAIN


def train(fold,seed,role):
    sources=C.full_launch_ready()
    import generate_covariance as D
    eligibility=D.require_confirmation_eligible(fold,sources)
    U.setup_gpu(SELECTOR)
    opened=set();cache,anchor=C.train_scope(fold,opened)
    C.prerequisite();C.B.require(C.disk_bytes()<2850000000,'less than150MB training headroom')
    arm='pbulk' if role=='control' else C.ARM
    args=C.T.parser().parse_args(['--cache',str(cache),'--arm',arm,'--seed',str(seed),'--updates','20000',
        '--out-root',str(C.FITS/role),'--device',U.TRAIN_DEVICE,'--threads','3','--gpu-mem-fraction','.04'])
    peak={'peak_logical_task_bytes':C.disk_bytes()}
    with TRAIN.deferred_generation(),TRAIN.watch_checkpoint_peak(peak):fit=C.T.run(args)
    C.check_train_scope(cache,opened)
    C.B.require(C.sources()==sources and C.disk_bytes()<3000000000,'post-fit source/cap check failed')
    receipt=C.B.P.read(fit/'fit_receipt.json')
    C.B.require(receipt['status']=='training_completed' and receipt['completed_updates']==20000
        and receipt['device']==U.TRAIN_DEVICE and receipt['threads']==3,'native training endpoint differs')
    C.B.write(fit/'study_training_receipt.json',{'status':'native_training_completed_TRAIN_only','role':role,
        'fold':fold,'seed':seed,'fit':str(fit),'sources':sources,'cache_files_opened':sorted(opened),
        'cache_manifest_sha256':anchor['manifest_sha256'],'fit_receipt_sha256':C.B.sha(fit/'fit_receipt.json'),
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'default_changed':False,
        'gpu_resource':U.gpu_resource(),'confirmation_eligibility':eligibility,**peak,
        'weight_sha256':{p.name:C.B.sha(p) for p in fit.glob('*.pt')},'automatic_generation_deferred':True})
    return fit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold',type=int,choices=range(9),required=True)
    p.add_argument('--seed',type=int,choices=(0,1),required=True)
    p.add_argument('--role',choices=('control','candidate'),required=True)
    a=p.parse_args()
    with C.leases():train(a.fold,a.seed,a.role)
