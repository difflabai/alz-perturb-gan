"""Prospective GPU resource binding; the historical CPU213 sources stay frozen."""
import os
import cov_common as C

RESOURCE_FILES=('RESOURCE_AMENDMENT.txt','resource_common.py','covmom_gpu.py','train_gpu.py','verify_gpu.py')
TRAIN_DEVICE='cuda'
GPU_SELECTOR='1'
MEMORY_FRACTION=.04
GPU_STATUS='passed_native_GPU_TRAIN_parity_and_candidate_smoke'
_cpu_sources=C.sources
_cpu_full_ready=C.full_launch_ready


def sources():
    return {**_cpu_sources(),**{str(C.HERE/n):C.B.sha(C.HERE/n) for n in RESOURCE_FILES}}


def cpu_proof():
    p=C.ROOT/'training_verification.json';proof=C.B.P.read(p)
    C.B.require(proof['sources']==C.verification_sources()
        and proof['sources']==C.B.P.read(C.HERE/'verification_source_manifest_prelaunch.json')
        and proof['status']=='passed_native_TRAIN_parity_and_candidate_smoke'
        and proof['verification_manifest_sha256']==C.B.sha(C.HERE/'verification_source_manifest_prelaunch.json')
        and C.B.sha(p)==C.B.sha(C.HERE/'cpu_training_verification_complete_20261114.json'),
        'CPU verification proof differs')
    return C.B.sha(p)


def verification_ready():
    expected=C.B.P.read(C.HERE/'gpu_verification_source_manifest_prelaunch.json')
    clearance=C.B.P.read(C.HERE/'gpu_verification_prelaunch.json')
    C.B.require(len(expected)==222 and sources()==expected
        and clearance['status']=='cleared_before_bounded_GPU_TRAIN_verification'
        and clearance['source_manifest_sha256']==C.B.sha(C.HERE/'gpu_verification_source_manifest_prelaunch.json')
        and clearance['cpu_training_verification_sha256']==cpu_proof(),
        'GPU verification preflight incomplete')
    return expected


def full_launch_ready():
    expected=_cpu_full_ready()
    proof=C.B.P.read(C.ROOT/'gpu_training_verification.json')
    launch=C.B.P.read(C.HERE/'prelaunch_verification.json')
    C.B.require(expected==sources() and len(expected)==222
        and proof['status']==GPU_STATUS and proof['sources']==expected
        and proof['gpu_verification_manifest_sha256']==C.B.sha(C.HERE/'gpu_verification_source_manifest_prelaunch.json')
        and proof['cpu_training_verification_sha256']==cpu_proof()
        and launch['gpu_training_verification_sha256']==C.B.sha(C.ROOT/'gpu_training_verification.json')
        and proof['candidate_full_snapshot_resume_bitwise_equal'] is True
        and proof['ordinary_initial_and_5updates_full_snapshot_bitwise_equal'] is True,
        'GPU full launch preflight incomplete')
    return expected


def setup_gpu(selector):
    C.cpu()  # CPU limits and not-yet-initialized check precede CUDA initialization.
    C.B.require(selector==GPU_SELECTOR,'only physical GPU1 permitted')
    os.environ['CUDA_VISIBLE_DEVICES']=selector
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    C.torch.use_deterministic_algorithms(True)
    C.torch.backends.cudnn.benchmark=False
    C.torch.backends.cudnn.deterministic=True
    C.torch.backends.cuda.matmul.allow_tf32=False
    C.torch.backends.cudnn.allow_tf32=False
    C.B.require(C.torch.cuda.is_available() and C.torch.cuda.device_count()==1,'GPU1 unavailable or visibility differs')
    C.torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
    C.torch.cuda.reset_peak_memory_stats()
    return gpu_resource()


def gpu_resource():
    C.B.require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_SELECTOR
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        and C.torch.are_deterministic_algorithms_enabled()
        and not C.torch.backends.cuda.matmul.allow_tf32
        and not C.torch.backends.cudnn.allow_tf32
        and C.torch.backends.cudnn.deterministic and not C.torch.backends.cudnn.benchmark,
        'GPU numerical route changed')
    total=C.torch.cuda.get_device_properties(0).total_memory
    allocated=C.torch.cuda.max_memory_allocated();reserved=C.torch.cuda.max_memory_reserved()
    C.B.require(allocated<=total*MEMORY_FRACTION and reserved<=total*MEMORY_FRACTION,
        'GPU memory cap exceeded')
    return {'physical_selector':GPU_SELECTOR,'memory_fraction':MEMORY_FRACTION,
        'device_name':C.torch.cuda.get_device_name(0),'cublas_workspace_config':':4096:8',
        'deterministic_algorithms':True,'allow_tf32':False,'cudnn_deterministic':True,
        'cudnn_benchmark':False,'total_device_bytes':total,
        'max_allocated_bytes':allocated,'max_reserved_bytes':reserved}


C.sources=sources
C.full_launch_ready=full_launch_ready
C.RECIPES=('original32','current32','empirical32','training_control32','covariance32')
