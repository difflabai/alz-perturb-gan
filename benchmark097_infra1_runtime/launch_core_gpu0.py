"""Proposed exact finite GPU0 owner; no admission activation or successor."""
import argparse
import fcntl
import os
from pathlib import Path
import sys

# Supported invocation is Python-I-S-B; only this exact source directory is added.
sys.path.insert(0,str(Path(__file__).resolve().parent))
import binding as b
import project_runner_gpu0

def run(auth_path,layout=b.NATIVE_LAYOUT,*,observe=None):
    auth,_=b.verify(auth_path,layout)
    from dual_coordination import execution_locks
    with execution_locks(layout.science,auth):
        b.verify(auth_path,layout)
        os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',WANDB_DISABLED='true',DO_NOT_TRACK='1',OTEL_SDK_DISABLED='true',MPLCONFIGDIR=str(layout.science.parent/'isolated/mpl-cache'),NUMBA_CACHE_DIR=str(layout.science.parent/'isolated/numba-cache'))
        return project_runner_gpu0.run(auth_path,layout,observe=observe)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path,required=True)
    args=parser.parse_args()
    return run(args.authorization)

if __name__=='__main__':raise SystemExit(main())
