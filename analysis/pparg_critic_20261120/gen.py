"""Generate a readout archive for a finished fit with the application generation streams.

python -B gen.py --fit FIT --mode snapens|cellshape@0.5:biascal:stagehalf --out OUT.npz [--fold 0]
"""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
import argparse
import json
from pathlib import Path
import host_pparg as H


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fit', required=True, type=Path)
    p.add_argument('--mode', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--fold', type=int, default=0)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists')
    T, G, prov = H.activate()
    H.deterministic(T, 'cpu', None)
    H.app_streams(T, G)
    _, receipt = G.generate_fit(a.fit, H.CACHE/f'fold{a.fold}', mode=a.mode, out=a.out, device='cpu')
    receipt.update(generation_streams=[20261114, 6, a.fold, 'development donor index', 0], provenance=prov)
    (a.out.with_name(a.out.stem + '.receipt.json')).write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
