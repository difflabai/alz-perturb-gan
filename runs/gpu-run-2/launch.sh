#!/bin/sh
# Exact command that produced runs/gpu-run-2 on pop-os (2026-10-13, started ~14:20 MT).
# Repo checkout: /ml2/altgan/decision-label-fix at d5f7da5.
# The shared venv has no pip, so `pip install -e` was impossible and its editable
# install points at /ml2/altgan/real-data-run-1. PYTHONPATH makes this checkout win.
# A first launch WITHOUT PYTHONPATH (which imported the old real-data-run-1 code) was
# killed during data loading and its run dir and log were deleted before this relaunch.
cd /ml2/altgan/decision-label-fix && \
PYTHONPATH=/ml2/altgan/decision-label-fix/src TMPDIR=/ml2/altgan/.cache XDG_CACHE_HOME=/ml2/altgan/.cache \
NUMBA_CACHE_DIR=/ml2/altgan/.cache/numba MPLCONFIGDIR=/ml2/altgan/.cache/matplotlib \
nohup /ml2/altgan/.venv/bin/python -u -m alz_perturb_gan.real_data \
  --data /ml2/altgan/data/adata_Training.h5ad --input-scale counts --device cuda --seed 42 \
  --steps 20000 --batch-size 256 --hidden 128 --lr 0.0002 --train-cells 128 --eval-cells 128 \
  --eval-fraction 0.2 --draws 64 --cell-eval --run-dir /ml2/altgan/runs/gpu-run-2/main \
  > /ml2/altgan/runs/gpu-run-2/main.log 2>&1 < /dev/null &
