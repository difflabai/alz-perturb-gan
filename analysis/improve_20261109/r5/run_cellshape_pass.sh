#!/usr/bin/env bash
set -euo pipefail
R=/ml2/alz/runs/improve_20261109
S=/ml2/alz/analysis/improve_20261109/r5
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
mkdir -p "$R/logs_r5/default_pass" "$R/smoke_r5/default_pass/modes"
# Regenerate zcal references through the neutral path for complete spread guards;
# the historical compact zcal score files omitted pc_spread_rep.
for pair in '0 0 0' '0 1 0' '1 0 0' '1 1 0' '2 0 0' '2 1 0' \
            '0 0 0.5' '0 1 0.5' '1 0 0.5' '1 1 0.5' '2 0 0.5' '2 1 0.5'; do
    read -r fold seed alpha <<< "$pair"
    out="$R/smoke_r5/default_pass/modes/fold${fold}_seed${seed}_alpha${alpha}.npz"
    if [[ ! -f "$out" ]]; then
        "$PY" -B "$S/mode_cellshape.py" --fit "$R/fits_r3/pbulk/hvg2000/fold$fold/seed$seed" \
            --cache "$R/caches/hvg2000/fold$fold" --out "$out" --device cuda --gpu-mem-fraction .04 \
            --mode "cellshape@$alpha:biascal:stagehalf" \
            > "$R/logs_r5/default_pass/fold${fold}_seed${seed}_alpha${alpha}.log" 2>&1
    fi
    printf 'Generated fold %s seed %s alpha %s\n' "$fold" "$seed" "$alpha"
done
export CUDA_VISIBLE_DEVICES=''
"$PY" -B "$S/evaluate.py" --fits-root "r3=$R/fits_r3" --modes-root "$R/smoke_r5/default_pass/modes" \
    --arm 'pbulk~r3#cellshape*' --panels hvg2000 --folds 0,1,2 --seeds 0,1 \
    --out "$R/results_r5/default_pass" --no-baselines --no-ensembles \
    > "$R/logs_r5/default_pass/evaluate.log" 2>&1
if grep -q '"msg": "ERROR' "$R/logs_r5/default_pass/evaluate.log"; then exit 1; fi
printf 'Candidate generation and evaluation completed.\n'
