#!/usr/bin/env bash
# Resume the documented Block-A evaluation without changing training or frozen outputs.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
R=/ml2/alz/runs/improve_20261109
R5=/ml2/alz/analysis/improve_20261109/r5
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
OUT=$R/results_r5/pzero
mkdir -p "$OUT/fits" "$R/logs_r5"
"$PY" -B - "$R" "$OUT" <<'PY'
import pathlib, shutil, sys
r, out = map(pathlib.Path, sys.argv[1:])
for name in ('baselines', 'real'):
    shutil.copytree(r / 'results_r3' / name, out / name, dirs_exist_ok=True)
shutil.copy2(r / 'results_r3/floor.json', out / 'floor.json')
for path in (r / 'results_r3/fits').iterdir():
    if any(path.name.startswith(arm + '~r3') for arm in
           ('pbulk', 'pbulk,t.kappa@0.25', 'g.dispz,pbulk,t.kappa@0.25')):
        shutil.copytree(path, out / 'fits' / path.name, dirs_exist_ok=True)
PY
SEL=(--fits-root "r3=$R/fits_r3" --fits-root "r3=$R/fits_r5"
     --modes-root "$R/smoke_r5/pzero/modes_r3" --out "$OUT"
     --panels hvg2000 --seeds 0,1
     --arm 'pbulk~r3*' --arm 'pbulk,t.kappa@0.25~r3*'
     --arm 'pbulk,t.kappa@0.19~r3*' --arm '*pzero*'
     --arm 'g.dispz,pbulk,t.kappa@0.25~r3*' --arm 'g.dispz,pbulk,t.kappa@0.19~r3*'
     --ensemble-arm '*#snapens' --ensemble-arm '*#stagehalf'
     --ensemble-arm '*#stageblind' --ensemble-arm '*#biascal@stagehalf')
pids=()
for k in 0 1 2; do
    "$PY" -B "$R5/evaluate.py" "${SEL[@]}" --folds "$k" \
        > "$R/logs_r5/eval_pzero_fold$k.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status )); then exit "$status"; fi
# evaluate.py records caught fit failures in its log and can still exit zero.
if grep -q '"msg": "ERROR' "$R"/logs_r5/eval_pzero_fold[012].log; then
    printf 'Evaluator errors found; inspect the fold logs.\n' >&2
    exit 1
fi
"$PY" -B "$R5/evaluate.py" "${SEL[@]}" --folds 0,1,2 \
    > "$R/logs_r5/eval_pzero_all.log" 2>&1
if grep -q '"msg": "ERROR' "$R/logs_r5/eval_pzero_all.log"; then
    printf 'Evaluator errors found; inspect eval_pzero_all.log.\n' >&2
    exit 1
fi
"$PY" -B "$R5/pzero_compare.py" --results "$OUT" --folds 0,1,2 --seeds 0,1 \
    > "$R/logs_r5/pzero_compare.txt"
printf 'Evaluation and paired comparison finished: %s\n' "$OUT/pzero_compare.json"
