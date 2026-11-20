#!/bin/bash
# Score every confirmation fit as it finishes training, two scorers at a time.
# score.py skips nothing itself; this loop skips fits that already have score.json.
ROOT=/ml2/alz; R=$ROOT/runs/coverage_empf_confirm_20261120; LOG=$R/logs
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
cd $ROOT
declare -A running fails
while :; do
  pending=0
  for s in 0 1; do for k in 0 1 2 3 4 5 6 7 8; do
    [ "$k$s" = 00 ] && continue
    [ -f $R/scores/fold$k/seed$s/score.json ] && continue
    [ "${fails[$k$s]:-0}" -ge 2 ] && continue
    pending=$((pending+1))
    key=$k$s; pid=${running[$key]:-}
    if [ -n "$pid" ]; then kill -0 $pid 2>/dev/null && continue; unset running[$key]; fails[$key]=$(( ${fails[$key]:-0} + 1 ))
      echo "$(date -Is) scorer fold$k seed$s exited without score.json, attempt ${fails[$key]} of 2 (see score_fold${k}_seed${s}.log)" >> $LOG/score_queue.log; continue; fi
    n=0; for p in "${running[@]}"; do kill -0 $p 2>/dev/null && n=$((n+1)); done
    [ $n -ge 2 ] && continue
    grep -q training_completed $ROOT/runs/microglia_empf/fits/pbulk/hvg2000/fold$k/seed$s/fit_receipt.json 2>/dev/null || continue
    ls $ROOT/runs/microglia_empf/generated/fold$k/seed$s/*.npz >/dev/null 2>&1 || continue
    taskset -c $(( 12 + 3*n ))-$(( 14 + 3*n )) /mnt/nvme/ai/conda/anaconda3/bin/python3 -B analysis/coverage_empf_confirm_20261120/score.py --fold $k --seed $s >> $LOG/score_fold${k}_seed${s}.log 2>&1 &
    running[$key]=$!; echo "$(date -Is) start scorer fold$k seed$s pid $!" >> $LOG/score_queue.log
  done; done
  [ $pending -eq 0 ] && break
  sleep 120
done
echo "$(date -Is) scoring loop done; failed: ${!fails[*]}" >> $LOG/score_queue.log
CUDA_VISIBLE_DEVICES='' /mnt/nvme/ai/conda/anaconda3/bin/python3 -B analysis/coverage_empf_confirm_20261120/analyze.py > $R/analysis_final.txt 2>&1
echo "$(date -Is) analyze exit $?" >> $LOG/score_queue.log
