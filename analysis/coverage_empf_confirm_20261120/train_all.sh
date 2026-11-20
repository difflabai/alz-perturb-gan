#!/bin/bash
# Train the 17 pre-declared native EMPF fits (PROTOCOL.md) through the committed CLI.
# Usage: train_all.sh [max_parallel]; resumes or skips fits the CLI finds complete.
P=${1:-8}
ROOT=/ml2/alz; LOG=$ROOT/runs/coverage_empf_confirm_20261120/logs
jobs=""
for s in 0 1; do for k in 0 1 2 3 4 5 6 7 8; do
  [ "$k$s" = "00" ] && continue
  jobs="$jobs $k:$s"
done; done
i=0
for js in $jobs; do echo "$js $((i % 2))"; i=$((i+1)); done | \
xargs -P "$P" -L 1 bash -c '
  ks=$0; gpu=$1; k=${ks%:*}; s=${ks#*:}
  cd /ml2/alz
  echo "$(date -Is) start fold$k seed$s gpu$gpu" >> '"$LOG"'/queue.log
  CUDA_VISIBLE_DEVICES=$gpu /mnt/nvme/ai/conda/anaconda3/bin/python3 -B scripts/microglia.py train --fold $k --seed $s --device cuda > '"$LOG"'/train_fold${k}_seed${s}.log 2>&1
  rc=$?
  echo "$(date -Is) end fold$k seed$s rc=$rc" >> '"$LOG"'/queue.log
'
echo "$(date -Is) queue finished" >> $LOG/queue.log
