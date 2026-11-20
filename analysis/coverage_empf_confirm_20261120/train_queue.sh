#!/bin/bash
# Pinned-core training queue for the confirmation fits (PROTOCOL.md).
# scripts/microglia_runtime.py pins each process to the first three cores it is
# allowed, so unpinned parallel launches all land on cores 0-2. Each worker w
# runs its jobs under `taskset -c 3w..3w+2`, which the runtime keeps.
# Usage: train_queue.sh JOBFILE [WAITPIDS]   (JOBFILE lines: "fold seed";
# WAITPIDS: comma list, worker w first waits for the w-th pid to exit)
JOBS=$1; WAIT=${2:-}; W=8
ROOT=/ml2/alz; LOG=$ROOT/runs/coverage_empf_confirm_20261120/logs
LOCK=$LOG/queue.lock; START=$LOG/start.lock; IDX=$LOG/queue.next
[ -f "$IDX" ] || echo 1 > "$IDX"
IFS=, read -ra WP <<< "$WAIT"
run_one() {  # cores fold seed
  local c=$1 k=$2 s=$3 rc tries=0 tag
  while :; do
    tries=$((tries+1)); tag=fold${k}_seed${s}.try$tries
    rm -f $LOG/rc_$tag
    ( flock 8; echo "$(date -Is) start fold$k seed$s cores$c try$tries" >> $LOG/queue.log
      cd $ROOT
      bash -c "CUDA_VISIBLE_DEVICES=$(( (${c%-*} / 3) % 2 )) taskset -c $c /mnt/nvme/ai/conda/anaconda3/bin/python3 -B scripts/microglia.py train --fold $k --seed $s --device cuda > $LOG/train_$tag.log 2>&1; echo \$? > $LOG/rc_$tag" 8>&- &
      echo $! > $LOG/pid_$tag; sleep 25 ) 8> $START
    while kill -0 $(cat $LOG/pid_$tag) 2>/dev/null; do sleep 30; done
    rc=$(cat $LOG/rc_$tag 2>/dev/null || echo missing)
    if [ "$rc" != 0 ] && grep -q "existing derived EMPF host differs" $LOG/train_$tag.log && [ $tries -lt 3 ]; then
      echo "$(date -Is) retry fold$k seed$s rc=$rc (activation race)" >> $LOG/queue.log; continue
    fi
    echo "$(date -Is) end fold$k seed$s rc=$rc" >> $LOG/queue.log; return
  done
}
worker() {
  local w=$1 c="$(( $1*3 ))-$(( $1*3+2 ))"
  [ -n "${WP[$w]:-}" ] && while kill -0 ${WP[$w]} 2>/dev/null; do sleep 30; done
  while :; do
    line=$(flock $LOCK bash -c 'n=$(cat '"$IDX"'); l=$(sed -n "${n}p" '"$JOBS"'); [ -n "$l" ] && echo $((n+1)) > '"$IDX"'; echo "$l"')
    [ -z "$line" ] && return
    run_one $c $line
  done
}
for w in $(seq 0 $((W-1))); do worker $w & done; wait
echo "$(date -Is) queue2 finished" >> $LOG/queue.log
