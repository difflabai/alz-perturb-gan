#!/bin/bash
# Part 2 queue: one lane of runs, sequential. Usage: synth_queue.sh WIDTH DEVICE "SEEDS"
W=$1; DEV=$2; SEEDS=$3
O=/ml2/alz/runs/research_20261120/cond-residual/synthetic
mkdir -p $O
export PYTHONPATH=/ml2/alz/artifacts/native-releases/particlegan-0.10.0/site-packages
cd "$(dirname "$0")"
for s in $SEEDS; do for arm in native adapter; do
  f=$O/w${W}_${arm}_s${s}.json
  [ -f $f ] && continue
  taskset -c 3-23 /mnt/nvme/ai/conda/anaconda3/bin/python3 -B synthetic.py train --arm $arm --seed $s --width $W \
    --updates 2000 --device $DEV --out $f >> $O/queue_w${W}.log 2>&1
done; done
