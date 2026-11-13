#!/usr/bin/env bash
# improve_20261109 screen launcher.
#
# Launches a list of (arm, panel, fold, seed) fits, each as its own trainer.py
# process with a thread cap, at most MAX_PARALLEL at a time, round-robin over a
# device list.  Completed fits (generated.npz + fit_receipt.json with status
# "completed") are skipped; an interrupted fit (recovery.pt) is resumed by
# trainer.py automatically; final.pt without generated.npz runs generation only.
# Exits non-zero if any fit fails.
#
# Usage:
#   run_screen.sh -d DEVICES -p MAX_PARALLEL -j JOBFILE [options]
#
#   -d DEVICES      comma list; each entry is "cpu" or "cuda:N" (N = physical
#                   GPU index, exported as CUDA_VISIBLE_DEVICES=N for that
#                   process).  Slot i uses DEVICES[i % len(DEVICES)].
#                   Example: -d cuda:0,cuda:1   or   -d cpu
#   -p MAX_PARALLEL maximum concurrent fits (slots)
#   -j JOBFILE      one fit per line: "arm panel fold seed"
#                   (whitespace separated; '#' comments and blank lines ignored;
#                   arm is the comma-joined sorted flag list or "base")
#   -t THREADS      CPU threads per process (default 6; protocol maximum 6)
#   -u UPDATES      updates per fit (default 20000)
#   -o OUT_ROOT     fit root (default /ml2/alz/runs/improve_20261109/fits)
#   -c CACHES_ROOT  cache root (default /ml2/alz/runs/improve_20261109/caches)
#   -l LOG_DIR      log dir (default /ml2/alz/runs/improve_20261109/logs)
#   -g FRACTION     --gpu-mem-fraction passed to trainer.py for cuda fits
#   -n              dry run: print what would run, launch nothing
#
# Example job file:
#   base                    hvg2000 0 0
#   covars,genemean,offset  hvg2000 0 0
#   offset                  wide    0 0
set -uo pipefail

# The whole script is one function, parsed completely before it runs, so editing
# this file while a screen is running cannot change the running instance.
main() {
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
RUNS=/ml2/alz/runs/improve_20261109
DEVICES="" ; MAXP="" ; JOBFILE="" ; THREADS=6 ; UPDATES=20000
OUT_ROOT="$RUNS/fits" ; CACHES="$RUNS/caches" ; LOGDIR="$RUNS/logs" ; GPUFRAC="" ; DRY=0

usage() { sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }
while getopts "d:p:j:t:u:o:c:l:g:nh" opt; do
  case $opt in
    d) DEVICES=$OPTARG ;; p) MAXP=$OPTARG ;; j) JOBFILE=$OPTARG ;; t) THREADS=$OPTARG ;;
    u) UPDATES=$OPTARG ;; o) OUT_ROOT=$OPTARG ;; c) CACHES=$OPTARG ;; l) LOGDIR=$OPTARG ;;
    g) GPUFRAC=$OPTARG ;; n) DRY=1 ;; *) usage ;;
  esac
done
[[ -n $DEVICES && -n $MAXP && -n $JOBFILE ]] || usage
[[ -f $JOBFILE ]] || { echo "job file not found: $JOBFILE" >&2; exit 2; }
[[ $MAXP =~ ^[1-9][0-9]*$ ]] || { echo "-p must be a positive integer" >&2; exit 2; }
[[ $THREADS =~ ^[1-9][0-9]*$ ]] && (( THREADS <= 6 )) || { echo "-t must be 1..6 (protocol thread cap)" >&2; exit 2; }
IFS=',' read -r -a DEVS <<< "$DEVICES"
for d in "${DEVS[@]}"; do
  [[ $d == cpu || $d =~ ^cuda:[0-9]+$ ]] || { echo "bad device '$d' (use cpu or cuda:N)" >&2; exit 2; }
done

# Parse and validate the job list.
JOBS=()
while IFS= read -r line || [[ -n $line ]]; do
  line="${line%%#*}"; read -r arm panel fold seed extra <<< "$line" || true
  [[ -z ${arm:-} ]] && continue
  if [[ -z ${seed:-} || -n ${extra:-} || ! $fold =~ ^[0-2]$ || ! $seed =~ ^[0-9]+$ || ! $panel =~ ^(hvg2000|wide)$ ]]; then
    echo "bad job line: '$line' (want: arm panel fold seed)" >&2; exit 2
  fi
  [[ -d $CACHES/$panel/fold$fold ]] || { echo "missing cache $CACHES/$panel/fold$fold" >&2; exit 2; }
  JOBS+=("$arm $panel $fold $seed")
done < "$JOBFILE"
NJOBS=${#JOBS[@]}
(( NJOBS > 0 )) || { echo "no jobs in $JOBFILE" >&2; exit 2; }
mkdir -p "$LOGDIR"
STATE=$(mktemp -d "$LOGDIR/.screen.XXXXXX")
echo 0 > "$STATE/next"
: > "$STATE/failed"; : > "$STATE/done"; : > "$STATE/skipped"
echo "[screen] $NJOBS jobs, devices=${DEVICES} max_parallel=$MAXP threads=$THREADS updates=$UPDATES out=$OUT_ROOT"

is_complete() {  # fit dir -> 0 if generated.npz + completed receipt
  local d=$1
  [[ -f $d/generated.npz && -f $d/fit_receipt.json ]] && grep -q '"status": "completed"' "$d/fit_receipt.json"
}

canon_arm() {  # trainer's arm directory name (sorted flags)
  local a=$1
  [[ $a == base ]] && { echo base; return; }
  tr ',' '\n' <<< "$a" | sed '/^$/d' | sort | paste -sd, -
}

worker() {
  local slot=$1 dev=${DEVS[$(( $1 % ${#DEVS[@]} ))]}
  local idx job arm panel fold seed fitdir tag log rc devflag cvd
  # Survive INT/TERM sent to the whole process group: keep waiting for the running
  # trainer (which checkpoints on INT/TERM) and record its exit status. A no-op trap
  # (not an ignore) so the trainer starts with default dispositions.
  trap 'touch "$STATE/stop"' INT TERM
  while [[ ! -f $STATE/stop ]]; do
    idx=$(flock "$STATE/next" bash -c 'n=$(cat "$0"); echo $((n+1)) > "$0"; echo $n' "$STATE/next")
    (( idx < NJOBS )) || break
    job=${JOBS[$idx]}; read -r arm panel fold seed <<< "$job"
    arm=$(canon_arm "$arm")
    fitdir="$OUT_ROOT/$arm/$panel/fold$fold/seed$seed"
    tag="${arm//,/+}_${panel}_fold${fold}_seed${seed}"
    log="$LOGDIR/$tag.log"
    if is_complete "$fitdir"; then
      if grep -q "\"updates\": $UPDATES," "$fitdir/config.json"; then
        echo "[screen] skip (complete) $job"; echo "$job" >> "$STATE/skipped"
      else
        echo "[screen] FAILED: $fitdir is complete with a different --updates than $UPDATES" >&2
        echo "$job rc=updates-mismatch" >> "$STATE/failed"
      fi
      continue
    fi
    if [[ $dev == cpu ]]; then devflag=cpu; cvd=""; else devflag=cuda; cvd=${dev#cuda:}; fi
    cmd=("$PY" -B "$HERE/trainer.py" --cache "$CACHES/$panel/fold$fold" --arm "$arm" --seed "$seed"
         --out-root "$OUT_ROOT" --updates "$UPDATES" --device "$devflag" --threads "$THREADS")
    [[ $devflag == cuda && -n $GPUFRAC ]] && cmd+=(--gpu-mem-fraction "$GPUFRAC")
    if (( DRY )); then
      echo "[screen] slot $slot CUDA_VISIBLE_DEVICES='$cvd' ${cmd[*]} > $log"; continue
    fi
    echo "[screen] $(date +%H:%M:%S) slot $slot $dev start $job -> $log"
    {
      echo "### $(date -Is) host=$(hostname) slot=$slot device=$dev job='$job'"
      echo "### ${cmd[*]}"
    } >> "$log"
    CUDA_VISIBLE_DEVICES="$cvd" OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
      PYTHONDONTWRITEBYTECODE=1 "${cmd[@]}" >> "$log" 2>&1 &
    local pid=$!
    echo "$pid" > "$STATE/pid.$slot"
    wait "$pid"; rc=$?
    while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done  # wait returns early on a trapped signal
    rm -f "$STATE/pid.$slot"
    echo "### exit $rc at $(date -Is)" >> "$log"
    if (( rc == 0 )) && is_complete "$fitdir" && ! grep -q "\"updates\": $UPDATES," "$fitdir/config.json"; then
      echo "[screen] $(date +%H:%M:%S) FAILED: $fitdir completed with a different --updates than $UPDATES" >&2
      echo "$job rc=updates-mismatch" >> "$STATE/failed"
    elif (( rc == 0 )) && is_complete "$fitdir"; then
      echo "[screen] $(date +%H:%M:%S) done $job"; echo "$job" >> "$STATE/done"
    else
      echo "[screen] $(date +%H:%M:%S) FAILED (rc=$rc) $job, see $log" >&2; echo "$job rc=$rc" >> "$STATE/failed"
    fi
  done
}

on_signal() {  # forward to running trainers: they checkpoint at an update boundary and exit
  echo "[screen] signal received; asking running fits to checkpoint and stop" >&2
  touch "$STATE/stop"
  for f in "$STATE"/pid.*; do [[ -f $f ]] && kill -TERM "$(cat "$f")" 2>/dev/null; done
}
trap on_signal INT TERM

NSLOTS=$(( MAXP < NJOBS ? MAXP : NJOBS ))
WPIDS=()
for (( s=0; s<NSLOTS; s++ )); do worker "$s" & WPIDS+=($!); done
for p in "${WPIDS[@]}"; do
  while kill -0 "$p" 2>/dev/null; do wait "$p" 2>/dev/null; done  # wait returns early on a trapped signal
done

nd=$(wc -l < "$STATE/done"); ns=$(wc -l < "$STATE/skipped"); nf=$(wc -l < "$STATE/failed")
echo "[screen] finished: $nd completed, $ns skipped (already complete), $nf failed, of $NJOBS"
status=0
if (( nf > 0 )); then echo "[screen] failed jobs:" >&2; cat "$STATE/failed" >&2; status=1; fi
if (( DRY == 0 && nd + ns + nf < NJOBS )); then
  echo "[screen] $(( NJOBS - nd - ns - nf )) jobs not run (interrupted)" >&2; status=1
fi
[[ -f $STATE/stop ]] && status=130
rm -rf "$STATE"
return $status
}

main "$@"
exit $?
