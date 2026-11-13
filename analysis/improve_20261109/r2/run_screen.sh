#!/usr/bin/env bash
# improve_20261109 round-2 screen launcher (copy of ../run_screen.sh, same interface).
#
# Launches a list of (arm, panel, fold, seed[, modes]) fits, each as its own
# r2/trainer.py process with a thread cap, at most MAX_PARALLEL at a time,
# round-robin over a device list.  Completed fits (generated.npz +
# fit_receipt.json with status "completed") are not retrained; an interrupted
# fit (recovery.pt) is resumed by trainer.py automatically; final.pt without
# generated.npz runs generation only.  After a fit is complete, every
# generation mode of the optional 5th column whose generated_{mode}.npz is
# missing is run with r2/generate.py --mode (modes never retrain).
# Exits non-zero if any fit or mode fails.
#
# Usage:
#   r2/run_screen.sh -d DEVICES -p MAX_PARALLEL -j JOBFILE [options]
#
#   -d DEVICES      comma list; each entry is "cpu" or "cuda:N" (N = physical
#                   GPU index, exported as CUDA_VISIBLE_DEVICES=N for that
#                   process).  Slot i uses DEVICES[i % len(DEVICES)].
#   -p MAX_PARALLEL maximum concurrent fits (slots)
#   -j JOBFILE      one fit per line: "arm panel fold seed [modes]"
#                   (whitespace separated; '#' comments and blank lines ignored;
#                   arm = comma-joined flags, each "name" or "name@value" with
#                   canonical value text, or "base"; fold 0-8; modes = optional
#                   comma list such as snapens,snap@16000,margcov)
#   -t THREADS      CPU threads per process (default 3; round-2 maximum 3)
#   -u UPDATES      updates per fit (default 20000)
#   -o OUT_ROOT     fit root (default /ml2/alz/runs/improve_20261109/fits_r2)
#   -c CACHES_ROOT  cache root (default /ml2/alz/runs/improve_20261109/caches)
#   -l LOG_DIR      log dir (default /ml2/alz/runs/improve_20261109/logs_r2)
#   -g FRACTION     --gpu-mem-fraction passed to trainer.py for cuda fits
#   -n              dry run: print what would run, launch nothing
#
# Example job file:
#   base                    hvg2000 0 0  snapens
#   genemean@0.5            hvg2000 3 1  snapens,snap@19000
#   covars,genemean,offset  hvg2000 0 0
set -uo pipefail

# The whole script is one function, parsed completely before it runs, so editing
# this file while a screen is running cannot change the running instance.
main() {
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
RUNS=/ml2/alz/runs/improve_20261109
DEVICES="" ; MAXP="" ; JOBFILE="" ; THREADS=3 ; UPDATES=20000
OUT_ROOT="$RUNS/fits_r2" ; CACHES="$RUNS/caches" ; LOGDIR="$RUNS/logs_r2" ; GPUFRAC="" ; DRY=0

usage() { sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }
while getopts "d:p:j:t:u:o:c:l:g:nh" opt; do
  case $opt in
    d) DEVICES=$OPTARG ;; p) MAXP=$OPTARG ;; j) JOBFILE=$OPTARG ;; t) THREADS=$OPTARG ;;
    u) UPDATES=$OPTARG ;; o) OUT_ROOT=$OPTARG ;; c) CACHES=$OPTARG ;; l) LOGDIR=$OPTARG ;;
    g) GPUFRAC=$OPTARG ;; n) DRY=1 ;; *) usage ;;
  esac
done
[[ -n $DEVICES && -n $MAXP && -n $JOBFILE ]] || usage
[[ -f $JOBFILE ]] || { echo "job file not found: $JOBFILE" >&2; exit 2; }
# Repair (2026-11-10, review FID-2): never write fits into a frozen round-1 tree
# (trainer.py check_out_root refuses it too).
_ro=$(realpath -m -- "$OUT_ROOT")
for _fz in fits results results_screen48 screen_analysis caches; do
  _fp=$(realpath -m -- "$RUNS/$_fz")
  if [[ $_ro == "$_fp" || $_ro == "$_fp"/* ]]; then echo "refusing -o $_ro: inside frozen round-1 tree $_fp" >&2; exit 2; fi
done
[[ $MAXP =~ ^[1-9][0-9]*$ ]] || { echo "-p must be a positive integer" >&2; exit 2; }
[[ $THREADS =~ ^[1-9][0-9]*$ ]] && (( THREADS <= 3 )) || { echo "-t must be 1..3 (round-2 thread cap)" >&2; exit 2; }
IFS=',' read -r -a DEVS <<< "$DEVICES"
for d in "${DEVS[@]}"; do
  [[ $d == cpu || $d =~ ^cuda:[0-9]+$ ]] || { echo "bad device '$d' (use cpu or cuda:N)" >&2; exit 2; }
done

FLAGRE='^[a-z][a-z0-9_]*(@[^,@[:space:]]+)?$'
# Parse and validate the job list (arm flags are validated again by trainer.py).
JOBS=()
while IFS= read -r line || [[ -n $line ]]; do
  line="${line%%#*}"; arm="" ; modes="" ; extra=""
  read -r arm panel fold seed modes extra <<< "$line" || true
  [[ -z ${arm:-} ]] && continue
  if [[ -z ${seed:-} || -n ${extra:-} || ! $fold =~ ^[0-8]$ || ! $seed =~ ^[0-9]+$ || ! $panel =~ ^(hvg2000|wide)$ ]]; then
    echo "bad job line: '$line' (want: arm panel fold seed [modes])" >&2; exit 2
  fi
  if [[ $arm != base ]]; then
    IFS=',' read -r -a _fl <<< "$arm"
    for f in "${_fl[@]}"; do [[ $f =~ $FLAGRE ]] || { echo "bad arm flag '$f' in '$line'" >&2; exit 2; }; done
  fi
  if [[ -n $modes ]]; then
    IFS=',' read -r -a _ml <<< "$modes"
    for f in "${_ml[@]}"; do [[ $f =~ $FLAGRE ]] || { echo "bad mode '$f' in '$line'" >&2; exit 2; }; done
  fi
  [[ -d $CACHES/$panel/fold$fold ]] || { echo "missing cache $CACHES/$panel/fold$fold" >&2; exit 2; }
  JOBS+=("$arm $panel $fold $seed ${modes:--}")
done < "$JOBFILE"
NJOBS=${#JOBS[@]}
(( NJOBS > 0 )) || { echo "no jobs in $JOBFILE" >&2; exit 2; }
mkdir -p "$LOGDIR"
STATE=$(mktemp -d "$LOGDIR/.screen.XXXXXX")
echo 0 > "$STATE/next"
: > "$STATE/failed"; : > "$STATE/done"; : > "$STATE/skipped"
echo "[screen-r2] $NJOBS jobs, devices=${DEVICES} max_parallel=$MAXP threads=$THREADS updates=$UPDATES out=$OUT_ROOT"

is_complete() {  # fit dir -> 0 if generated.npz + completed receipt
  local d=$1
  [[ -f $d/generated.npz && -f $d/fit_receipt.json ]] && grep -q '"status": "completed"' "$d/fit_receipt.json"
}

canon_arm() {  # trainer's arm directory name (flags sorted in byte order, as Python sorted())
  local a=$1
  [[ $a == base ]] && { echo base; return; }
  tr ',' '\n' <<< "$a" | sed '/^$/d' | LC_ALL=C sort | paste -sd, -
}

run_logged() {  # log cmd... ; runs cmd with the slot environment, appends to log, returns rc
  local log=$1; shift
  echo "### $(date -Is) ${*}" >> "$log"
  CUDA_VISIBLE_DEVICES="$cvd" OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    PYTHONDONTWRITEBYTECODE=1 "$@" >> "$log" 2>&1 &
  local pid=$! rc
  echo "$pid" > "$STATE/pid.$slot"
  wait "$pid"; rc=$?
  while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done  # wait returns early on a trapped signal
  rm -f "$STATE/pid.$slot"
  echo "### exit $rc at $(date -Is)" >> "$log"
  return $rc
}

worker() {
  slot=$1
  local dev=${DEVS[$(( $1 % ${#DEVS[@]} ))]}
  local idx job arm panel fold seed modes fitdir tag log rc devflag mode mfail
  # Survive INT/TERM sent to the whole process group: keep waiting for the running
  # process (trainer.py checkpoints on INT/TERM) and record its exit status.
  trap 'touch "$STATE/stop"' INT TERM
  while [[ ! -f $STATE/stop ]]; do
    idx=$(flock "$STATE/next" bash -c 'n=$(cat "$0"); echo $((n+1)) > "$0"; echo $n' "$STATE/next")
    (( idx < NJOBS )) || break
    job=${JOBS[$idx]}; read -r arm panel fold seed modes <<< "$job"
    [[ $modes == - ]] && modes=""
    arm=$(canon_arm "$arm")
    fitdir="$OUT_ROOT/$arm/$panel/fold$fold/seed$seed"
    tag="${arm//,/+}_${panel}_fold${fold}_seed${seed}"
    log="$LOGDIR/$tag.log"
    if [[ $dev == cpu ]]; then devflag=cpu; cvd=""; else devflag=cuda; cvd=${dev#cuda:}; fi
    if is_complete "$fitdir"; then
      if ! grep -q "\"updates\": $UPDATES," "$fitdir/config.json"; then
        echo "[screen-r2] FAILED: $fitdir is complete with a different --updates than $UPDATES" >&2
        echo "$job rc=updates-mismatch" >> "$STATE/failed"; continue
      fi
      echo "[screen-r2] fit complete: $job"; fitstatus=skipped
    else
      cmd=("$PY" -B "$HERE/trainer.py" --cache "$CACHES/$panel/fold$fold" --arm "$arm" --seed "$seed"
           --out-root "$OUT_ROOT" --updates "$UPDATES" --device "$devflag" --threads "$THREADS")
      [[ $devflag == cuda && -n $GPUFRAC ]] && cmd+=(--gpu-mem-fraction "$GPUFRAC")
      if (( DRY )); then
        echo "[screen-r2] slot $slot CUDA_VISIBLE_DEVICES='$cvd' ${cmd[*]} > $log"
        fitstatus=dry
      else
        echo "[screen-r2] $(date +%H:%M:%S) slot $slot $dev start $job -> $log"
        echo "### host=$(hostname) slot=$slot device=$dev job='$job'" >> "$log"
        run_logged "$log" "${cmd[@]}"; rc=$?
        if (( rc == 0 )) && is_complete "$fitdir" && ! grep -q "\"updates\": $UPDATES," "$fitdir/config.json"; then
          echo "[screen-r2] $(date +%H:%M:%S) FAILED: $fitdir completed with a different --updates than $UPDATES" >&2
          echo "$job rc=updates-mismatch" >> "$STATE/failed"; continue
        elif ! { (( rc == 0 )) && is_complete "$fitdir"; }; then
          echo "[screen-r2] $(date +%H:%M:%S) FAILED (rc=$rc) $job, see $log" >&2; echo "$job rc=$rc" >> "$STATE/failed"; continue
        fi
        fitstatus=done
      fi
    fi
    mfail=0
    if [[ -n $modes ]]; then
      IFS=',' read -r -a ML <<< "$modes"
      for mode in "${ML[@]}"; do
        [[ -f $STATE/stop ]] && { mfail=1; break; }
        if [[ -f $fitdir/generated_$mode.npz ]]; then echo "[screen-r2] mode $mode present: $job"; continue; fi
        mcmd=("$PY" -B "$HERE/generate.py" --fit "$fitdir" --cache "$CACHES/$panel/fold$fold" --mode "$mode"
              --device "$devflag" --threads "$THREADS")
        if (( DRY )); then echo "[screen-r2] slot $slot CUDA_VISIBLE_DEVICES='$cvd' ${mcmd[*]} >> $log"; continue; fi
        run_logged "$log" "${mcmd[@]}"; rc=$?
        if (( rc != 0 )) || [[ ! -f $fitdir/generated_$mode.npz ]]; then
          echo "[screen-r2] $(date +%H:%M:%S) FAILED mode $mode (rc=$rc) $job, see $log" >&2; mfail=1
        fi
      done
    fi
    if (( mfail )); then echo "$job rc=mode-failed" >> "$STATE/failed"
    elif [[ $fitstatus == skipped ]]; then echo "$job" >> "$STATE/skipped"
    elif [[ $fitstatus == done ]]; then echo "[screen-r2] $(date +%H:%M:%S) done $job"; echo "$job" >> "$STATE/done"
    fi
  done
}

on_signal() {  # forward to running processes: trainers checkpoint at an update boundary and exit
  echo "[screen-r2] signal received; asking running fits to checkpoint and stop" >&2
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
echo "[screen-r2] finished: $nd completed, $ns skipped (already complete), $nf failed, of $NJOBS"
status=0
if (( nf > 0 )); then echo "[screen-r2] failed jobs:" >&2; cat "$STATE/failed" >&2; status=1; fi
if (( DRY == 0 && nd + ns + nf < NJOBS )); then
  echo "[screen-r2] $(( NJOBS - nd - ns - nf )) jobs not run (interrupted)" >&2; status=1
fi
[[ -f $STATE/stop ]] && status=130
rm -rf "$STATE"
return $status
}

main "$@"
exit $?
