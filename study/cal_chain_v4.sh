#!/usr/bin/env bash
# Plan v4.6 chain: one supervisor process (all seeds, no aggregate time cap), then the post-run probe.
# The harness commit, bundle and protocol are HARD-PINNED below (the deployed copy has the commit filled in);
# they are never taken from the command line. Exits nonzero if the supervisor or the probe fails.
# Usage: setsid cal_chain_v4_6.sh --dry-run   (or under systemd-run --user)
#        cal_chain_v4_6.sh <authorization-ref> <ownership-ref> <lease-valid-until ISO-8601 with tz> [--arm-timeout S] [--total-timeout S] [--teardown-timeout S]
# Extra args are ALLOWLISTED: only the three timeout flags, each at most once, as '--flag N' with N a positive number.
# Anything else (including any pinned option such as --harness-commit/--bundle/--bundle-sha256/--protocol-sha256/--seeds,
# '--flag=value' forms, duplicates, or extras in --dry-run) is rejected before the supervisor starts.
set -euo pipefail
shopt -s nullglob
umask 022
# v4.4: a hangup never kills the wrapper (the supervisor treats SIGHUP like SIGTERM and still writes its receipt).
trap '' HUP
# v4.5: SIGTERM (e.g. systemctl stop, which signals every process of the unit) is trapped so the receipt and the
# probe still run; the chain is then ALWAYS labelled incomplete. Only an unconditional CHAIN_DONE means complete.
GOT_TERM=
trap 'GOT_TERM=1' TERM
# v4.4 launch guard: never run attached to a terminal. Allowed: a systemd unit (INVOCATION_ID and a .service
# cgroup), or a session without a controlling terminal (setsid). A controlling terminal is refused even for a
# session leader, because a hangup signals the session leader first.
STAT=$(< "/proc/$$/stat"); STAT=${STAT##*") "}
read -r _ _ _ SID TTY_NR _ <<< "$STAT"
in_unit() { [[ -n "${INVOCATION_ID:-}" ]] && grep -qE '[.]service$' "/proc/$$/cgroup"; }
if in_unit; then LAUNCH=systemd-unit
elif [[ "$TTY_NR" != 0 ]]; then
  echo "refusing to start: controlling terminal present (tty_nr=$TTY_NR); launch with systemd-run --user or setsid, detached from any terminal" >&2; exit 2
elif [[ "$SID" == "$$" ]]; then LAUNCH=setsid
else LAUNCH=no-ctty
fi
SCOPE=/ml2/altgan
S=/ml2/altgan/studies/gene-feature-20261014
PY=/ml2/altgan/.venv/bin/python
CACHE=/ml2/altgan/.cache
PINNED_COMMIT=@PINNED_COMMIT@
PINNED_BUNDLE_SHA=60d5f71a60c47489febf4b0060ec6eed525b5b9b82a9c1e3c3cc7b1140f42686
PINNED_PROTOCOL_SHA=8db7d79597ac7311741349b968bb467337eb53e21d4a7fdf46ce9f80beafdd2a
SUPERVISOR_NAME=supervise_v4_6.py
in_scope() { local r; r=$(realpath -e -- "$1") || return 1; [[ "$r" == "$SCOPE" || "$r" == "$SCOPE"/* ]] || { echo "out of scope: $1" >&2; return 1; }; printf '%s\n' "$r"; }
[[ "$PINNED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit is not pinned in this copy" >&2; exit 2; }
[[ $# -ge 1 ]] || { echo "usage: $0 --dry-run | AUTH OWN LEASE [extra supervisor timeout flags]" >&2; exit 2; }
SUP=$(in_scope "$S/$SUPERVISOR_NAME")
BUNDLE=$(in_scope "$S/study_bundle_v2.pt")
cd -- "$(in_scope "$CACHE")"
export CUDA_VISIBLE_DEVICES= TMPDIR="$CACHE" XDG_CACHE_HOME="$CACHE" PYTHONDONTWRITEBYTECODE=1
echo "== v4.6 chain $(date --iso-8601=seconds) commit $PINNED_COMMIT supervisor sha $(sha256sum -- "$SUP" | cut -c1-64) launch=$LAUNCH"
PIN=(--harness-commit "$PINNED_COMMIT" --bundle "$BUNDLE" --bundle-sha256 "$PINNED_BUNDLE_SHA" --protocol-sha256 "$PINNED_PROTOCOL_SHA")
RC=0
if [[ "$1" == "--dry-run" ]]; then
  [[ $# -eq 1 ]] || { echo "--dry-run takes no extra arguments" >&2; exit 2; }
  OUT=$("$PY" "$SUP" --dry-run --seeds 42,43,44 "${PIN[@]}") || RC=$?
else
  [[ $# -ge 3 ]] || { echo "need AUTH OWN LEASE" >&2; exit 2; }
  AUTH=$1; OWN=$2; LEASE=$3; shift 3
  EXTRA=(); SEEN=" "
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --arm-timeout|--total-timeout|--teardown-timeout) ;;
      *) echo "rejected argument: $1 (only --arm-timeout/--total-timeout/--teardown-timeout N are allowed)" >&2; exit 2 ;;
    esac
    [[ "$SEEN" != *" $1 "* ]] || { echo "duplicate argument: $1" >&2; exit 2; }
    [[ $# -ge 2 && "$2" =~ ^[0-9]+([.][0-9]+)?$ && ! "$2" =~ ^0+([.]0+)?$ ]] || { echo "$1 needs a positive number" >&2; exit 2; }
    SEEN="$SEEN$1 "; EXTRA+=("$1" "$2"); shift 2
  done
  unset CUDA_VISIBLE_DEVICES
  OUT=$("$PY" "$SUP" --seeds 42,43,44 "${PIN[@]}" --authorization-reference "$AUTH" --ownership-reference "$OWN" --lease-valid-until "$LEASE" "${EXTRA[@]}") || RC=$?
fi
echo "supervisor rc=$RC"
printf '%s\n' "$OUT"
PARSED=$(printf '%s\n' "$OUT" | "$PY" -c 'import json,sys
rec = coh = None
for l in sys.stdin.read().splitlines():
    if not l.startswith("{"): continue
    try: d = json.loads(l)
    except ValueError: continue
    if isinstance(d, dict) and d.get("receipt"): rec = d["receipt"]
    if isinstance(d, dict) and d.get("event") == "start": coh = d.get("cohort")
print("R " + rec if rec else "C " + coh if coh else "N -")') || PARSED="N -"
KIND=${PARSED%% *}; WHERE=${PARSED#* }
REASON=""
[[ $RC -eq 0 ]] || REASON+="supervisor_rc=$RC "
[[ $RC -ne 137 ]] || REASON+="supervisor_sigkill(manual_or_oom_kill;cause_unknown) "
PRC=0
# v4.7: perform potentially slow probes and hashing while TERM is still accepted.
# The probe writes only a provisional observation, never a clean terminal receipt.
if [[ "$KIND" == R ]]; then
  REC=$(in_scope "$WHERE")
  RSHA=$(sha256sum -- "$REC" | cut -c1-64)
  "$PY" "$SUP" --post-run-probe --supervisor-receipt "$REC" --sha256 "$RSHA" --defer-probe-finalization || PRC=$?
  DIR=$(dirname -- "$REC")
elif [[ "$KIND" == C ]]; then
  REASON+="no_supervisor_receipt "
  DIR=$(in_scope "$WHERE")
  "$PY" "$SUP" --post-run-probe --cohort "$DIR" --defer-probe-finalization || PRC=$?
else
  REASON+="no_supervisor_receipt no_cohort "; DIR=""
fi
OBS=""; OSHA=""
if [[ -n "$DIR" ]]; then
  FILES=("$DIR"/*.json)
  [[ ${#FILES[@]} -eq 0 ]] || sha256sum -- "${FILES[@]}"
  OBS=$(in_scope "$DIR/post_run_probe_observation.json") || { [[ $PRC -ne 0 ]] || PRC=8; }
  [[ -z "$OBS" ]] || OSHA=$(sha256sum -- "$OBS" | cut -c1-64)
fi
# Cancellation commit boundary: all accepted TERM signals are now recorded.
# Later signals concern an already finalized chain. No slow probe runs below.
trap '' TERM
STOPARG=()
[[ -z "$GOT_TERM" ]] || STOPARG=(--stop-requested wrapper_sigterm)
if [[ -n "$OBS" && -n "$OSHA" && $PRC -eq 0 ]]; then
  "$PY" "$SUP" --finalize-probe --probe-observation "$OBS" --sha256 "$OSHA" "${STOPARG[@]}" || PRC=$?
fi
echo "probe rc=$PRC"
[[ $PRC -eq 0 ]] || REASON+="probe_rc=$PRC "
[[ -z "$GOT_TERM" ]] || REASON+="wrapper_sigterm(stop_requested) "
if [[ -n "$REASON" ]]; then
  echo "CHAIN_INCOMPLETE supervisor=$RC probe=$PRC reason=${REASON% }"
  echo "CHAIN_INCOMPLETE supervisor=$RC probe=$PRC reason=${REASON% }" >&2
  exit $(( RC ? RC : (PRC ? PRC : 8) ))
fi
echo CHAIN_DONE
