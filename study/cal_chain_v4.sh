#!/usr/bin/env bash
# Plan v4.2 chain: one supervisor process (all seeds, no aggregate time cap), then the post-run probe.
# The harness commit, bundle and protocol are HARD-PINNED below (the deployed copy has the commit filled in);
# they are never taken from the command line. Exits nonzero if the supervisor or the probe fails.
# Usage: cal_chain_v4_2.sh --dry-run
#        cal_chain_v4_2.sh <authorization-ref> <ownership-ref> <lease-valid-until ISO-8601 with tz> [--arm-timeout S] [--total-timeout S]
set -euo pipefail
shopt -s nullglob
umask 022
SCOPE=/ml2/altgan
S=/ml2/altgan/studies/gene-feature-20261014
PY=/ml2/altgan/.venv/bin/python
CACHE=/ml2/altgan/.cache
PINNED_COMMIT=@PINNED_COMMIT@
PINNED_BUNDLE_SHA=60d5f71a60c47489febf4b0060ec6eed525b5b9b82a9c1e3c3cc7b1140f42686
PINNED_PROTOCOL_SHA=8db7d79597ac7311741349b968bb467337eb53e21d4a7fdf46ce9f80beafdd2a
SUPERVISOR_NAME=supervise_v4_2.py
in_scope() { local r; r=$(realpath -e -- "$1") || return 1; [[ "$r" == "$SCOPE" || "$r" == "$SCOPE"/* ]] || { echo "out of scope: $1" >&2; return 1; }; printf '%s\n' "$r"; }
[[ "$PINNED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit is not pinned in this copy" >&2; exit 2; }
[[ $# -ge 1 ]] || { echo "usage: $0 --dry-run | AUTH OWN LEASE [extra supervisor timeout flags]" >&2; exit 2; }
SUP=$(in_scope "$S/$SUPERVISOR_NAME")
BUNDLE=$(in_scope "$S/study_bundle_v2.pt")
cd -- "$(in_scope "$CACHE")"
export CUDA_VISIBLE_DEVICES= TMPDIR="$CACHE" XDG_CACHE_HOME="$CACHE" PYTHONDONTWRITEBYTECODE=1
echo "== v4.2 chain $(date --iso-8601=seconds) commit $PINNED_COMMIT supervisor sha $(sha256sum -- "$SUP" | cut -c1-64)"
PIN=(--harness-commit "$PINNED_COMMIT" --bundle "$BUNDLE" --bundle-sha256 "$PINNED_BUNDLE_SHA" --protocol-sha256 "$PINNED_PROTOCOL_SHA")
RC=0
if [[ "$1" == "--dry-run" ]]; then
  OUT=$("$PY" "$SUP" --dry-run --seeds 42,43,44 "${PIN[@]}") || RC=$?
else
  [[ $# -ge 3 ]] || { echo "need AUTH OWN LEASE" >&2; exit 2; }
  AUTH=$1; OWN=$2; LEASE=$3; shift 3
  unset CUDA_VISIBLE_DEVICES
  OUT=$("$PY" "$SUP" --seeds 42,43,44 "${PIN[@]}" --authorization-reference "$AUTH" --ownership-reference "$OWN" --lease-valid-until "$LEASE" "$@") || RC=$?
fi
echo "supervisor rc=$RC"
printf '%s\n' "$OUT"
REC=$(printf '%s\n' "$OUT" | "$PY" -c 'import json,sys
lines=[l for l in sys.stdin.read().splitlines() if l.startswith("{")]
print(json.loads(lines[-1])["receipt"])') || { echo "no supervisor receipt line; supervisor rc=$RC" >&2; exit $(( RC ? RC : 4 )); }
REC=$(in_scope "$REC")
RSHA=$(sha256sum -- "$REC" | cut -c1-64)
PRC=0
"$PY" "$SUP" --post-run-probe --supervisor-receipt "$REC" --sha256 "$RSHA" || PRC=$?
echo "probe rc=$PRC"
FILES=("$(dirname -- "$REC")"/*.json)
[[ ${#FILES[@]} -gt 0 ]] || { echo "no receipts found" >&2; exit 3; }
sha256sum -- "${FILES[@]}"
if [[ $RC -ne 0 || $PRC -ne 0 ]]; then echo "CHAIN_FAILED supervisor=$RC probe=$PRC" >&2; exit $(( RC ? RC : PRC )); fi
echo CHAIN_DONE
